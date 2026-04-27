# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Learner server runner for distributed HILSerl robot policy training.

This script implements the learner component of the distributed HILSerl architecture.
It initializes the policy network, maintains replay buffers, and updates
the policy based on transitions received from the actor server.

Examples of usage:

- Start a learner server for training:
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: Start the learner server before launching the actor server. The learner opens a gRPC server
to communicate with actors.

**NOTE**: Training progress can be monitored through Weights & Biases if wandb.enable is set to true
in your configuration.

**WORKFLOW**:
1. Create training configuration with proper policy, dataset, and environment settings
2. Start this learner server with the configuration
3. Start an actor server with the same configuration
4. Monitor training progress through wandb dashboard

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

# from frrl.robots.franka_real.cameras import opencv  # 真机才需要
from frrl.configs import parser
from frrl.configs.train import TrainRLServerPipelineConfig
from frrl.datasets.factory import make_dataset
from frrl.datasets.lerobot_dataset import LeRobotDataset
from frrl.policies.factory import make_policy
from frrl.policies.sac.modeling_sac import SACPolicy
from frrl.rl.core.buffer import ReplayBuffer, concatenate_batch_transitions
from frrl.rl.infra.process import ProcessSignalHandler
from frrl.rl.infra.wandb_utils import WandBLogger
# from frrl.robots import so100_follower  # 真机才需要
from frrl.teleoperators.utils import TeleopEvents
from frrl.rl.infra.transport import services_pb2_grpc
from frrl.rl.infra.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from frrl.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from frrl.utils.random_utils import set_seed
from frrl.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from frrl.utils.transition import move_state_dict_to_device, move_transition_to_device
from frrl.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initializes and runs the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration
        job_name (str | None, optional): Job name for logging. Defaults to None.
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from frrl.rl.infra.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    Start the learner threads for training.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        wandb_logger (WandBLogger | None): Logger for metrics
        shutdown_event: Event to signal shutdown
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    # Pretrain-only mode: no actor will connect, so we don't spawn the gRPC server.
    # The training function still runs but its main online loop returns early after
    # offline pretrain completes.
    communication_process = None
    if not cfg.policy.offline_only_mode:
        communication_process = concurrency_entity(
            target=start_learner,
            args=(
                parameters_queue,
                transition_queue,
                interaction_message_queue,
                shutdown_event,
                cfg,
            ),
            daemon=True,
        )
        communication_process.start()
    else:
        logging.info(
            "[LEARNER] offline_only_mode=True: skipping gRPC actor server "
            "(no online actor expected)"
        )

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    if communication_process is not None:
        communication_process.join()
        logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


# Core algorithm functions


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    Handles data transfer from the actor to the learner, manages training updates,
    and logs training progress in an online reinforcement learning setup.

    This function continuously:
    - Transfers transitions from the actor to the replay buffer.
    - Logs received interaction messages.
    - Ensures training begins only when the replay buffer has a sufficient number of transitions.
    - Samples batches from the replay buffer and performs multiple critic updates.
    - Periodically updates the actor, critic, and temperature optimizers.
    - Logs training statistics, including loss values and optimization frequency.

    NOTE: This function doesn't have a single responsibility, it should be split into multiple functions
    in the future. The reason why we did that is the  GIL in Python. It's super slow the performance
    are divided by 200. So we need to have a single thread that does all the work.

    Args:
        cfg (TrainRLServerPipelineConfig): Configuration object containing hyperparameters.
        wandb_logger (WandBLogger | None): Logger for tracking training progress.
        shutdown_event (Event): Event to signal shutdown.
        transition_queue (Queue): Queue for receiving transitions from the actor.
        interaction_message_queue (Queue): Queue for receiving interaction messages from the actor.
        parameters_queue (Queue): Queue for sending policy parameters to the actor.
    """
    # Extract all configuration variables at the beginning, it improve the speed performance
    # of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    # Load dataset metadata before initializing policy to get correct normalization stats
    ds_meta = None
    if cfg.dataset is not None and not cfg.resume:
        logging.info("Loading dataset metadata for policy initialization")
        from frrl.datasets.lerobot_dataset import LeRobotDatasetMetadata
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        logging.info(f"Dataset stats loaded: {list(ds_meta.stats.keys())}")

    logging.info("Initializing policy")

    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        ds_meta=ds_meta,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    # Initialize offline buffer whenever there's an offline data source:
    #   - cfg.dataset 非空 → 走 LeRobotDataset 路径
    #   - cfg.policy.demo_pickle_paths 非空 → 走 pickle adapter 路径（两种模式共用）
    # 注意 demo_pickle_paths 对两种模式都生效：
    #   * offline_only_mode=True  → 跑 pretrain 后退出
    #   * offline_only_mode=False → warmup 后进入 online 主循环，RLPD 50/50 mix
    needs_offline_buffer = cfg.dataset is not None or bool(cfg.policy.demo_pickle_paths)
    # 防呆：warmup 需要跑但没有数据源时直接报错，避免静默跳过 warmup 浪费真机 run。
    # 有效 warmup_steps 取决于 offline_only_mode：True→pretrain_steps，False→warmup_steps。
    effective_warmup_steps = (
        cfg.policy.offline_pretrain_steps
        if cfg.policy.offline_only_mode
        else cfg.policy.offline_warmup_steps
    )
    if effective_warmup_steps > 0 and not needs_offline_buffer and not cfg.resume:
        raise ValueError(
            f"warmup_steps={effective_warmup_steps}>0 but no offline source: "
            f"offline_only_mode={cfg.policy.offline_only_mode}, "
            f"demo_pickle_paths={cfg.policy.demo_pickle_paths}, dataset={cfg.dataset}. "
            "Set cfg.dataset or fill cfg.policy.demo_pickle_paths."
        )
    if needs_offline_buffer:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
        # RLPD 50/50 混合：online/offline 各采一半，batch 保持不变。
        batch_size: int = batch_size // 2

    logging.info("Starting learner thread")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # ================================================================
    # 热启动 / Pretrain-only：在 offline demo 上预训练
    #   offline_only_mode=True  → 跑 offline_pretrain_steps 步，然后保存退出
    #   offline_only_mode=False → 跑 offline_warmup_steps 步，然后进入主 online loop
    # ================================================================
    if cfg.policy.offline_only_mode:
        warmup_steps = cfg.policy.offline_pretrain_steps
    else:
        warmup_steps = cfg.policy.offline_warmup_steps
    if offline_replay_buffer is not None and not cfg.resume and warmup_steps > 0:
        logging.info(f"[LEARNER] Warmup: pre-training on offline demo for {warmup_steps} steps...")
        offline_warmup_iter = offline_replay_buffer.get_iterator(
            batch_size=batch_size * 2, async_prefetch=async_prefetch, queue_size=2
        )
        policy.train()
        for i in range(warmup_steps):
            batch = next(offline_warmup_iter)
            observations = batch["state"]
            next_observations = batch["next_state"]

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            forward_batch = {
                ACTION: batch[ACTION],
                "reward": batch["reward"],
                "state": observations,
                "next_state": next_observations,
                "done": batch["done"],
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                # Keep warmup consistent with the main loop so discrete_penalty
                # (gripper-toggle penalty) is actually applied during warmup.
                "complementary_info": batch.get("complementary_info"),
            }

            # Critic
            loss_critic = policy.forward(forward_batch, model="critic")["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value)
            optimizers["critic"].step()

            # Discrete critic
            if policy.config.num_discrete_actions is not None:
                loss_dc = policy.forward(forward_batch, model="discrete_critic")["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_dc.backward()
                torch.nn.utils.clip_grad_norm_(policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value)
                optimizers["discrete_critic"].step()

            # Actor + temperature
            if i % policy_update_freq == 0:
                loss_actor = policy.forward(forward_batch, model="actor")["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), max_norm=clip_grad_norm_value)
                optimizers["actor"].step()

                # P0-2: pretrain 期默认冻 log_alpha。demo 上 actor 很快高似然，
                # log_probs+target_entropy 大概率 > 0 → log_alpha 被推向 -∞，
                # α 坍塌到 1e-3 以下后 entropy bonus 消失、真机零探索。
                if not policy.config.freeze_temperature_in_pretrain:
                    loss_temp = policy.compute_loss_temperature(observations, observation_features)
                    optimizers["temperature"].zero_grad()
                    loss_temp.backward()
                    optimizers["temperature"].step()
                    policy.update_temperature()

            policy.update_target_networks()
            optimization_step += 1

            if (i + 1) % 100 == 0:
                logging.info(f"[LEARNER] Warmup step {i+1}/{warmup_steps}, critic_loss={loss_critic.item():.4f}")

        push_actor_policy_to_queue(parameters_queue, policy)
        logging.info(f"[LEARNER] Warmup complete. Pushed initial policy to Actor.")

    # Pretrain-only mode: save checkpoint and exit before the main online loop.
    # The saved checkpoint is later loaded as a resume point when the online
    # actor+learner pipeline starts (阶段 6).
    if cfg.policy.offline_only_mode:
        logging.info(
            f"[LEARNER] offline_only_mode: pretraining complete at step "
            f"{optimization_step}, saving checkpoint and exiting"
        )
        if saving_checkpoint:
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )
            logging.info(f"[LEARNER] Pretrain checkpoint saved at step {optimization_step}")
        # Close wandb cleanly — without this the run stays "crashed" in the UI
        # when wandb is enabled, since we return before the main loop's cleanup.
        if wandb_logger is not None:
            try:
                import wandb
                wandb.finish()
            except Exception as e:
                logging.warning(f"wandb.finish() failed: {e}")
        return

    # Initialize iterators
    online_iterator = None
    offline_iterator = None

    # NOTE: THIS IS THE MAIN LOOP OF THE LEARNER
    while True:
        # Exit the training loop if shutdown is requested
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Saving checkpoint before exiting...")
            # Save checkpoint when manually stopping training (Ctrl+C)
            if optimization_step > 0:
                save_training_checkpoint(
                    cfg=cfg,
                    optimization_step=optimization_step,
                    online_steps=online_steps,
                    interaction_message=interaction_message,
                    policy=policy,
                    optimizers=optimizers,
                    replay_buffer=replay_buffer,
                    offline_replay_buffer=offline_replay_buffer,
                    dataset_repo_id=dataset_repo_id,
                    fps=fps,
                )
                logging.info(f"[LEARNER] Checkpoint saved at step {optimization_step}")
                # 退出时保存replay buffer（用于resume）
                try:
                    logging.info("[LEARNER] Saving online replay buffer...")
                    dataset_path = os.path.join(cfg.output_dir, "dataset")
                    replay_buffer.to_lerobot_dataset(
                        repo_id=dataset_repo_id or "frrl/online_buffer",
                        fps=fps,
                        root=dataset_path,
                    )
                    logging.info(f"[LEARNER] Online buffer saved to {dataset_path}")
                    if offline_replay_buffer is not None:
                        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
                        offline_replay_buffer.to_lerobot_dataset(
                            repo_id=dataset_repo_id or "frrl/offline_buffer",
                            fps=fps,
                            root=dataset_offline_path,
                        )
                        logging.info(f"[LEARNER] Offline buffer saved to {dataset_offline_path}")
                except Exception as e:
                    logging.warning(f"[LEARNER] Failed to save replay buffer: {e}")
            logging.info("[LEARNER] Exiting...")
            break

        # Process all available transitions to the replay buffer, send by the actor server
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # Process all available interaction messages sent by the actor server
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # Wait until the replay buffer has enough samples to start training
        if len(replay_buffer) < online_step_before_learning:
            continue

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # Sample from the iterators
            batch = next(online_iterator)

            # Mix offline/prior buffer 50/50 whenever it exists — RLPD core.
            # Previously checked `dataset_repo_id is not None`, but pretrain-only
            # mode populates the offline buffer from pickle with `cfg.dataset=None`
            # (and thus `dataset_repo_id=None`), causing the mix to silently
            # disappear after resuming online. Use the buffer presence directly.
            if offline_replay_buffer is not None:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            actions = batch[ACTION]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            # P0-6: NaN 不能 silent skip。一条污染 transition 的梯度会把整个 critic
            # backward 出 NaN 权重，后面所有 update 都成 NaN。检测到立刻跳过这条 batch。
            if check_nan_in_transition(
                observations=observations, actions=actions, next_state=next_observations
            ):
                continue

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            # Create a batch dictionary with all required elements for the forward method
            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "complementary_info": batch["complementary_info"],
            }

            # Use the forward method for critic loss
            critic_output = policy.forward(forward_batch, model="critic")

            # Main critic optimization
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()

            # Discrete critic optimization (if available)
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            # Update target networks (main and discrete)
            policy.update_target_networks()

        # Sample for the last update in the UTD ratio
        batch = next(online_iterator)

        # Mix offline 50/50 here too — see UTD-sampling block above for why
        # this cannot key off dataset_repo_id.
        if offline_replay_buffer is not None:
            batch_offline = next(offline_iterator)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch[ACTION]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        # P0-6: 与 UTD-loop 内分支保持一致语义；最后一次 update 撞 NaN 则跳过整步。
        # 必须在 continue 前 ++step：否则连续 NaN 会让 step 卡住，save_freq 永远
        # 触发不到，wandb 进度条也停滞，违反 fail-loud 期望。
        if check_nan_in_transition(
            observations=observations, actions=actions, next_state=next_observations
        ):
            logging.warning(
                f"[LEARNER] NaN in transition at step {optimization_step}; skipping update."
            )
            optimization_step += 1
            continue

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # Create a batch dictionary with all required elements for the forward method
        forward_batch = {
            ACTION: actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
            # Must match the UTD-loop forward_batch above — discrete_critic reads
            # this to apply gripper-toggle penalties. Without it the final update
            # each step computes a discrete-critic loss that's inconsistent with
            # the utd_ratio-1 preceding updates.
            "complementary_info": batch.get("complementary_info"),
        }

        critic_output = policy.forward(forward_batch, model="critic")

        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        # Initialize training info dictionary
        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
        }

        # Discrete critic optimization (if available)
        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            # Add discrete critic info to training info
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        # Actor and temperature optimization (at specified frequency).
        # Phase 2 (critic-only warmup on real env)：_should_freeze_actor 为 True
        # 时跳过 actor/temperature 梯度更新，critic 继续训、target network 继续
        # 同步；actor 参数保持 pretrain 出来的状态。
        freeze_actor = _should_freeze_actor(cfg, len(replay_buffer))
        training_infos["freeze_actor"] = int(freeze_actor)
        if _should_run_actor_optimization(
            cfg, len(replay_buffer), optimization_step, policy_update_freq
        ):
            for _ in range(policy_update_freq):
                # Actor optimization
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # Add actor info to training info
                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm

                # Temperature optimization
                temperature_output = policy.forward(forward_batch, model="temperature")
                loss_temperature = temperature_output["loss_temperature"]
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                ).item()
                optimizers["temperature"].step()

                # Add temperature info to training info
                training_infos["loss_temperature"] = loss_temperature.item()
                training_infos["temperature_grad_norm"] = temp_grad_norm
                training_infos["temperature"] = policy.temperature

                # Update temperature
                policy.update_temperature()

        # Push policy to actors if needed.
        # Phase 2 期间 actor 参数不变（被 _should_freeze_actor 冻结），但 critic
        # target 每步都在更新；push 的 state_dict 是整个 policy，所以仍然正常发
        # 送——actor 侧拿到的 actor 参数和上一次一致，不会影响真机采样行为。
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # Update target networks (main and discrete)
        policy.update_target_networks()

        # Log training metrics at specified intervals
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # Log training metrics
            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # Calculate and log optimization frequency
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # Log optimization frequency
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # Save checkpoint at specified intervals
        # Always save at the end (when optimization_step == online_steps), regardless of save_checkpoint setting
        should_save_periodic = saving_checkpoint and (optimization_step % save_freq == 0)
        should_save_final = (optimization_step == online_steps)

        if should_save_periodic or should_save_final:
            if should_save_final:
                logging.info(f"[LEARNER] Training complete. Saving final checkpoint at step {optimization_step}")
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )
            if should_save_final:
                logging.info("[LEARNER] Training finished. Exiting...")
                break


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training.
    It will receive transitions and interaction messages from the actor server,
    and send policy parameters to the actor server.

    Args:
        parameters_queue: Queue for sending policy parameters to the actor
        transition_queue: Queue for receiving transitions from the actor
        interaction_message_queue: Queue for receiving interaction messages from the actor
        shutdown_event: Event to signal shutdown
        cfg: Training configuration
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        # Return back for MP
        # TODO: Check if its useful
        _ = ProcessSignalHandler(False, display_pid=True)

    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    Save training checkpoint and associated data.

    This function performs the following steps:
    1. Creates a checkpoint directory with the current optimization step
    2. Saves the policy model, configuration, and optimizer states
    3. Saves the current interaction step for resuming training
    4. Updates the "last" checkpoint symlink to point to this checkpoint
    5. Saves the replay buffer as a dataset for later use
    6. If an offline replay buffer exists, saves it as a separate dataset

    Args:
        cfg: Training configuration
        optimization_step: Current optimization step
        online_steps: Total number of online steps
        interaction_message: Dictionary containing interaction information
        policy: Policy model to save
        optimizers: Dictionary of optimizers
        replay_buffer: Replay buffer to save as dataset
        offline_replay_buffer: Optional offline replay buffer to save
        dataset_repo_id: Repository ID for dataset
        fps: Frames per second for dataset
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # Create checkpoint directory
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # Save checkpoint
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # Save interaction step manually
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # Update the "last" symlink
    update_last_checkpoint(checkpoint_dir)

    # FIXED: Removed replay buffer dataset saving to prevent training from hanging
    # Issue: replay_buffer.to_lerobot_dataset() was blocking the main training loop
    # for several minutes (or indefinitely), causing the learner to stop processing
    # new transitions from the actor after each checkpoint save.
    #
    # The checkpoint still saves all essential training state:
    # - Model weights (policy.safetensors)
    # - Optimizer states (for all optimizers)
    # - Training state (optimization_step, interaction_step)
    # This is sufficient for resuming training.
    #
    # Note: If you need to save the replay buffer as a dataset, you can do so
    # manually after training completes by calling:
    # replay_buffer.to_lerobot_dataset(repo_id, fps, root)

    logging.info("Resume training")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Creates and returns optimizers for the actor, critic, and temperature components of a reinforcement learning policy.

    This function sets up Adam optimizers for:
    - The **actor network**, ensuring that only relevant parameters are optimized.
    - The **critic ensemble**, which evaluates the value function.
    - The **temperature parameter**, which controls the entropy in soft actor-critic (SAC)-like methods.

    It also initializes a learning rate scheduler, though currently, it is set to `None`.

    NOTE:
    - If the encoder is shared, its parameters are excluded from the actor's optimization process.
    - The policy's log temperature (`log_alpha`) is wrapped in a list to ensure proper optimization as a standalone tensor.

    Args:
        cfg: Configuration object containing hyperparameters.
        policy (nn.Module): The policy model containing the actor, critic, and temperature components.

    Returns:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        A tuple containing:
        - `optimizers`: A dictionary mapping component names ("actor", "critic", "temperature") to their respective Adam optimizers.
        - `lr_scheduler`: Currently set to `None` but can be extended to support learning rate scheduling.

    """
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
    return optimizers, lr_scheduler


# Training setup functions


# Resume 时允许从新 JSON 覆盖的白名单字段。
# 动机：checkpoint 里冻结的 Phase 2 超参常常需要在 resume 后调整（比如把 critic_only
# 窗口从 500 调到 1000），但默认 from_pretrained 会把这些字段也覆盖回旧值。
# 只放训练阶段相关超参，不放模型 / 网络结构，避免 resume 加载 optimizer state 时错位。
# Phase 2 / cold-start 阈值：resume 时允许从命令行 / JSON 覆盖 ckpt 里的旧值。
# 注意：warmup-only 字段（如 freeze_temperature_in_pretrain）不进白名单——resume
# 路径在 learner.py 顶部就 bypass 了 warmup 分支，这类字段值不会被读到。
RESUMABLE_POLICY_OVERRIDES = (
    "critic_only_online_steps",
    "online_step_before_learning",
)


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic for training.

    If resume is True:
    - Verifies that a checkpoint exists
    - Loads the checkpoint configuration
    - Applies whitelisted overrides from the current cfg (Phase 2 knobs)
    - Logs resumption details
    - Returns the checkpoint configuration

    If resume is False:
    - Checks if an output directory exists (to prevent accidental overwriting)
    - Returns the original configuration

    Args:
        cfg (TrainRLServerPipelineConfig): The training configuration

    Returns:
        TrainRLServerPipelineConfig: The updated configuration

    Raises:
        RuntimeError: If resume is True but no checkpoint found, or if resume is False but directory exists
    """
    out_dir = cfg.output_dir

    # Case 1: Not resuming, but need to check if directory exists to prevent overwrites
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # Case 2: Resuming training
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # 白名单覆盖：把当前命令行/JSON 里的 Phase 2 超参写回 checkpoint cfg。
    # 只在值真的变化时打日志，避免 resume 正常路径刷屏。
    # 旧 ckpt 可能完全没有这些字段（Phase 2 引入之前的版本），用 sentinel default
    # 防 AttributeError，并强制落成新 cfg 提供的值。
    _MISSING = object()
    for field in RESUMABLE_POLICY_OVERRIDES:
        new_val = getattr(cfg.policy, field)
        old_val = getattr(checkpoint_cfg.policy, field, _MISSING)
        if old_val is _MISSING or new_val != old_val:
            shown_old = "<missing in ckpt>" if old_val is _MISSING else old_val
            logging.info(
                colored(
                    f"Resume override: policy.{field} {shown_old} -> {new_val}",
                    color="yellow",
                    attrs=["bold"],
                )
            )
            setattr(checkpoint_cfg.policy, field, new_val)

    # setattr 绕过了 __post_init__，必须再跑一次 Phase 2 契约校验：防止覆盖后
    # critic_only_online_steps <= online_step_before_learning 却没被捕获，导致
    # Phase 2 窗口被 cold-start 阈值吃空、wandb 显示 critic_only>0 却形同未启用。
    if hasattr(checkpoint_cfg.policy, "_validate_phase2"):
        checkpoint_cfg.policy._validate_phase2()

    # P1-5：resume 时 online buffer 内容不持久化（save_training_checkpoint 默认不存
    # online_replay_buffer 真机数据），所以 buffer 从 0 重填。`_should_freeze_actor`
    # 用 buffer 大小判断 Phase 2，故 resume 后会从头再走一次 critic-only warmup。
    # 这是预期行为（重新冷启动 + Phase 2 适配），但 wandb 上看像"为什么 critic_only
    # 又重新跑了"——明确写出避免操作者排查浪费时间。
    logging.info(
        colored(
            "Resume note: online buffer 不持久化、resume 后从 0 重填。"
            f" _should_freeze_actor 用 buffer 大小判断，"
            f" Phase 2（critic_only_online_steps={checkpoint_cfg.policy.critic_only_online_steps}）"
            " 会重新进入直到真机交互填满阈值。",
            color="cyan",
        )
    )

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    Loads the training state (optimizers, step count, etc.) from a checkpoint.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        optimizers (Optimizer | dict): Optimizers to load state into

    Returns:
        tuple: (optimization_step, interaction_step) or (None, None) if not resuming
    """
    if not cfg.resume:
        return None, None

    # Construct path to the last checkpoint directory
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # Use the utility function from train_utils which loads the optimizer state
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # Load interaction step separately from training_state.pt
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    Log information about the training process.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        policy (nn.Module): Policy model
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized replay buffer
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    # Resume时尝试加载之前的online dataset，如果不存在则从空buffer开始
    dataset_path = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(os.path.join(dataset_path, "meta", "info.json")):
        logging.info(f"Resume: loading online dataset from {dataset_path}")
        repo_id = cfg.dataset.repo_id if cfg.dataset is not None else None
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_path)
        return ReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=dataset,
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            optimize_memory=True,
        )
    else:
        logging.info("Resume: no online dataset found, starting with empty online buffer")
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )


def _should_freeze_actor(cfg: TrainRLServerPipelineConfig, replay_buffer_size: int) -> bool:
    """HIL-SERL critic-only 阶段判断。

    `critic_only_online_steps > 0` 时，online buffer 真机 transition 数还不够阈值，
    就冻结 actor / temperature（只训 critic）。进 Phase 3 后 actor 解冻，恢复
    正常 SAC 联合更新。0 或负数都视为关闭（SACConfig.__post_init__ 已把负数
    规范化到 0），此处的 `n > 0` 只是兜底。
    """
    n = cfg.policy.critic_only_online_steps
    return n > 0 and replay_buffer_size < n


def _should_run_actor_optimization(
    cfg: TrainRLServerPipelineConfig,
    replay_buffer_size: int,
    optimization_step: int,
    policy_update_freq: int,
) -> bool:
    """主循环 actor/temperature 本步是否跑。

    两件事同时满足：
      1. 不在 Phase 2 critic-only 冻结窗口内
      2. 到 policy_update_freq 整周期
    抽成 helper 是为了让 freeze_actor 的守卫有一个可直接 mock 测的接口，
    防止主循环被重构时误把 Phase 2 冻结逻辑绕过去。
    """
    if _should_freeze_actor(cfg, replay_buffer_size):
        return False
    return optimization_step % policy_update_freq == 0


def _validate_resize_map(raw: dict) -> dict:
    """Coerce demo_resize_images list→tuple + validate every value is [H, W]."""
    out = {}
    for k, v in raw.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(
                f"demo_resize_images[{k}] must be [H, W] (len 2); got {v!r}"
            )
        out[k] = (int(v[0]), int(v[1]))
    return out


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    Initialize an offline replay buffer from a dataset or from demo pickles.

    When `cfg.policy.demo_pickle_paths` is non-empty, the buffer is populated
    directly from hil-serl-format pickle files (see
    ReplayBuffer.from_pickle_transitions)。对 `offline_only_mode=True`（纯
    pretrain）和 `False`（HIL-SERL 混合模式，warmup 后 RLPD 50/50 mix）都适用。
    否则走常规 LeRobotDataset 路径。

    Args:
        cfg (TrainRLServerPipelineConfig): Training configuration
        device (str): Device to store tensors on
        storage_device (str): Device for storage optimization

    Returns:
        ReplayBuffer: Initialized offline replay buffer
    """
    # Pickle path: 两种模式都走这里，只要 demo_pickle_paths 非空。
    if cfg.policy.demo_pickle_paths:
        mode = "pretrain-only" if cfg.policy.offline_only_mode else "hil-serl mixed"
        logging.info(
            f"[LEARNER] {mode}: loading demos from "
            f"{len(cfg.policy.demo_pickle_paths)} pickle(s)"
        )
        state_keys = None
        if cfg.policy.input_features is not None:
            state_keys = list(cfg.policy.input_features.keys())
        return ReplayBuffer.from_pickle_transitions(
            pickle_paths=cfg.policy.demo_pickle_paths,
            capacity=cfg.policy.offline_buffer_capacity,
            device=device,
            storage_device=storage_device,
            state_keys=state_keys,
            # Disable optimize_memory for pickle path: avoids the "last
            # transition's next_state is the first transition's state" bug
            # when capacity == len(transitions) (all-episode-boundary confusion).
            optimize_memory=False,
            # DrQ aug assumes all image keys share the same (H,W), which held
            # under front=224 / wrist=128. Both are 128 now, but we keep it
            # disabled for offline path: demos are augmented separately via
            # processor pipeline if desired.
            use_drq=False,
            key_map=dict(cfg.policy.demo_key_map) if cfg.policy.demo_key_map else None,
            transpose_hwc_to_chw=list(cfg.policy.demo_transpose_hwc_to_chw)
                if cfg.policy.demo_transpose_hwc_to_chw else None,
            resize_images=_validate_resize_map(cfg.policy.demo_resize_images)
                if cfg.policy.demo_resize_images else None,
            normalize_to_unit=list(cfg.policy.demo_normalize_to_unit)
                if cfg.policy.demo_normalize_to_unit else None,
        )

    if not cfg.resume:
        logging.info("make_dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        # Resume时尝试加载保存的offline dataset，不存在则重新从demo加载
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(os.path.join(dataset_offline_path, "meta", "info.json")):
            logging.info(f"Resume: loading offline dataset from {dataset_offline_path}")
            offline_dataset = LeRobotDataset(
                repo_id=cfg.dataset.repo_id,
                root=dataset_offline_path,
            )
        else:
            logging.info("Resume: no saved offline dataset, reloading from demo")
            offline_dataset = make_dataset(cfg)

    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


# Utilities/Helpers functions


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Get observation features from the policy encoder. It act as cache for the observation features.
    when the encoder is frozen, the observation features are not updated.
    We can save compute by caching the observation features.

    Args:
        policy: The policy model
        observations: The current observations
        next_observations: The next observations

    Returns:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    Check for NaN values in transition data.

    Args:
        observations: Dictionary of observation tensors
        actions: Action tensor
        next_state: Dictionary of next state tensors
        raise_error: If True, raises ValueError when NaN is detected

    Returns:
        bool: True if NaN values were detected, False otherwise
    """
    nan_detected = False

    # Check observations
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # Check next state
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # Check actions
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """Process a single interaction message with consistent handling."""
    message = bytes_to_python_object(message)
    # Shift interaction step for consistency with checkpointed state
    message["Interaction step"] += interaction_step_shift

    # Log if logger available
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
):
    """Process all available transitions from the queue.

    Args:
        transition_queue: Queue for receiving transitions from the actor
        replay_buffer: Replay buffer to add transitions to
        offline_replay_buffer: Offline replay buffer to add transitions to
        device: Device to move transitions to
        dataset_repo_id: Repository ID for dataset
        shutdown_event: Event to signal shutdown
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)
        logging.info(f"[DEBUG] Received {len(transition_list)} transitions")


        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # Skip transitions with NaN values
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            is_intervention = transition.get("info", {}).get(TeleopEvents.IS_INTERVENTION.value) or \
                              transition.get("complementary_info", {}).get(TeleopEvents.IS_INTERVENTION.value)

            # 所有transition都进online buffer
            replay_buffer.add(**transition)

            if replay_buffer.size % 100 == 0:
                logging.info(f"[DEBUG] Replay buffer size: {len(replay_buffer)}")

            # 人类干预的数据额外加入 offline buffer
            # 之前 gate 用 `dataset_repo_id is not None` —— online HIL 配置
            # `dataset: null` 让 gate 永 False，干预永远不进 offline buffer，
            # HIL-SERL "human demos accumulate" 语义破坏。改成 mirror line 593
            # 的 fix：只要 offline buffer 存在（pickle 加载或 dataset 都行）就路由。
            if offline_replay_buffer is not None and is_intervention:
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """Process all available interaction messages from the queue.

    Args:
        interaction_message_queue: Queue for receiving interaction messages
        interaction_step_shift: Amount to shift interaction step by
        wandb_logger: Logger for tracking progress
        shutdown_event: Event to signal shutdown

    Returns:
        dict | None: The last interaction message processed, or None if none were processed
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
