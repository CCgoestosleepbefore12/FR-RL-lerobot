#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import functools
import pickle as pkl
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from frrl.datasets.lerobot_dataset import LeRobotDataset
from frrl.utils.constants import ACTION, DONE, OBS_IMAGE, REWARD
from frrl.utils.transition import Transition


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


def _to_batched_tensor(v: Any) -> torch.Tensor:
    """Convert numpy array, torch tensor, or python scalar to a batched tensor.

    Always adds exactly one leading batch dim:
      scalar     → (1,)
      shape (N,) → (1, N)
      shape (H,W,C) → (1, H, W, C)
    """
    if isinstance(v, torch.Tensor):
        t = v
    elif isinstance(v, np.ndarray):
        t = torch.from_numpy(v.copy())
    elif isinstance(v, (bool, int, float)):
        t = torch.tensor(v)
    else:
        raise TypeError(f"cannot convert {type(v).__name__} to tensor")
    return t.unsqueeze(0)


def _flatten_obs_to_tensor_dict(
    obs: Any, prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Recursively flatten a nested dict observation into flat-key torch tensors.

    ``obs["pixels"]["front"]`` → ``{"pixels.front": tensor}``. Leaves are converted
    via ``_to_batched_tensor``. A non-dict top-level ``obs`` is wrapped under the key
    ``"state"`` to match the fallback expected by the Transition schema.
    """
    if not isinstance(obs, dict):
        return {("state" if not prefix else prefix): _to_batched_tensor(obs)}
    out: dict[str, torch.Tensor] = {}
    for k, v in obs.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_obs_to_tensor_dict(v, key))
        else:
            out[key] = _to_batched_tensor(v)
    return out


def _require_keys(d: dict, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"transition dict missing required keys: {missing}")


def _remap_and_transform(
    flat_obs: dict[str, torch.Tensor],
    key_map: dict[str, str],
    transpose_set: set[str],
    resize_map: dict[str, tuple[int, int]] | None = None,
    normalize_set: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply key rename + optional HWC→CHW transpose + optional resize + optional /255 normalize.

    Called once per observation (and once per next_observation) during pickle
    loading. Pipeline per key:

      1. Rename via ``key_map`` (pickle key → buffer key). Unmapped keys pass through.
      2. If new key is in ``transpose_set``: (..., H, W, C) → (..., C, H, W).
      3. If new key is in ``resize_map``: bilinear resize to (target_h, target_w).
         Runs AFTER the transpose, so it expects CHW layout.
      4. If new key is in ``normalize_set``: cast to float32 and divide by 255.
         Matches VanillaObservationProcessorStep in the online path so offline
         and online image distributions agree.
    """
    out: dict[str, torch.Tensor] = {}
    resize_map = resize_map or {}
    normalize_set = normalize_set or set()
    for k, v in flat_obs.items():
        new_key = key_map.get(k, k)
        if new_key in transpose_set:
            # (B, H, W, C) → (B, C, H, W)
            if v.ndim == 4:
                v = v.permute(0, 3, 1, 2).contiguous()
            elif v.ndim == 3:
                v = v.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(
                    f"cannot HWC→CHW transpose tensor of ndim {v.ndim} (key={new_key})"
                )
        if new_key in resize_map:
            target_h, target_w = resize_map[new_key]
            # Assume CHW (post-transpose). Input v shape: (1, C, H, W).
            if v.ndim != 4:
                raise ValueError(
                    f"resize expects (B, C, H, W); got ndim={v.ndim} for key={new_key}"
                )
            if (v.shape[-2], v.shape[-1]) != (target_h, target_w):
                orig_dtype = v.dtype
                v_f = v.float()
                v_f = F.interpolate(v_f, size=(target_h, target_w), mode="bilinear", align_corners=False)
                v = v_f.to(orig_dtype)
        if new_key in normalize_set:
            # uint8 [0, 255] → float32 [0, 1]
            v = v.float() / 255.0
        out[new_key] = v
    return out


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        Replay buffer for storing transitions.
        It will allocate tensors on the specified device, when the first transition is added.
        NOTE: If you encounter memory issues, you can try to use the `optimize_memory` flag to save memory or
        and use the `storage_device` flag to store the buffer on a different device.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory

        # Track episode boundaries for memory optimization
        self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        # Determine shapes from the first transition
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # Pre-allocate tensors for storage
        self.states = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # Standard approach: store states and next_states separately
            self.next_states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # Memory-optimized approach: don't allocate next_states buffer
            # Just create a reference to states for consistent API
            self.next_states = self.states  # Just a reference for API consistency

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # Initialize storage for complementary_info
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # Pre-allocate tensors for each key in complementary_info
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), device=self.storage_device
                    )
                elif isinstance(value, (int | float)):
                    # Handle scalar values similar to reward
                    self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Saves a transition, ensuring tensors are stored on the designated storage device."""
        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

        # Store the transition in pre-allocated tensors
        for key in self.states:
            self.states[key][self.position].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][self.position].copy_(next_state[key].squeeze(dim=0))

        self.actions[self.position].copy_(action.squeeze(dim=0))
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated

        # Handle complementary_info if provided and storage is initialized
        if complementary_info is not None and self.has_complementary_info:
            # Store the complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][self.position].copy_(value.squeeze(dim=0))
                    elif isinstance(value, (int | float)):
                        self.complementary_info[key][self.position] = value

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        # When optimize_memory is enabled and buffer is not full, exclude the last item
        # because its next_state (at position `self.position`) has not been written yet.
        if self.optimize_memory and self.size < self.capacity and self.size > 1:
            high = self.size - 1
        else:
            high = self.size

        # Random indices for sampling - create on the same device as storage
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)

        # Identify image keys that need augmentation
        image_keys = [k for k in self.states if k.startswith(OBS_IMAGE)] if self.use_drq else []

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}

        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # Standard approach - load next_states directly
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
            else:
                # Memory-optimized approach - get next_state from the next index
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # Optimization: Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 2*batch_size images (batch_size for states, batch_size for next_states)
                # States start at index i*2*batch_size and take up batch_size slots
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states start after the states at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # Sample other tensors
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransition: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.

        Yields:
            BatchTransition: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransition: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_pickle_transitions(
        cls,
        pickle_paths: Sequence[str | Path],
        capacity: int | None = None,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        key_map: dict[str, str] | None = None,
        transpose_hwc_to_chw: Sequence[str] | None = None,
        resize_images: dict[str, tuple[int, int]] | None = None,
        normalize_to_unit: Sequence[str] | None = None,
    ) -> "ReplayBuffer":
        """Load HIL-SERL-format demo pickles into a ReplayBuffer.

        Expected pickle layout (matches `rail-berkeley/hil-serl`
        `examples/record_demos.py` and our own `scripts/real/collect_demo_task_policy.py`):

            [
              {
                "observations":      dict | np.ndarray,
                "actions":           np.ndarray,
                "next_observations": dict | np.ndarray,
                "rewards":           float,
                "masks":             float,      # ignored; 1 - done
                "dones":             bool,
                "infos":             dict,       # ignored
              },
              ...
            ]

        Nested observation dicts are flattened with dot-separated keys, so
        ``obs["pixels"]["front"]`` becomes state key ``"pixels.front"``. NumPy
        arrays are converted to torch tensors with a leading batch dim of 1.

        Args:
            pickle_paths: one or more pickle files. Transitions are concatenated
                in the order given (no shuffling).
            capacity: buffer capacity. If None, auto-sized to the total number of
                loaded transitions.
            key_map: optional mapping from pickle-side state keys to buffer-side
                state keys. Example: ``{"pixels.front": "observation.images.front",
                "agent_pos": "observation.state"}``. Keys not in the map pass through
                unchanged. Keys in the map but not in the pickle are ignored.
            transpose_hwc_to_chw: optional list of buffer-side keys whose tensors
                should be permuted (..., H, W, C) → (..., C, H, W) at load time.
                Image encoders expect CHW; pickles from our collection script
                are HWC uint8.
            normalize_to_unit: optional list of buffer-side keys whose tensors
                should be converted from ``uint8 [0, 255]`` to ``float32 [0, 1]``
                at load time. Needed to match online-path processors which do
                this divide; offline/online distribution drift otherwise.

        Returns:
            ReplayBuffer containing the loaded transitions, ready to be sampled
            like an online buffer.

        Raises:
            ValueError: if no transitions are loaded or capacity is too small.
            KeyError: if any transition is missing a required key.
        """
        all_transitions: list[dict] = []
        for path in pickle_paths:
            with open(path, "rb") as f:
                batch = pkl.load(f)
            if not isinstance(batch, list):
                raise ValueError(f"{path} must unpickle to a list; got {type(batch)}")
            all_transitions.extend(batch)

        if not all_transitions:
            raise ValueError(f"no transitions loaded from {list(pickle_paths)}")

        if capacity is None:
            capacity = len(all_transitions)
        if capacity < len(all_transitions):
            raise ValueError(
                f"capacity ({capacity}) < loaded transitions ({len(all_transitions)})"
            )

        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        transpose_set = set(transpose_hwc_to_chw or [])
        key_map = key_map or {}
        resize_map = resize_images or {}
        normalize_set = set(normalize_to_unit or [])

        for t in all_transitions:
            _require_keys(t, ("observations", "actions", "next_observations", "rewards", "dones"))
            state = _remap_and_transform(
                _flatten_obs_to_tensor_dict(t["observations"]),
                key_map=key_map,
                transpose_set=transpose_set,
                resize_map=resize_map,
                normalize_set=normalize_set,
            )
            next_state = _remap_and_transform(
                _flatten_obs_to_tensor_dict(t["next_observations"]),
                key_map=key_map,
                transpose_set=transpose_set,
                resize_map=resize_map,
                normalize_set=normalize_set,
            )
            action = _to_batched_tensor(t["actions"])

            # Preserve complementary_info schema so subsequent online-phase
            # transitions (which carry `discrete_penalty` + `is_intervention`
            # via frrl.rl.actor.TeleopEvents routing) can share the same buffer
            # storage. Default: all demo steps are intervention (human teleop).
            info = t.get("infos", {}) or {}
            is_intervention = bool(info.get("is_intervention", True))
            complementary_info = {
                "discrete_penalty": torch.tensor([info.get("discrete_penalty", 0.0)], dtype=torch.float32),
                "is_intervention": is_intervention,
            }

            replay_buffer.add(
                state={k: v.to(storage_device) for k, v in state.items()},
                action=action.to(storage_device),
                reward=float(t["rewards"]),
                next_state={k: v.to(storage_device) for k, v in next_state.items()},
                done=bool(t["dones"]),
                truncated=False,  # hil-serl schema has no truncated flag
                complementary_info=complementary_info,
            )

        return replay_buffer

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # Convert dataset to transitions
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

        # Initialize the buffer with the first transition to set up storage tensors
        if list_transition:
            first_transition = list_transition[0]
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition[ACTION].to(device)

            # Get complementary info if available
            first_complementary_info = None
            if (
                "complementary_info" in first_transition
                and first_transition["complementary_info"] is not None
            ):
                first_complementary_info = {
                    k: v.to(device) for k, v in first_transition["complementary_info"].items()
                }

            replay_buffer._initialize_storage(
                state=first_state, action=first_action, complementary_info=first_complementary_info
            )

        # Fill the buffer with all transitions
        for data in list_transition:
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(storage_device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(storage_device)

            action = data[ACTION]

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
                complementary_info=data.get("complementary_info", None),
            )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object.
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name=ACTION)
        features[ACTION] = act_info

        # Add "reward" and "done"
        features[REWARD] = {"dtype": "float32", "shape": (1,)}
        features[DONE] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Add complementary_info keys if available
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0]
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = guess_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames

        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict[ACTION] = self.actions[actual_idx].cpu()
            frame_dict[REWARD] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict[DONE] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()
            frame_dict["task"] = task_name

            # Add complementary_info if available
            if self.has_complementary_info:
                for key in self.complementary_info_keys:
                    val = self.complementary_info[key][actual_idx]
                    # Convert tensors to CPU
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        frame_dict[f"complementary_info.{key}"] = val.cpu()
                    # Non-tensor values can be used directly
                    else:
                        frame_dict[f"complementary_info.{key}"] = val

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode()

        lerobot_dataset.stop_image_writer()
        lerobot_dataset.finalize()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> list[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Sequence[str] | None):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = DONE in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample[ACTION].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample[REWARD].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample[DONE].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            transitions.append(transition)

        return transitions


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """

    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION], right_batch_transition[ACTION]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
