#!/bin/bash
# ============================================================
# 从 pretrain ckpt 启动 online HIL 训练 helper
#
# 解决 lerobot resume 设计的两个限制：
#   (a) --config_path 必须指 ckpt 内的 train_config.json（不是 online config）
#   (b) RESUMABLE_POLICY_OVERRIDES 白名单只允许覆盖 2 个字段，不够把 pretrain
#       的 offline_only_mode=true 切到 online 模式
#
# 流程：
#   1. cp pretrain ckpt 到独立 online 目录（避免污染 pretrain）
#   2. 用 task 对应的 train_hil_sac_<task>_real.json 覆盖 ckpt 内的 train_config.json
#      （保留 model.safetensors + training_state，仅切训练 meta）
#   3. resume 这个 online 目录
#
# 用法：
#   bash scripts/real/resume_online_from_pretrain.sh PRETRAIN_DIR ROLE [TASK]
#
#   PRETRAIN_DIR: pretrain output dir，例如 checkpoints/wipe_pretrain_20260427_144114
#   ROLE:         learner / actor
#   TASK:         可选，wipe / pickup / ...，决定 ONLINE_CONFIG。默认 wipe
#                 兼容历史调用。配 ONLINE_CONFIG 环境变量也可手动指定 path。
#
# 第一次跑 ROLE=learner 会建 online dir（保存到 .online_dir 里给 actor 读）；
# 第二次跑 ROLE=actor 复用同一个 online dir。
# ============================================================
set -e

PRETRAIN_DIR="${1:?需要 pretrain dir 作第 1 个参数}"
ROLE="${2:?需要 learner 或 actor 作第 2 个参数}"
TASK="${3:-wipe}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_FILE="$PRETRAIN_DIR/.online_dir"

# ONLINE_CONFIG 仅 learner 第一次建 online_dir 时需要（用来覆盖 ckpt 内 train_config.json）。
# actor 始终从 ckpt 内 patched train_config.json 启动，跟 ONLINE_CONFIG 无关。
# learner 复用 STATE_FILE 时也跳过校验：config 已经 paste 过。
case "$ROLE" in
    learner)
        if [ -f "$STATE_FILE" ]; then
            ONLINE_DIR=$(cat "$STATE_FILE")
            echo "[INFO] 复用已存在的 online dir: $ONLINE_DIR"
            if [ ! -d "$ONLINE_DIR" ]; then
                echo "[ERR] $STATE_FILE 指向 $ONLINE_DIR，但目录不存在；删 .online_dir 重跑"
                exit 1
            fi
            # 用户复用现有 online_dir 时再传 $3 TASK 容易误以为能切任务，但 paste
            # 已发生且 ckpt train_config.json 已固化 — 这里显式 warn。
            if [ "$#" -ge 3 ]; then
                echo "[WARN] 第 3 个参数 TASK='$TASK' 被忽略 — online_dir 已存在，"
                echo "       ckpt 内 train_config.json 已 paste 完毕。要切 task 必须删除 $STATE_FILE 重起 learner。"
            fi
        else
            ONLINE_CONFIG="${ONLINE_CONFIG:-$PROJECT_DIR/scripts/configs/train_hil_sac_${TASK}_real.json}"
            if [ ! -f "$ONLINE_CONFIG" ]; then
                echo "[ERR] online config 不存在: $ONLINE_CONFIG"
                echo "      可用 task: $(cd "$PROJECT_DIR/scripts/configs" && ls train_hil_sac_*_real.json | sed 's/train_hil_sac_//;s/_real.json//' | xargs)"
                exit 1
            fi
            echo "[INFO] task=$TASK, online_config=$ONLINE_CONFIG"
            ONLINE_DIR="checkpoints/${TASK}_online_$(date +%Y%m%d_%H%M%S)"
            echo "[INFO] 建新 online dir: $ONLINE_DIR"
            # 跳过 wandb/ 和 logs/ 子目录：pretrain 的 wandb 历史 / log 文件
            # 不该带过来，否则 online learner 启动时 wandb 会试图 resume pretrain
            # 的旧 run（partial init 失败 → wandb 报 resume='must' invalid）。
            rsync -a --exclude='wandb' --exclude='logs' "$PRETRAIN_DIR/" "$ONLINE_DIR/"
            # 用 online config 覆盖 ckpt 内的 train_config.json，但同时把 output_dir
            # 字段改成 ONLINE_DIR：lerobot resume 走 ckpt 内 train_config.json 作为
            # cfg，不在 RESUMABLE_POLICY_OVERRIDES 白名单的字段会被 ckpt 覆盖
            # （含 output_dir）。如果 paste 进去时 output_dir 是 task config 顶层
            # 默认值（如 "checkpoints/wipe_real" / "checkpoints/pickup_real"），
            # learner load training_state 时会去那个错路径找不到 → 必须把
            # output_dir patch 成 ONLINE_DIR 自身。
            python3 -c "
import json, sys
with open(sys.argv[1]) as f: cfg = json.load(f)
cfg['output_dir'] = sys.argv[2]
with open(sys.argv[3], 'w') as f: json.dump(cfg, f, indent=4)
" "$ONLINE_CONFIG" "$ONLINE_DIR" \
                "$ONLINE_DIR/checkpoints/last/pretrained_model/train_config.json"
            # 保存 ONLINE_DIR 给 actor 终端读
            echo "$ONLINE_DIR" > "$STATE_FILE"
            echo "[INFO] online config 已 paste 到 $ONLINE_DIR/checkpoints/last/pretrained_model/train_config.json"
            echo "[INFO] online dir 路径已写入 $STATE_FILE，actor 终端会自动读"
        fi

        echo "[INFO] 启动 Learner..."
        # lerobot parse_arg 只认 --key=value，不认 --key value（空格）
        python -m frrl.rl.core.learner \
            "--config_path=$ONLINE_DIR/checkpoints/last/pretrained_model/train_config.json" \
            "--output_dir=$ONLINE_DIR" \
            "--resume=true"
        ;;
    actor)
        if [ ! -f "$STATE_FILE" ]; then
            echo "[ERR] $STATE_FILE 不存在 —— 必须先跑 learner（会建 online dir 并写入此文件）"
            exit 1
        fi
        ONLINE_DIR=$(cat "$STATE_FILE")
        echo "[INFO] 复用 online dir: $ONLINE_DIR"
        echo "[INFO] 启动 Actor..."
        python -m frrl.rl.core.actor \
            "--config_path=$ONLINE_DIR/checkpoints/last/pretrained_model/train_config.json" \
            "--output_dir=$ONLINE_DIR" \
            "--resume=true"
        ;;
    *)
        echo "未知 ROLE: $ROLE （需要 learner 或 actor）"
        exit 1
        ;;
esac
