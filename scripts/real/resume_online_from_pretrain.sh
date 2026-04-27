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
#   2. 用 train_hil_sac_task_real.json 覆盖 ckpt 内的 train_config.json
#      （保留 model.safetensors + training_state，仅切训练 meta）
#   3. resume 这个 online 目录
#
# 用法：
#   bash scripts/real/resume_online_from_pretrain.sh PRETRAIN_DIR ROLE
#
#   PRETRAIN_DIR: pretrain output dir，例如 checkpoints/wipe_pretrain_20260427_144114
#   ROLE:         learner / actor
#
# 第一次跑 ROLE=learner 会建 online dir（保存到 .online_dir 里给 actor 读）；
# 第二次跑 ROLE=actor 复用同一个 online dir。
# ============================================================
set -e

PRETRAIN_DIR="${1:?需要 pretrain dir 作第 1 个参数}"
ROLE="${2:?需要 learner 或 actor 作第 2 个参数}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ONLINE_CONFIG="$PROJECT_DIR/scripts/configs/train_hil_sac_task_real.json"
STATE_FILE="$PRETRAIN_DIR/.online_dir"

case "$ROLE" in
    learner)
        if [ -f "$STATE_FILE" ]; then
            ONLINE_DIR=$(cat "$STATE_FILE")
            echo "[INFO] 复用已存在的 online dir: $ONLINE_DIR"
            if [ ! -d "$ONLINE_DIR" ]; then
                echo "[ERR] $STATE_FILE 指向 $ONLINE_DIR，但目录不存在；删 .online_dir 重跑"
                exit 1
            fi
        else
            ONLINE_DIR="checkpoints/wipe_online_$(date +%Y%m%d_%H%M%S)"
            echo "[INFO] 建新 online dir: $ONLINE_DIR"
            cp -r "$PRETRAIN_DIR" "$ONLINE_DIR"
            # 用 online config 覆盖 ckpt 内的 train_config.json
            cp "$ONLINE_CONFIG" \
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
