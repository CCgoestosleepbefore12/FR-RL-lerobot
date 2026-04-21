#!/bin/bash
# ============================================================
# FR-RL HIL-SERL 训练启动脚本
#
# 完整流程:
#   1. 录制Demo:  bash scripts/train_hil_sac.sh baseline record
#   2. 启动训练:
#      终端1: bash scripts/train_hil_sac.sh baseline learner
#      终端2: bash scripts/train_hil_sac.sh baseline actor
#
# 键盘操作（录制/Actor模式）:
#   方向键↑↓    : delta_x (前/后)
#   方向键←→    : delta_y (左/右)
#   Shift/Shift_R: delta_z (下/上)
#   Ctrl_L/Ctrl_R: 夹爪关闭/打开
#   Space        : 激活/取消干预模式
#   Enter        : 标记成功（episode结束）
#   ESC          : 标记失败
#   R            : 重录当前episode
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/configs/train_hil_sac_base.json"

VARIANT="${1:-baseline}"
ROLE="${2:-both}"
EXTRA_ARGS=""
TASK_ID=""

# 选择任务变体
case "$VARIANT" in
    baseline)
        echo "=== FR-RL pick-and-place 无故障 baseline ==="
        TASK_ID="FRRLPandaPickPlaceKeyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_baseline"
        ;;
    bias_j4_random)
        echo "=== FR-RL pick-and-place Joint4 随机偏差 [0,1] rad ==="
        TASK_ID="FRRLPandaPickPlaceBiasJ4Random-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_bias_j4_random"
        ;;
    bias_j4_fixed)
        echo "=== FR-RL pick-and-place Joint4 固定偏差 0.3 rad ==="
        TASK_ID="FRRLPandaPickPlaceBiasJ4Fixed03-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_bias_j4_fixed"
        ;;
    bias_all)
        echo "=== FR-RL pick-and-place 全关节随机偏差 ==="
        TASK_ID="FRRLPandaPickPlaceBiasAllJoints-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_bias_all"
        ;;
    pick_cube)
        echo "=== PandaPickCube: 抓取方块并举高（无偏差）==="
        TASK_ID="PandaPickCubeKeyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_pick_cube"
        ;;
    pick_cube_bias)
        echo "=== PandaPickCube + Joint4固定偏差0.2rad（TCP偏移~13cm）==="
        TASK_ID="PandaPickCubeBiasJ4Fixed02Keyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_pick_cube_bias"
        ;;
    pick_cube_bias_random)
        echo "=== PandaPickCube + Joint4随机偏差[0,0.25]rad（每episode不同）==="
        TASK_ID="PandaPickCubeBiasJ4RandomKeyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_pick_cube_bias_random"
        ;;
    arrange_boxes)
        echo "=== PandaArrangeBoxes: 整理多个盒子 ==="
        TASK_ID="PandaArrangeBoxesKeyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_arrange_boxes"
        ;;
    safe)
        CONFIG="$PROJECT_DIR/configs/train_hil_sac_safe.json"
        echo "=== 安全场景: PickPlaceSafe 无偏差 ==="
        TASK_ID="PandaPickPlaceSafeKeyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_safe"
        ;;
    safe_bias)
        CONFIG="$PROJECT_DIR/configs/train_hil_sac_safe_bias.json"
        echo "=== 安全场景: PickPlaceSafe + Joint1随机偏差 ==="
        TASK_ID="PandaPickPlaceSafeBiasJ1RandomKeyboard-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_safe_bias"
        ;;
    backup)
        CONFIG="$PROJECT_DIR/configs/train_hil_sac_backup_s1.json"
        echo "=== Backup Policy S1: 单障碍物避障训练 ==="
        TASK_ID="PandaBackupPolicyS1-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_backup_policy_s1"
        ;;
    backup_s2)
        CONFIG="$PROJECT_DIR/configs/train_hil_sac_backup_s2.json"
        echo "=== Backup Policy S2: 移动+静止障碍物避障训练 ==="
        TASK_ID="PandaBackupPolicyS2-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_backup_policy_s2"
        ;;
    backup_tracking)
        CONFIG="$PROJECT_DIR/configs/train_hil_sac_backup_s1_tracking.json"
        echo "=== Backup Policy S1-TRACKING: 手追 TCP 避障训练（6D, 300k, 20step, disp=0.15）==="
        TASK_ID="PandaBackupPolicyS1-v0"
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_backup_policy_s1_tracking"
        ;;
    custom)
        TASK_ID="${3:?请指定环境ID，例如: FRRLPandaPickPlaceKeyboard-v0}"
        echo "=== 自定义任务: $TASK_ID ==="
        EXTRA_ARGS="--env.task $TASK_ID --job_name frrl_hil_sac_custom"
        ;;
    *)
        echo "未知任务: $VARIANT"
        echo "可选: baseline, bias_j4_random, bias_j4_fixed, bias_all, pick_cube, pick_cube_bias, pick_cube_bias_random, arrange_boxes, safe, safe_bias, backup, backup_s2, backup_tracking, custom"
        exit 1
        ;;
esac

# 选择角色
case "$ROLE" in
    learner)
        echo "启动 Learner (GPU训练进程)..."
        python -m frrl.rl.learner \
            --config_path "$CONFIG" $EXTRA_ARGS
        ;;
    actor)
        echo "启动 Actor (环境交互 + 键盘遥操)..."
        echo "请确保 Learner 已在另一个终端启动"
        python -m frrl.rl.actor \
            --config_path "$CONFIG" $EXTRA_ARGS
        ;;
    record)
        # safe变体用专用的录制config和cache路径
        case "$VARIANT" in
            safe)
                RECORD_CONFIG="$PROJECT_DIR/configs/record_demo_safe.json"
                DEMO_CACHE="$HOME/.cache/huggingface/lerobot/frrl/pick_place_safe_demo"
                ;;
            safe_bias)
                RECORD_CONFIG="$PROJECT_DIR/configs/record_demo_safe_bias.json"
                DEMO_CACHE="$HOME/.cache/huggingface/lerobot/frrl/pick_place_safe_bias_demo"
                ;;
            backup|backup_s2)
                RECORD_CONFIG="$PROJECT_DIR/configs/record_demo_backup.json"
                DEMO_CACHE="$HOME/.cache/huggingface/lerobot/frrl/backup_policy_demo"
                ;;
            *)
                RECORD_CONFIG="$PROJECT_DIR/configs/record_demo.json"
                DEMO_CACHE="$HOME/.cache/huggingface/lerobot/frrl/pick_place_demo"
                ;;
        esac
        if [ -d "$DEMO_CACHE" ]; then
            echo "清理旧录制数据: $DEMO_CACHE"
            rm -rf "$DEMO_CACHE"
        fi
        echo "录制Demo（键盘遥操）..."
        echo "任务: $TASK_ID"
        echo "操作: Space激活干预 → 方向键移动 → Shift上下 → Ctrl夹爪 → Enter标记成功 → ESC退出"
        python -m frrl.rl.env_factory \
            --config_path "$RECORD_CONFIG" \
            --env.task "$TASK_ID"
        ;;
    both)
        echo ""
        echo "完整训练流程:"
        echo ""
        echo "  步骤1 — 录制Demo（30个episode）:"
        echo "    bash scripts/train_hil_sac.sh $VARIANT record"
        echo ""
        echo "  步骤2 — 启动训练:"
        echo "    终端1: bash scripts/train_hil_sac.sh $VARIANT learner"
        echo "    终端2: bash scripts/train_hil_sac.sh $VARIANT actor"
        echo ""
        ;;
esac
