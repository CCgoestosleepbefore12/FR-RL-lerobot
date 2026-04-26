#!/usr/bin/env python3
"""
编码器校准偏差注入模块

模拟工业机器人编码器校准误差（encoder calibration bias）：
- 一个故障源（编码器bias）→ 同时影响控制和观测
- 控制器基于错误的关节角度计算Jacobian → 力矩不准 → 执行偏差
- 观测基于错误的FK计算 → EE位姿报告不准 → 感知偏差
- 物理仿真本身不受影响（关节真实位置正确）

这精确模拟了真实机器人编码器校准偏差的因果链：
    编码器偏差 → q_measured = q_true + bias
        ├→ 控制器用 q_measured 计算（控制受影响）
        └→ FK(q_measured) 得到错误的 EE 位姿（观测受影响）
"""

import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Tuple

import numpy as np
import yaml


@dataclass
class EncoderBiasConfig:
    """编码器偏差配置"""

    # 基础设置
    enable: bool = True
    error_probability: float = 1.0  # Episode级别故障概率

    # 目标关节（None=所有关节）
    target_joints: Optional[List[int]] = None
    per_joint_probability: float = 0.7  # 每关节被影响概率（仅target_joints=None时）

    # Bias模式
    bias_mode: str = 'random_uniform'  # 'fixed' 或 'random_uniform'
    fixed_bias_value: float = 0.1  # fixed模式下的bias值（弧度）
    bias_range: Tuple[float, float] = (0.0, 1.0)  # random_uniform模式下的范围（弧度）

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> 'EncoderBiasConfig':
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # 处理target_joints字段
        target_joints = data.get('target_joints', None)
        if isinstance(target_joints, list) and len(target_joints) == 0:
            target_joints = None

        return cls(
            enable=data.get('enable', True),
            error_probability=data.get('error_probability', 1.0),
            target_joints=target_joints,
            per_joint_probability=data.get('per_joint_probability', 0.7),
            bias_mode=data.get('bias_mode', 'random_uniform'),
            fixed_bias_value=data.get('fixed_bias_value', 0.1),
            bias_range=tuple(data.get('bias_range', [0.0, 1.0])),
        )


class EncoderBiasInjector:
    """
    编码器校准偏差注入器

    模拟编码器校准误差：每个episode开始时采样一个bias向量，
    该bias同时影响控制器（通过biased Jacobian）和观测（通过biased FK）。
    """

    def __init__(self, config: EncoderBiasConfig):
        self.config = config
        self._current_bias: Optional[np.ndarray] = None
        self._active = False
        self.episode_count = 0

        # 统计
        self.stats = {
            'total_episodes': 0,
            'fault_episodes': 0,
            'bias_magnitudes': [],
        }

        print("=" * 60)
        print("EncoderBiasInjector 初始化")
        print("=" * 60)
        print(f"  启用: {config.enable}")
        print(f"  故障概率: {config.error_probability * 100:.0f}%")

        if config.target_joints is not None:
            joint_names = [f"Joint {j+1}" for j in config.target_joints]
            print(f"  目标关节: {', '.join(joint_names)}")
        else:
            print(f"  目标关节: 全部 (每关节概率 {config.per_joint_probability*100:.0f}%)")

        print(f"  Bias模式: {config.bias_mode}")
        if config.bias_mode == 'fixed':
            print(f"  固定值: {config.fixed_bias_value:.3f} rad ({np.rad2deg(config.fixed_bias_value):.1f} deg)")
        else:
            print(f"  范围: [{config.bias_range[0]:.3f}, {config.bias_range[1]:.3f}] rad")
            print(f"         [{np.rad2deg(config.bias_range[0]):.1f}, {np.rad2deg(config.bias_range[1]):.1f}] deg")
        print("=" * 60)

    @property
    def current_bias(self) -> Optional[np.ndarray]:
        """当前episode的bias向量，None表示本episode无故障"""
        if self._active:
            return self._current_bias
        return None

    @property
    def is_active(self) -> bool:
        """当前episode是否有故障"""
        return self._active

    def on_episode_start(self, num_joints: int = 7) -> bool:
        """
        Episode开始时决定是否注入故障，并生成bias向量

        Returns:
            是否注入了故障
        """
        self.episode_count += 1
        self.stats['total_episodes'] += 1

        # 按概率决定是否注入
        self._active = (
            self.config.enable
            and np.random.random() < self.config.error_probability
        )

        if self._active:
            self._current_bias = self._sample_bias(num_joints)
            self.stats['fault_episodes'] += 1
            self.stats['bias_magnitudes'].append(
                np.linalg.norm(self._current_bias)
            )
            return True
        else:
            self._current_bias = None
            return False

    def get_biased_qpos(self, true_qpos: np.ndarray) -> np.ndarray:
        """
        返回编码器报告的关节角度（带bias）

        真实机器人上：编码器读数 = 真实角度 + 校准偏差
        用于控制器计算和观测FK。

        Args:
            true_qpos: 真实关节角度（前7个为手臂关节）

        Returns:
            带bias的关节角度（只修改手臂关节，不影响gripper和物体）
        """
        if not self._active or self._current_bias is None:
            return true_qpos

        biased = true_qpos.copy()
        n = min(7, len(self._current_bias))
        biased[:n] += self._current_bias[:n]
        return biased

    def _sample_bias(self, num_joints: int) -> np.ndarray:
        """采样bias向量"""
        bias = np.zeros(num_joints)

        if self.config.target_joints is not None:
            # 指定关节
            for j in self.config.target_joints:
                if j < num_joints:
                    bias[j] = self._sample_single_bias()
        else:
            # 所有关节，按概率独立采样
            for j in range(num_joints):
                if np.random.random() < self.config.per_joint_probability:
                    bias[j] = self._sample_single_bias()

        return bias

    def _sample_single_bias(self) -> float:
        """采样单个关节的bias值"""
        if self.config.bias_mode == 'fixed':
            return self.config.fixed_bias_value
        else:  # random_uniform
            return np.random.uniform(
                self.config.bias_range[0],
                self.config.bias_range[1],
            )

    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats['total_episodes'] > 0:
            stats['fault_rate'] = stats['fault_episodes'] / stats['total_episodes']
        if stats['bias_magnitudes']:
            mags = stats['bias_magnitudes']
            stats['avg_magnitude'] = np.mean(mags)
            stats['max_magnitude'] = np.max(mags)
            stats['min_magnitude'] = np.min(mags)
        return stats

    def print_stats(self):
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("编码器偏差注入统计")
        print("=" * 60)
        print(f"  总Episode数: {stats['total_episodes']}")
        print(f"  故障Episode数: {stats['fault_episodes']}")
        print(f"  故障率: {stats.get('fault_rate', 0) * 100:.1f}%")
        if 'avg_magnitude' in stats:
            print(f"  平均偏差幅度: {stats['avg_magnitude']:.4f} rad ({np.rad2deg(stats['avg_magnitude']):.2f} deg)")
            print(f"  最大偏差幅度: {stats['max_magnitude']:.4f} rad ({np.rad2deg(stats['max_magnitude']):.2f} deg)")
        print("=" * 60)


class BiasMonitor:
    """Real-time plot of true vs biased joint values during deployment.

    Designed as a **low-impact diagnostic overlay** for the main control loop:
      - Appends (q_true, q_biased, bias) each step (cheap)
      - Redraws only every `update_period` seconds (default 0.5s = 2 Hz)
      - Auto-detects which joints have nonzero bias; only plots those
      - Optional npz save on close for offline analysis / paper figures

    Usage:
        monitor = BiasMonitor(update_hz=2.0, save_path="bias_log.npz")
        # per main-loop step:
        monitor.update(q_true, q_biased, bias)
        # on shutdown:
        monitor.close()

    Args:
        history_seconds: Sliding window of samples to keep (default 30s).
        sample_hz:       Expected rate of update() calls (default 10Hz, for
                         buffer sizing only).
        update_hz:       Redraw rate of the matplotlib window (default 2Hz).
        bias_eps:        Threshold for "this joint has bias" (default 1e-4 rad).
        save_path:       If set, save buffered data as npz on close().
    """

    def __init__(
        self,
        history_seconds: float = 30.0,
        sample_hz: float = 10.0,
        update_hz: float = 2.0,
        bias_eps: float = 1e-4,
        save_path: Optional[str] = None,
        render: bool = True,
    ):
        self._update_period = 1.0 / float(update_hz) if update_hz > 0 else float("inf")
        self._bias_eps = float(bias_eps)
        self._save_path = save_path
        self._render = bool(render)

        # Ring buffers — sized for worst case at `sample_hz`
        max_samples = max(16, int(history_seconds * sample_hz))
        self._times = deque(maxlen=max_samples)
        self._q_true = deque(maxlen=max_samples)
        self._q_biased = deque(maxlen=max_samples)
        self._biases = deque(maxlen=max_samples)

        self._t0 = time.time()
        self._last_draw_time = 0.0

        # Matplotlib figure state — populated lazily on first nonzero bias
        self._fig = None
        self._axes = None
        self._lines_true = []
        self._lines_biased = []
        self._active_joints: Optional[List[int]] = None
        self._initialized = False
        self._disabled = False  # set True if plot init fails (e.g., no display)

        # Episode boundary markers — each entry is a dict:
        #   {"t": float, "ep": int, "label": str,
        #    "vlines": [Line2D...], "texts": [Text...]}
        # Populated by mark_episode_boundary(); artists created lazily (after
        # plot init) and pruned when the marker's t falls off the sliding
        # window. Also saved to npz on close() for offline segmentation.
        self._ep_markers: List[dict] = []

    def _try_init_plot(self, bias: np.ndarray) -> None:
        """Lazy plot setup on first nonzero bias. Idempotent."""
        if self._initialized or self._disabled or not self._render:
            return
        active = [i for i, b in enumerate(bias) if abs(float(b)) > self._bias_eps]
        if not active:
            return  # wait for nonzero bias

        try:
            import matplotlib.pyplot as plt
            # 禁掉 matplotlib 默认按键，否则与 KeyboardRewardListener 冲突：
            #   's' (save figure dialog) ↔ S (start episode)  ← 用户报告的弹窗根因
            #   'q' (close figure)        ↔ 一般退出快捷键
            #   'f'/'F' (fullscreen)      ↔ 用户可能误触
            #   'r'    (reset zoom)
            for _km in ("keymap.save", "keymap.quit", "keymap.quit_all",
                        "keymap.fullscreen", "keymap.home"):
                if _km in plt.rcParams:
                    plt.rcParams[_km] = []
            plt.ion()
            n = len(active)
            fig, axes = plt.subplots(n, 1, figsize=(9, 2.3 * n), sharex=True)
            if n == 1:
                axes = [axes]
            self._fig = fig
            self._axes = axes
            for i, j in enumerate(active):
                ax = axes[i]
                (lt,) = ax.plot([], [], color="tab:blue", lw=1.5,
                                label=f"q_true[J{j+1}]")
                (lb,) = ax.plot([], [], color="tab:red", lw=1.5,
                                label=f"q_biased[J{j+1}]")
                self._lines_true.append(lt)
                self._lines_biased.append(lb)
                ax.set_title(f"Joint {j+1}   bias = {float(bias[j]):+.4f} rad "
                             f"({np.rad2deg(float(bias[j])):+.2f}°)",
                             fontsize=10)
                ax.set_ylabel("rad")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("time (s)")
            fig.suptitle("BiasMonitor — real-time joint bias (q_true vs q_biased)",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.canvas.draw()
            fig.canvas.flush_events()
            self._active_joints = active
            self._initialized = True
            # Retroactively draw any episode boundaries that were marked before
            # the plot was initialized (typical: reset() marks boundary → first
            # step() update triggers init).
            for m in self._ep_markers:
                if not m["vlines"]:
                    self._draw_boundary_artists(m)
        except Exception as e:
            print(f"[BiasMonitor] plot init failed (will only save data): {e}")
            self._disabled = True

    def _draw_boundary_artists(self, marker: dict) -> None:
        """Draw the vline + top-anchored label on every subplot for one marker."""
        from matplotlib.transforms import blended_transform_factory
        t = marker["t"]
        for ax in self._axes:
            line = ax.axvline(t, color="gray", linestyle="--",
                              linewidth=0.8, alpha=0.6)
            tx = ax.text(
                t, 0.98, " " + marker["label"],
                transform=blended_transform_factory(ax.transData, ax.transAxes),
                va="top", ha="left", fontsize=8, color="dimgray",
            )
            marker["vlines"].append(line)
            marker["texts"].append(tx)

    def mark_episode_boundary(
        self,
        ep_num: int,
        bias: Optional[np.ndarray] = None,
    ) -> None:
        """Record a reset event so the plot shows a dashed vline + label.

        Call from env.reset() *after* the new bias has been set. `bias` (if
        given) is embedded in the label so each episode's injected value is
        visible on the plot without needing to read subplot titles.

        Safe to call before the figure is initialized — artists are buffered
        and drawn when the first nonzero-bias `update()` triggers init.
        """
        t = time.time() - self._t0
        if bias is not None:
            active = [
                f"J{j+1} {float(bias[j]):+.3f}"
                for j in range(len(bias))
                if abs(float(bias[j])) > self._bias_eps
            ]
            label = f"ep{ep_num}  " + " ".join(active) if active else f"ep{ep_num}"
        else:
            label = f"ep{ep_num}"
        marker = {"t": t, "ep": ep_num, "label": label,
                  "vlines": [], "texts": []}
        self._ep_markers.append(marker)
        if self._initialized and not self._disabled:
            try:
                self._draw_boundary_artists(marker)
                # 同步刷新各 active subplot 标题，显示当前 episode 的 bias 值。
                # 否则标题永远停在 _try_init_plot 时第一个 episode 的 bias 上。
                if bias is not None and self._active_joints is not None:
                    for i, j in enumerate(self._active_joints):
                        b = float(bias[j])
                        self._axes[i].set_title(
                            f"Joint {j+1}   ep{ep_num} bias = {b:+.4f} rad "
                            f"({np.rad2deg(b):+.2f}°)",
                            fontsize=10,
                        )
            except Exception as e:
                print(f"[BiasMonitor] boundary draw failed: {e}")

    def update(
        self,
        q_true: np.ndarray,
        q_biased: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        """Buffer one sample; redraw plot at the throttled rate."""
        t = time.time() - self._t0
        self._times.append(t)
        self._q_true.append(np.asarray(q_true, dtype=np.float32).copy())
        self._q_biased.append(np.asarray(q_biased, dtype=np.float32).copy())
        self._biases.append(np.asarray(bias, dtype=np.float32).copy())

        # Skip drawing if disabled (either by user request or after a prior
        # draw failure) — data still accumulates for save-on-close.
        if self._disabled:
            return

        if not self._initialized:
            self._try_init_plot(np.asarray(bias, dtype=np.float32))
            if not self._initialized:
                return  # still waiting for bias to be active

        now = time.time()
        if now - self._last_draw_time < self._update_period:
            return
        self._last_draw_time = now

        t_arr = np.fromiter(self._times, dtype=np.float32)
        qt_arr = np.stack(list(self._q_true))
        qb_arr = np.stack(list(self._q_biased))

        for i, j in enumerate(self._active_joints):
            self._lines_true[i].set_data(t_arr, qt_arr[:, j])
            self._lines_biased[i].set_data(t_arr, qb_arr[:, j])
            ax = self._axes[i]
            ax.relim()
            ax.autoscale_view()

        # Prune episode boundary artists that have scrolled off the ring buffer
        # (t < oldest retained sample). Matplotlib keeps them in the axes list
        # forever unless explicitly removed.
        if self._ep_markers and self._times:
            t_min = float(self._times[0])
            surviving = []
            for m in self._ep_markers:
                if m["t"] >= t_min:
                    surviving.append(m)
                    continue
                for artist in m["vlines"] + m["texts"]:
                    try:
                        artist.remove()
                    except Exception:
                        pass
            self._ep_markers = surviving

        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception as e:
            print(f"[BiasMonitor] draw failed (disabling): {e}")
            self._disabled = True

    def close(self) -> None:
        """Save buffered data (npz) + figure snapshot (png) and close.

        Both files share the prefix ``self._save_path`` (.npz removed if
        present): e.g. ``charts/bias_2026-04-26_18-05-00`` →
        ``...npz`` + ``...png``. ``_save_path=None`` 时只关图不存。
        """
        if self._save_path is not None and len(self._times) > 0:
            from pathlib import Path
            base = Path(str(self._save_path))
            if base.suffix == ".npz":
                base = base.with_suffix("")
            try:
                ep_t = np.array([m["t"] for m in self._ep_markers],
                                dtype=np.float32)
                ep_num = np.array([m["ep"] for m in self._ep_markers],
                                  dtype=np.int32)
                npz_path = base.with_suffix(".npz")
                np.savez(
                    str(npz_path),
                    t=np.array(self._times, dtype=np.float32),
                    q_true=np.stack(list(self._q_true)),
                    q_biased=np.stack(list(self._q_biased)),
                    bias=np.stack(list(self._biases)),
                    ep_boundary_t=ep_t,
                    ep_boundary_num=ep_num,
                )
                print(f"[BiasMonitor] saved {len(self._times)} samples to {npz_path}")
            except Exception as e:
                print(f"[BiasMonitor] npz save failed: {e}")
            if self._fig is not None:
                try:
                    png_path = base.with_suffix(".png")
                    self._fig.savefig(str(png_path), dpi=120, bbox_inches="tight")
                    print(f"[BiasMonitor] saved figure to {png_path}")
                except Exception as e:
                    print(f"[BiasMonitor] png save failed: {e}")
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass


if __name__ == '__main__':
    print("测试 EncoderBiasInjector...\n")

    config = EncoderBiasConfig(
        enable=True,
        error_probability=0.5,
        target_joints=[3],  # Joint 4
        bias_mode='random_uniform',
        bias_range=(0.0, 0.5),
    )

    injector = EncoderBiasInjector(config)

    for ep in range(10):
        active = injector.on_episode_start(num_joints=7)
        if active:
            true_q = np.zeros(7)
            biased_q = injector.get_biased_qpos(true_q)
            print(f"Episode {ep+1}: 故障 | bias = {injector.current_bias}")
        else:
            print(f"Episode {ep+1}: 正常")

    injector.print_stats()
    print("\n测试完成")
