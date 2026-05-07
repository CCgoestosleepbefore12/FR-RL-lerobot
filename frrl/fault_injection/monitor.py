#!/usr/bin/env python3
"""BiasMonitor — 部署期间 q_true vs q_biased 实时双线图 + npz/png 存盘.

低开销诊断 overlay：每步追加样本（cheap），按 update_period 节流重绘。自动选取
非零 bias 的关节 only-plot；NaN 样本会被 matplotlib 自动断线（用于 episode
边界 / recovery 期间避免曲线伪交叉）。
"""
import time
from collections import deque
from typing import List, Optional

import numpy as np


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
