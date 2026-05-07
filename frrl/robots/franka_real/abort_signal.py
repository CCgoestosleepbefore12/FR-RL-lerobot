"""Reliable SIGINT (Ctrl+C) handling for deploy scripts.

为什么要这个 module：matplotlib TkAgg backend 在 ``Tkinter.__call__`` 里有个
try/except 默默吞 KeyboardInterrupt，BiasMonitor 每 0.5s 一次 ``flush_events``
让主线程频繁进 Tk callback —— Ctrl+C 大概率降在 callback 里被吞，主循环 ``while
True`` 完全收不到。表象是按 Ctrl+C 没反应，要按好几次才能停。

解决：装 SIGINT 信号处理器，**只设 flag 不抛 KeyboardInterrupt**。Tk callback
继续正常完成，主循环下一次迭代检查 flag 干净退出。

用法：
    from frrl.robots.franka_real.abort_signal import install, aborted

    def main():
        install()                       # 入口立刻装
        ...
        while not aborted():            # 替代 `while True:`
            # 主循环体
            ...
        # finally 里 cleanup 自然继续，第 2 次 Ctrl+C 也只是再 set flag 不抛异常
"""
import os
import signal

_state = {"abort": False, "count": 0}


def _handler(sig, frame):
    _state["count"] += 1
    if _state["count"] >= 3:
        # 3rd Ctrl+C：紧急逃生，绕过任何 finally / 析构 / 资源清理
        print("\n[abort] Force quit (3rd Ctrl+C)", flush=True)
        os._exit(1)
    if not _state["abort"]:
        print(
            "\n[abort] Ctrl+C received — finishing current step + cleanup. "
            "Press Ctrl+C 3x total for force quit.",
            flush=True,
        )
        _state["abort"] = True
    else:
        print(
            f"[abort] Ctrl+C #{_state['count']} — cleanup in progress, "
            f"please wait ({3 - _state['count']} more for force quit)",
            flush=True,
        )


def install() -> None:
    """Install SIGINT handler. Idempotent; safe to call multiple times."""
    signal.signal(signal.SIGINT, _handler)


def aborted() -> bool:
    """Main loops poll this every iteration to decide whether to exit."""
    return _state["abort"]


def reset() -> None:
    """For tests / interactive sessions to clear state. Production deploy
    scripts don't need this."""
    _state["abort"] = False
    _state["count"] = 0
