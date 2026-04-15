"""Standalone SpaceMouse smoke test — prints 6D state + buttons.

Run:
    python scripts/test_spacemouse.py

Move the SpaceMouse; values should update. Press buttons; flags should flip.
Ctrl-C to exit.
"""
import time
import pyspacemouse


def main():
    device = pyspacemouse.open()
    if device is None or device is False:
        print("[FAIL] could not open SpaceMouse — check udev rule and unplug/replug")
        return

    print("SpaceMouse opened. Move / press buttons. Ctrl-C to exit.\n")
    try:
        while True:
            s = device.read()
            line = (
                f"x={s.x:+.3f}  y={s.y:+.3f}  z={s.z:+.3f}   "
                f"roll={s.roll:+.3f}  pitch={s.pitch:+.3f}  yaw={s.yaw:+.3f}   "
                f"buttons={list(s.buttons)}"
            )
            print(line, end="\r", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n\nexit")
    finally:
        try:
            device.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
