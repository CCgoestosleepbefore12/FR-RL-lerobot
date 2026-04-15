#!/usr/bin/env python3
"""Interactive test: random-mode encoder bias injection on real Franka.

Samples biases from the same distribution that EncoderBiasInjector uses in
random_uniform mode, and pushes each one through franka_server's
/set_encoder_bias HTTP endpoint. Between samples the bias is cleared so the
robot returns to its nominal pose.

Safety:
- Script pauses before every injection so the user can abort if a sampled
  value is too aggressive for the current environment.
- Predicted end-effector sweep is printed so it can be compared to
  available clearance.
- Requires franka_server to already be running on 127.0.0.1:5000 with the
  biased_state pipeline active (kill + restart if you haven't loaded the
  new controller build yet).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import requests


SERVER = "http://127.0.0.1:5000"

# --- Sampling config (mirrors EncoderBiasConfig random_uniform mode) ---
N_EPISODES = 5
TARGET_JOINT = 0                 # Joint 1 (0-indexed)
BIAS_RANGE = (0.1, 0.25)         # rad, matches阶段二 target

# --- Timing ---
SETTLE_AFTER_INJECT = 2.5        # let the robot glide to the biased equilibrium
SETTLE_AFTER_CLEAR = 2.5         # let it glide back

# Rough Jacobian magnitude for J1 at typical Panda ready pose; used only for
# printing a predicted EE offset. Not used by the control loop.
J1_ROUGH_ARM_RADIUS_M = 0.7


def _post(path: str, **kwargs) -> dict | None:
    r = requests.post(f"{SERVER}{path}", **kwargs)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


def _read_state() -> dict:
    return _post("/getstate") or {}


def main() -> int:
    rng = np.random.default_rng()

    print("=" * 60)
    print("Random encoder-bias injection test")
    print("=" * 60)
    print(f"  Server:        {SERVER}")
    print(f"  Target joint:  J{TARGET_JOINT + 1}")
    print(f"  Bias range:    [{BIAS_RANGE[0]:.3f}, {BIAS_RANGE[1]:.3f}] rad")
    print(f"  Episodes:      {N_EPISODES}")
    print("=" * 60)

    # Baseline: confirm server is up and bias currently zero
    s = _read_state()
    if not s:
        print("ERROR: could not read /getstate. Is franka_server running?")
        return 1
    print("Baseline state:")
    print(f"  bias: {s['bias']}")
    print(f"  q[0]: {s['q'][0]:.4f}")
    print(f"  pose: {[round(x, 4) for x in s['pose'][:3]]}")
    if any(abs(b) > 1e-6 for b in s["bias"]):
        print("WARNING: baseline bias is non-zero, clearing before starting.")
        _post("/clear_encoder_bias")
        time.sleep(SETTLE_AFTER_CLEAR)

    sampled = []
    for ep in range(1, N_EPISODES + 1):
        bias = np.zeros(7)
        bias[TARGET_JOINT] = rng.uniform(*BIAS_RANGE)
        sampled.append(bias[TARGET_JOINT])

        predicted_offset_cm = abs(bias[TARGET_JOINT]) * J1_ROUGH_ARM_RADIUS_M * 100

        print("\n" + "-" * 60)
        print(f"Episode {ep}/{N_EPISODES}")
        print(f"  Sampled J{TARGET_JOINT + 1} bias: "
              f"{bias[TARGET_JOINT]:.4f} rad ({np.rad2deg(bias[TARGET_JOINT]):.2f} deg)")
        print(f"  Predicted real EE sweep: ~{predicted_offset_cm:.1f} cm (horizontal arc)")
        print("-" * 60)
        try:
            input("  [Enter] inject    [Ctrl+C] abort  > ")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.")
            break

        _post("/set_encoder_bias", json={"bias": bias.tolist()})
        print(f"  Injected. Waiting {SETTLE_AFTER_INJECT}s for settling...")
        time.sleep(SETTLE_AFTER_INJECT)

        s = _read_state()
        print(f"  Biased q:    {[round(x, 4) for x in s['q']]}")
        print(f"  Biased pose: {[round(x, 4) for x in s['pose'][:3]]}")
        print(f"  bias readback: {s['bias']}")

        try:
            input("  [Enter] clear bias > ")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user (clearing anyway).")
            _post("/clear_encoder_bias")
            break

        _post("/clear_encoder_bias")
        print(f"  Cleared. Waiting {SETTLE_AFTER_CLEAR}s for return...")
        time.sleep(SETTLE_AFTER_CLEAR)

    # Final safety: always end with bias cleared
    _post("/clear_encoder_bias")

    if sampled:
        arr = np.array(sampled)
        print("\n" + "=" * 60)
        print("Sampling stats")
        print("=" * 60)
        print(f"  n:      {len(arr)}")
        print(f"  min:    {arr.min():.4f} rad ({np.rad2deg(arr.min()):.2f} deg)")
        print(f"  max:    {arr.max():.4f} rad ({np.rad2deg(arr.max()):.2f} deg)")
        print(f"  mean:   {arr.mean():.4f} rad ({np.rad2deg(arr.mean()):.2f} deg)")
        print("=" * 60)
    print("Done. Bias is cleared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
