"""Pure helpers used by the RL actor loop.

Kept separate from `actor.py` so that lightweight tests (and downstream tools
that only need the contract) can import this module without pulling in gRPC,
torch.multiprocessing, or protobuf-generated services.
"""

from __future__ import annotations

from typing import Optional


# Episode discard key — set by FrankaRealEnv when the operator hits Backspace
# (HIL-SERL keyboard reward backend). When True, the entire episode's
# transitions are dropped instead of pushed to the learner. Keep in sync with
# frrl/envs/franka_real_env.py step() info dict.
DISCARD_INFO_KEY = "discard"


def should_discard_episode(last_info: Optional[dict]) -> bool:
    """Returns True if the operator marked the finished episode as 'discard'.

    Accepts any mapping; missing key / None input → False. Defined as a pure
    function so the discard hook in the actor loop can be unit-tested without
    mocking the whole distributed training stack.
    """
    if not last_info:
        return False
    return bool(last_info.get(DISCARD_INFO_KEY, False))
