"""Tests for the offline_only_mode pretraining path — config + contract.

We don't run a full learner training here (needs policy + optimizer +
multiprocessing) — that's validated by running scripts/tools/pretrain_task_policy.py
against real demos. These tests pin down the public contract:

  * SACConfig exposes the three new fields with sensible defaults
  * Each config instance gets its own demo_pickle_paths list (no shared mutable state)

The learner-side integration (initialize_offline_replay_buffer routing to
pickle loader) can't be unit-tested here because importing frrl.rl.learner
pulls in gRPC stubs which need a protobuf version not installed in this env.
That path is exercised via the end-to-end pretrain script.
"""
from frrl.policies.sac.configuration_sac import SACConfig


class TestSACConfigFields:
    def test_offline_only_mode_default_false(self):
        cfg = SACConfig()
        assert cfg.offline_only_mode is False

    def test_offline_pretrain_steps_default_5000(self):
        cfg = SACConfig()
        assert cfg.offline_pretrain_steps == 5000

    def test_demo_pickle_paths_default_empty_list(self):
        cfg = SACConfig()
        assert cfg.demo_pickle_paths == []

    def test_demo_pickle_paths_is_assignable(self):
        cfg = SACConfig()
        cfg.demo_pickle_paths = ["demos/a.pkl", "demos/b.pkl"]
        assert len(cfg.demo_pickle_paths) == 2

    def test_each_config_instance_gets_fresh_list(self):
        """Dataclass field(default_factory=list) must not share mutable state."""
        a = SACConfig()
        b = SACConfig()
        a.demo_pickle_paths.append("x.pkl")
        assert b.demo_pickle_paths == []
