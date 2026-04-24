# External-repo patches

FR-RL-lerobot's real-hardware stack depends on one upstream repo whose changes
cannot be vendored into this Python package because it is a native catkin ROS
package. The changes live here as patch files so they remain under version
control and can be reapplied if the upstream checkout is ever wiped or refreshed.

**Server-side Python (`franka_server.py` and the gripper servers) was vendored
into [`frrl/robots/franka_real/servers/`](../frrl/robots/franka_real/servers/) — no patch needed for that.**
Only the C++ impedance controller still requires patching.

---

## `serl_franka_controllers_bias_injection.patch`

**Upstream**: `https://github.com/rail-berkeley/serl_franka_controllers.git`
**Base commit**: `1f140ef0d8e3fc443569c193d3ede1856e50d521`
(`add instruction to ignore realtime constraint` on `main`)

**What it adds**:
- `msg/BiasedState.msg` — new custom message carrying `q_biased[7]`,
  `O_T_EE_biased[16]`, `bias[7]`, published at ~1 kHz by the impedance
  controller.
- `CMakeLists.txt` / `package.xml` — register the new message and depend on
  `std_msgs` (for the `Float64MultiArray` subscription).
- `include/serl_franka_controllers/cartesian_impedance_controller.h` — add
  `RealtimeBuffer<std::array<double, 7>>` for the encoder bias, a subscriber
  for `/encoder_bias`, and a `RealtimePublisher` for `BiasedState`.
- `src/cartesian_impedance_controller.cpp` — fault injection point B. In
  `update()`, read the bias from the RT buffer, add it to `robot_state.q`,
  and feed the biased joints into `model_handle_->getPose(...)` /
  `getZeroJacobian(...)` overloads so FK, Jacobian, nullspace term, and the
  final torque are all computed under the biased state. Publishes biased q
  and O_T_EE on the `biased_state` topic for the server side to forward.

See [`docs/fault_injection_realhw.md`](../docs/fault_injection_realhw.md)
for the full architecture, verification procedure, and physical-behavior
analysis.

### Apply

From a fresh clone of the upstream repo:

```bash
cd ~/serl_ws/src
git clone https://github.com/rail-berkeley/serl_franka_controllers.git
cd serl_franka_controllers
git checkout 1f140ef0d8e3fc443569c193d3ede1856e50d521
git apply ~/FR-RL-lerobot/patches/serl_franka_controllers_bias_injection.patch
cd ~/serl_ws
source /opt/ros/noetic/setup.bash
catkin_make --only-pkg-with-deps serl_franka_controllers
```

Verify it loaded correctly after a fresh `start_franka_server.sh`:

```bash
# in another terminal
source ~/serl_ws/devel/setup.bash
rostopic hz /cartesian_impedance_controller/biased_state   # expect ~1000 Hz
curl -s -X POST http://127.0.0.1:5000/getstate | python3 -c "
import sys, json; s = json.load(sys.stdin)
print('bias:', s['bias'])"
```

### Regenerate after further edits

If you touch the controller source again, refresh this patch rather than
committing a second one:

```bash
cd ~/serl_ws/src/serl_franka_controllers
git add -N msg/BiasedState.msg   # so git diff picks up the untracked file
git diff > ~/FR-RL-lerobot/patches/serl_franka_controllers_bias_injection.patch
git restore --staged msg/BiasedState.msg
```

Then commit the updated `.patch` file to FR-RL-lerobot.

### Why not a fork

A proper fork of `serl_franka_controllers` under the user's GitHub would be
cleaner long-term, but would add a second repo to maintain and one more
clone step to every deployment. Until the C++ changes stabilize or require
upstream-style collaboration, the patch-file approach is lighter and keeps
every FR-RL-lerobot checkout self-describing.
