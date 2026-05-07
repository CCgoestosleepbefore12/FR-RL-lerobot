"""Franka Flask server HTTP wrapper + 控制器通信辅助.

集中放 deploy 脚本里散落的 ``requests.post(URL + ...)`` 模板。所有的 deploy
脚本（pickup/wipe/pickandplace _with_backup + deploy_backup_policy）原本都
在自己脚本顶部重复声明同一套：

    URL = "http://192.168.100.1:5000/"
    def post(...): ...
    def get_state_true(): ...
    def get_state_biased(): ...
    def send_pose(...): ...
    def align_quat_sign(...): ...

抽到这里统一维护。`URL` 仍以 module-level 常量暴露，调用方需要用裸
``requests.post(URL + "open_gripper", ...)`` 这种 endpoint 时也能直接 import。
"""
from typing import List

import numpy as np
import requests

URL = "http://192.168.100.1:5000/"


def post(path: str, **kw):
    """POST 到 Franka Flask server 的相对 path（默认 timeout=2s）。"""
    return requests.post(URL + path, timeout=kw.pop("timeout", 2.0), **kw)


def get_state_true() -> dict:
    """``/getstate_true``：privileged 真实关节/位姿（不带 encoder bias）。"""
    r = post("getstate_true")
    r.raise_for_status()
    return r.json()


def get_state_biased() -> dict:
    """``/getstate``：controller 视角的关节/位姿（应用 encoder bias 后）。"""
    r = post("getstate")
    r.raise_for_status()
    return r.json()


def send_pose(target_xyz: np.ndarray, target_quat_xyzw: List[float]) -> None:
    """向 ``/pose`` 发一个 7D setpoint（xyz + quat xyzw）。

    注意：caller 负责 quat 同半球归一化（用 align_quat_sign），否则 impedance
    可能走 360° 长路。0.5s timeout，控制循环里的 stale 帧不会阻塞主线程。
    """
    pose7 = [*target_xyz.tolist(), *target_quat_xyzw]
    try:
        requests.post(URL + "pose", json={"arr": pose7}, timeout=0.5)
    except requests.exceptions.Timeout:
        pass


def align_quat_sign(q, q_ref) -> List[float]:
    """把 q 翻到跟 q_ref 同半球，避免 impedance controller 收到 sign-flipped
    quat 后做 360° slerp。每次 send_pose 之前调用。
    """
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    if float(np.dot(q, q_ref)) < 0:
        q = -q
    return q.tolist()
