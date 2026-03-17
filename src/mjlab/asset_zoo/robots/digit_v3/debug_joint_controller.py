"""Debug script: fix Digit V3 base and drive joints in the full MJLab simulation.

Runs the real training simulation with PD actuators, gravity, and contact
physics — but freezes the robot base in place to test joint control
in isolation.

Actions are OFFSETS from the default joint position, scaled by DIGIT_ACTION_SCALE:
    joint_target = default_pos + action * scale

Usage:
    # Print joint names and action indices first:
    uv run python src/mjlab/asset_zoo/robots/digit_v3/debug_joint_controller.py --print-joints

    # Then edit JOINT_TARGETS_RAD and run:
    uv run python src/mjlab/asset_zoo/robots/digit_v3/debug_joint_controller.py

    # Record a video (saved to videos/debug_joint_controller.mp4):
    uv run python src/mjlab/asset_zoo/robots/digit_v3/debug_joint_controller.py --video
"""

import os
import sys
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401 — populates task registry
from mjlab.asset_zoo.robots.digit_v3.digit_with_load import get_spec_upright_with_load
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

TASK_ID = "Mjlab-Velocity-Flat-Digit-V3"
VIDEO_LENGTH = 200  # steps to record

# Joint position targets (radians, offset from default pose)
# Keys are joint names. Values are offsets in RADIANS from the default pose.
# These are written directly to the PD controller target — not scaled.
# Run with --print-joints to see available names and indices.
JOINT_TARGETS_RAD: dict[str, float] = {
  "left_hip_roll_joint": 0.3,
  "right_hip_roll_joint": -0.3,
  "left_hip_pitch_joint": 1,
  "right_hip_pitch_joint": 1,
  "left_knee_joint": -0.5,
  "right_knee_joint": -0.5,
  "left_toe_A_joint": 0.3,
  "right_toe_A_joint": 0.3,
  "left_shoulder_pitch_joint": -0.5,
  "right_shoulder_pitch_joint": 0.5,
}

# Fixed root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
_FIXED_ROOT_HEIGHT = 1.0
_FIXED_ROOT_STATE = torch.tensor(
  [[0.0, 0.0, _FIXED_ROOT_HEIGHT, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
)


def _freeze_base(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Event function: freeze robot base at fixed pose every step."""
  robot: Entity = env.scene[asset_cfg.name]
  root_state = _FIXED_ROOT_STATE.to(env.device).expand(env.num_envs, -1)
  robot.write_root_state_to_sim(root_state)


def main() -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  record_video = "--video" in sys.argv

  env_cfg = load_env_cfg(TASK_ID, play=True)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.scene.num_envs = 1

  # Swap robot spec to the variant with a box fixed between the hands.
  env_cfg.scene.entities["robot"].spec_fn = get_spec_upright_with_load

  env_cfg.events["freeze_base"] = EventTermCfg(
    func=_freeze_base,
    mode="step",
    params={"asset_cfg": SceneEntityCfg("robot")},
  )

  render_mode = "rgb_array" if record_video else None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if record_video:
    video_folder = Path("videos")
    video_folder.mkdir(exist_ok=True)
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      name_prefix="debug_joint_controller",
      step_trigger=lambda step: step == 0,
      video_length=VIDEO_LENGTH,
      disable_logger=True,
    )
    print(
      f"[INFO] Recording {VIDEO_LENGTH} steps → {video_folder}/debug_joint_controller.mp4"
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Resolve joint names from the action manager.
  inner_env: ManagerBasedRlEnv = env.unwrapped
  joint_action_term = inner_env.action_manager.get_term("joint_pos")
  target_joint_names: list[str] = joint_action_term._target_names  # type: ignore[attr-defined]
  action_dim = len(target_joint_names)

  print(f"\nAction dimension: {action_dim}")
  print("Actuated joints (index → name):")
  for i, name in enumerate(target_joint_names):
    print(f"  [{i:2d}] {name}")

  if "--print-joints" in sys.argv:
    env.close()
    return

  # Validate JOINT_TARGETS_RAD keys.
  for name in JOINT_TARGETS_RAD:
    if name not in target_joint_names:
      print(f"WARNING: '{name}' not found — check spelling above.")

  # Get the action scale per joint.
  raw_scale = joint_action_term._scale  # type: ignore[attr-defined]
  if isinstance(raw_scale, float):
    scale = torch.full((action_dim,), raw_scale, device=device)
  else:
    scale = raw_scale[0]

  # Convert desired joint offsets (radians) → action values: action = offset / scale.
  action = torch.zeros(1, action_dim, device=device)
  for name, offset_rad in JOINT_TARGETS_RAD.items():
    if name in target_joint_names:
      idx = target_joint_names.index(name)
      action[0, idx] = offset_rad / scale[idx]

  print("\nAction values (offset_rad / scale):")
  for name, offset_rad in JOINT_TARGETS_RAD.items():
    if name in target_joint_names:
      idx = target_joint_names.index(name)
      print(
        f"  {name}: offset={offset_rad:.3f} rad, scale={scale[idx]:.4f}, action={action[0, idx]:.2f}"
      )

  class JointControlPolicy:
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
      del obs
      return action

  policy = JointControlPolicy()

  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  if has_display:
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()

  if record_video:
    print(f"[INFO] Video saved to {video_folder}/")

  env.close()


if __name__ == "__main__":
  main()
