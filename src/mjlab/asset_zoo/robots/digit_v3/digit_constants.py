"""Agility Robotics Digit v3 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

DIGIT_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "digit_v3" / "digit.xml"
assert DIGIT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, DIGIT_XML.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(DIGIT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Per-joint overrides applied on top of the all-zero upright pose.
# Both left and right hip_roll share the same world-frame axis, so the same
# sign moves both legs in the same direction.  Tune these values with:
#   uv run python src/mjlab/asset_zoo/robots/digit_v3/digit_constants.py
_UPRIGHT_JOINT_OVERRIDES: dict[str, float] = {
  "left_hip_roll_joint": 0.3,
  "right_hip_roll_joint": -0.3,
  "left_shoulder_pitch_joint": 0.5,
  "right_shoulder_pitch_joint": -0.5,
  "left_elbow_joint": -0.5,
  "right_elbow_joint": 0.5,
}


def get_spec_upright() -> mujoco.MjSpec:
  """Get Digit spec with all hinge joints zeroed (upright standing pose).

  The Digit V3 MJCF is designed so that zero hinge-joint positions produce a
  symmetric, upright stance.  The Agility 'standing' keyframe uses non-zero
  hip_pitch/knee/tarsus values whose sign conventions differ between left and
  right sides (mirrored joint axes), which causes an asymmetric initial pose.
  Zeroing all hinges avoids that problem entirely.  Ball joints in the linkage
  mechanism are kept at identity quaternions.  _UPRIGHT_JOINT_OVERRIDES can
  fine-tune specific joints after zeroing.
  """
  spec = get_spec()
  model = spec.compile()
  key_qpos = list(model.key("standing").qpos)
  for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
      key_qpos[model.jnt_qposadr[i]] = 0.0
  for joint_name, angle in _UPRIGHT_JOINT_OVERRIDES.items():
    key_qpos[model.jnt_qposadr[model.joint(joint_name).id]] = angle
  spec.keys[0].qpos = key_qpos
  return spec


##
# Actuator config.
##

# Natural frequency and damping ratio for PD control tuning.
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0

# Armature values from digit.xml joint definitions.
_ARMATURE_A = 0.1728  # hip_roll, shoulder_roll, shoulder_pitch, elbow
_ARMATURE_B = 0.0675  # hip_yaw, shoulder_yaw
_ARMATURE_C = 0.120576  # hip_pitch, knee
_ARMATURE_D = 0.035  # toe_A, toe_B

_STIFFNESS_A = _ARMATURE_A * _NATURAL_FREQ**2
_STIFFNESS_B = _ARMATURE_B * _NATURAL_FREQ**2
_STIFFNESS_C = _ARMATURE_C * _NATURAL_FREQ**2
_STIFFNESS_D = _ARMATURE_D * _NATURAL_FREQ**2

_DAMPING_A = 2.0 * _DAMPING_RATIO * _ARMATURE_A * _NATURAL_FREQ
_DAMPING_B = 2.0 * _DAMPING_RATIO * _ARMATURE_B * _NATURAL_FREQ
_DAMPING_C = 2.0 * _DAMPING_RATIO * _ARMATURE_C * _NATURAL_FREQ
_DAMPING_D = 2.0 * _DAMPING_RATIO * _ARMATURE_D * _NATURAL_FREQ

# Effort limits (Nm) — approximate values for Digit v3.
_EFFORT_A = 80.0  # hip_roll, shoulder_roll, shoulder_pitch, elbow
_EFFORT_B = 80.0  # hip_yaw, shoulder_yaw
_EFFORT_C = 80.0  # hip_pitch, knee
_EFFORT_D = 50.0  # toe_A, toe_B

DIGIT_ACTUATOR_A = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_pitch_joint",
    ".*_elbow_joint",
  ),
  stiffness=_STIFFNESS_A,
  damping=_DAMPING_A,
  effort_limit=_EFFORT_A,
  armature=_ARMATURE_A,
)

DIGIT_ACTUATOR_B = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    "shoulder_yaw_joint_.*",
  ),
  stiffness=_STIFFNESS_B,
  damping=_DAMPING_B,
  effort_limit=_EFFORT_B,
  armature=_ARMATURE_B,
)

DIGIT_ACTUATOR_C = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
    ".*_knee_joint",
  ),
  stiffness=_STIFFNESS_C,
  damping=_DAMPING_C,
  effort_limit=_EFFORT_C,
  armature=_ARMATURE_C,
)

DIGIT_ACTUATOR_D = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_toe_A_joint",
    ".*_toe_B_joint",
  ),
  stiffness=_STIFFNESS_D,
  damping=_DAMPING_D,
  effort_limit=_EFFORT_D,
  armature=_ARMATURE_D,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.0),
  joint_pos=None,  # Use 'standing' keyframe from digit.xml (ball joints need qpos format).
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Foot geoms get condim=3 with friction; body geoms use XML defaults (condim=3).
FEET_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_toe_roll_collision$",),
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

DIGIT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DIGIT_ACTUATOR_A,
    DIGIT_ACTUATOR_B,
    DIGIT_ACTUATOR_C,
    DIGIT_ACTUATOR_D,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_digit_robot_cfg() -> EntityCfg:
  """Get a fresh Digit v3 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FEET_COLLISION,),
    spec_fn=get_spec_upright,
    articulation=DIGIT_ARTICULATION,
  )


DIGIT_ACTION_SCALE: dict[str, float] = {}
for _a in DIGIT_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    DIGIT_ACTION_SCALE[_n] = 0.25 * _e / _s


