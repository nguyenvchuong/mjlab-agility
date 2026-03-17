"""Digit v3 velocity environment configurations."""

import math
from dataclasses import replace

from mjlab.asset_zoo.robots import (
  DIGIT_ACTION_SCALE,
  get_digit_robot_cfg,
)
from mjlab.asset_zoo.robots.digit_v3.digit_with_load import get_spec_upright_with_load
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
import mjlab.terrains as terrain_gen
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def digit_v3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Digit v3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_digit_robot_cfg()}

  # Use a focused terrain set: flat, random bumps, and sinusoidal waves.
  assert (
    cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
  )
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0,
        noise_range=(0.02, 0.10),
        noise_step=0.02,
        border_width=0.25,
      ),
      "wave_terrain": terrain_gen.HfWaveTerrainCfg(
        proportion=1.0,
        amplitude_range=(0.0, 0.2),
        num_waves=4,
        border_width=0.25,
      ),
    },
  )

  # Tighten solver settings for closed-loop linkages.
  cfg.sim.mujoco.timestep = 0.005
  cfg.sim.mujoco.iterations = 50
  cfg.sim.mujoco.ls_iterations = 50

  # Set raycast sensor frame to Digit torso.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "torso"

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_toe_roll_collision", "right_toe_roll_collision")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_toe_roll|right_toe_roll)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DIGIT_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body (actuated).
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*toe_A.*": 0.2,
    r".*toe_B.*": 0.2,
    # Lower body (passive, linkage-driven — permissive std).
    r".*tarsus.*": 0.5,
    r".*toe_pitch.*": 0.5,
    r".*toe_roll.*": 0.5,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r"shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body (actuated).
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*toe_A.*": 0.3,
    r".*toe_B.*": 0.3,
    # Lower body (passive, linkage-driven — permissive std).
    r".*tarsus.*": 0.8,
    r".*toe_pitch.*": 0.8,
    r".*toe_roll.*": 0.8,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r"shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["foot_clearance"].weight = -1.0

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # -----------------------------------------------------------------------
  # Improvements from IsaacLab Digit V4 training pipeline
  # -----------------------------------------------------------------------

  # 1. Termination penalty: strongly penalise non-timeout episode endings
  #    (e.g. falling over). This stabilises early training significantly.
  cfg.rewards["termination_penalty"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-100.0,
  )

  # 2. Enable biped air-time reward (disabled at 0.0 in the base config).
  #    Encourages alternating foot lifts and a natural gait rhythm.
  #    Weight kept low (0.25) to avoid single-foot pivot exploitation.
  cfg.rewards["air_time"].weight = 0.25

  # 3. Reduce action-rate penalty to match V4 tuning (-0.1 → -0.008).
  #    The base value is far too aggressive and suppresses leg swing.
  cfg.rewards["action_rate_l2"].weight = -0.008

  # 4. Torque penalty: discourages energy-wasteful high-torque strategies.
  cfg.rewards["dof_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-1.0e-6,
  )

  # 5. Joint acceleration penalty: encourages smooth, non-jerky motion.
  #    Applied only to actuated leg and arm joints (skip passive linkages).
  cfg.rewards["dof_acc_l2"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2,
    weight=-2.0e-7,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_hip_roll_joint",
          ".*_hip_yaw_joint",
          ".*_hip_pitch_joint",
          ".*_knee_joint",
          ".*_toe_A_joint",
          ".*_toe_B_joint",
          ".*_shoulder_roll_joint",
          ".*_shoulder_pitch_joint",
          "shoulder_yaw_joint_.*",
          ".*_elbow_joint",
        ),
      )
    },
  )

  # 6. Undesired contacts: penalise rod/tarsus/shin bodies touching the
  #    terrain (mimics V4's undesired_contacts term, weight -0.1).
  rod_tarsus_contact_cfg = ContactSensorCfg(
    name="rod_tarsus_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left|right)_(achillies_rod|toe_A_rod|toe_B_rod|tarsus|shin)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (rod_tarsus_contact_cfg,)
  cfg.rewards["undesired_contacts"] = RewardTermCfg(
    func=mdp.undesired_contacts,
    weight=-0.1,
    params={
      "sensor_name": rod_tarsus_contact_cfg.name,
      "force_threshold": 1.0,
    },
  )

  # 7. Keep fall termination at default 70°.  The V4 value of 40° is too
  #    strict for early training — short episodes give no useful gradient
  #    signal.  Tighten only after the policy can reliably stand and walk.

  # 8. Strong flat-orientation penalty (V4: weight=-2.5).
  #    Much stronger than the positive `upright` reward already present;
  #    directly penalises projected-gravity xy deviation.
  cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
    func=envs_mdp.flat_orientation_l2,
    weight=-2.5,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso",))},
  )

  # 9. Vertical velocity penalty (V4: weight=-2.0).
  #    Prevents the robot from bouncing or hopping.
  cfg.rewards["lin_vel_z_l2"] = RewardTermCfg(
    func=mdp.lin_vel_z_l2,
    weight=-2.0,
  )

  # 10. Angular velocity penalty: increase weight to match V4 (-0.1).
  cfg.rewards["body_ang_vel"].weight = -0.1

  # 11. Joint deviation penalties keep specific joints near zero (V4 values).
  cfg.rewards["joint_deviation_hip_roll"] = RewardTermCfg(
    func=mdp.joint_deviation_l1,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_roll_joint",))},
  )
  cfg.rewards["joint_deviation_hip_yaw"] = RewardTermCfg(
    func=mdp.joint_deviation_l1,
    weight=-0.2,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint",))},
  )
  cfg.rewards["joint_deviation_arms"] = RewardTermCfg(
    func=mdp.joint_deviation_l1,
    weight=-0.2,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_shoulder_roll_joint",
          ".*_shoulder_pitch_joint",
          "shoulder_yaw_joint_.*",
          ".*_elbow_joint",
        ),
      )
    },
  )

  # 12. Stand-still penalty: penalise joint deviation when command is zero.
  #     Forces the robot to hold its default upright pose when not moving.
  cfg.rewards["stand_still"] = RewardTermCfg(
    func=mdp.stand_still_joint_deviation_l1,
    weight=-0.4,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          ".*_hip_roll_joint",
          ".*_hip_yaw_joint",
          ".*_hip_pitch_joint",
          ".*_knee_joint",
          ".*_toe_A_joint",
          ".*_toe_B_joint",
        ),
      ),
    },
  )

  # 13. Simplify the command space so the robot must learn to walk forward
  #     before being asked to turn.  Lateral velocity and heading control are
  #     disabled entirely; angular velocity is reintroduced by the curriculum
  #     at step 6000*24.  Without this, the robot exploits spinning on one
  #     foot to satisfy angular commands without ever walking.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.1
  # Angular tracking is irrelevant while ang_vel_z=0; reduce its weight to
  # avoid it dominating once turning is re-introduced by the curriculum.
  cfg.rewards["track_angular_velocity"].weight = 0.5

  # 14. Override velocity curriculum (train from scratch, 20k iters):
  #   0-2k:   stand stable before introducing any walking signal
  #   2k-6k:  forward only to build a clean walking gait
  #   6k-15k: symmetric forward+backward for equal exposure
  #   15k+:   add turning once both directions are solid
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (0.0, 0.0)},
    {"step": 2000 * 24, "lin_vel_x": (0.0, 1.0)},
    {"step": 6000 * 24, "lin_vel_x": (0.0, 1.0), "ang_vel_z": (-0.3, 0.3)},
  ]

  # 15. Disable initial joint-position randomisation — Digit V3 has closed
  #     kinematic loops; randomising joints violates the loop constraints.
  if "reset_joints" in cfg.events:
    cfg.events["reset_joints"].params["position_range"] = (1.0, 1.0)

  # 15. Disable push_robot (matches V4). Pushing before the robot can stand
  #     only destabilises early training.
  cfg.events.pop("push_robot", None)

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg


def digit_v3_flat_with_load_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat terrain config with a bar fixed between the robot's hands."""
  cfg = digit_v3_flat_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_spec_upright_with_load
  return cfg


def digit_v3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Digit v3 flat terrain velocity configuration."""
  cfg = digit_v3_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum.
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  return cfg
