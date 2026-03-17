from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  digit_v3_flat_env_cfg,
  digit_v3_flat_with_load_env_cfg,
  digit_v3_rough_env_cfg,
)
from .rl_cfg import digit_v3_flat_ppo_runner_cfg, digit_v3_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Digit-V3",
  env_cfg=digit_v3_rough_env_cfg(),
  play_env_cfg=digit_v3_rough_env_cfg(play=True),
  rl_cfg=digit_v3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Digit-V3",
  env_cfg=digit_v3_flat_env_cfg(),
  play_env_cfg=digit_v3_flat_env_cfg(play=True),
  rl_cfg=digit_v3_flat_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Digit-V3-Load",
  env_cfg=digit_v3_flat_with_load_env_cfg(),
  play_env_cfg=digit_v3_flat_with_load_env_cfg(play=True),
  rl_cfg=digit_v3_flat_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
