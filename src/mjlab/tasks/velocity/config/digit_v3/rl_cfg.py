"""RL configuration for Digit v3 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def digit_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Digit v3 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      init_noise_std=1.0,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="digit_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def digit_v3_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """RL runner config for Digit v3 flat terrain.

  Uses a smaller network than the rough terrain variant: flat terrain has a
  lower-dimensional observation (no height scan), so a smaller network trains
  faster and generalises better.  Architecture mirrors the IsaacLab V4 flat
  config ([128, 128, 128] vs [512, 256, 128] for rough).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=True,
      init_noise_std=1.0,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(128, 128, 128),
      activation="elu",
      obs_normalization=True,
      stochastic=False,
      init_noise_std=1.0,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="digit_velocity_flat",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
