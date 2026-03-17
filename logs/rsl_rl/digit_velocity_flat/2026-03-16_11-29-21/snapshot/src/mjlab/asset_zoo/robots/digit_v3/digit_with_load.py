"""Digit v3 spec with a bar rigidly attached between the hands."""

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.digit_v3.digit_constants import get_spec_upright

_BAR_RADIUS = 0.25  # metres
_BAR_LENGTH = 0.3  # metres — change this to resize
_BAR_MASS = 10
_BAR_RGBA = np.array([0.8, 0.2, 0.2, 1.0], dtype=np.float32)

# Direction computed from original endpoints; length controlled by _BAR_LENGTH.
_midpoint = np.array([0.3827, -0.0331, -0.2213])
_direction = np.array([-0.0244, 0.059, -0.3619])
_direction /= np.linalg.norm(_direction)
_BAR_FROM = _midpoint - _direction * _BAR_LENGTH / 2
_BAR_TO = _midpoint + _direction * _BAR_LENGTH / 2


def _find_body(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody | None:
  if root.name == name:
    return root
  child = root.first_body()
  while child:
    result = _find_body(child, name)
    if result is not None:
      return result
    child = root.next_body(child)
  return None


def get_spec_upright_with_load() -> mujoco.MjSpec:
  """Digit v3 upright spec with a bar fixed between the hands."""
  spec = get_spec_upright()

  elbow = _find_body(spec.worldbody, "left_elbow")
  assert elbow is not None, "left_elbow body not found in spec"

  bar_body = elbow.add_body()
  bar_body.name = "load"

  geom = bar_body.add_geom()
  geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
  geom.fromto = np.concatenate([_BAR_FROM, _BAR_TO])
  geom.size = np.array([_BAR_RADIUS, 0.0, 0.0])
  geom.rgba = _BAR_RGBA
  geom.mass = _BAR_MASS
  geom.condim = 4
  geom.friction = (1.0, 0.005, 0.0001)

  return spec


if __name__ == "__main__":
  import mujoco.viewer as viewer

  model = get_spec_upright_with_load().compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("standing").id)
  with viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [0, 0, 0.8]
    v.cam.distance = 3.0
    print("Showing training reset pose with bar — close window to exit.")
    while v.is_running():
      v.sync()
