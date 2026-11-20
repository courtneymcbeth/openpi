"""Policy transforms for a single arm with a parallel gripper and joint-position actions: the UR5e with a
Robotiq 2F-85 gripper and, with `action_dim=8`, the 7-joint Franka Panda with the Franka Hand.

Conventions for the UR5e (these match the `ur5e` embodiment the pi0 / pi0.5 base models were pretrained on,
see docs/norm_stats.md); the Panda is the same with seven arm joints (`panda_joint1` ... `panda_joint7`),
which is the layout the base models use for 7-DoF arms:

- state:   6 arm joint positions in radians (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
           followed by the gripper closure in [0, 1], where 0 is fully open and 1 is fully closed.
- actions: 6 absolute joint position targets in radians followed by the gripper command in [0, 1].
           The joint dimensions are converted to deltas relative to the current state by the data config.
- images:  one third-person camera and one wrist camera.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Number of action dimensions of the robot: 6 arm joints + 1 gripper.
UR5E_ACTION_DIM = 7


def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5e policy."""
    return {
        "joints": np.random.rand(6).astype(np.float32),
        "gripper": np.random.rand(1).astype(np.float32),
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(transforms.DataTransformFn):
    """Converts UR5e observations (dataset samples or live observations) to the model input format."""

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # The state is the concatenation of the arm joints and the gripper closure. Padding to the
        # model's state dimension happens later, in the model transforms.
        joints = np.asarray(data["joints"], dtype=np.float32)
        gripper = np.asarray(data["gripper"], dtype=np.float32).reshape(*joints.shape[:-1], 1)
        state = np.concatenate([joints, gripper], axis=-1)

        # LeRobot stores images as float32 (C,H,W); live observations are already uint8 (H,W,C).
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # There is no second wrist camera: pad with zeros and mask it out.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Padding images are only masked for pi0 / pi0.5, not for pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the robot's action format. Used for inference only."""

    # Number of action dimensions of the robot: arm joints + 1 gripper.
    action_dim: int = UR5E_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # The model predicts padded actions; only the first `action_dim` dimensions are meaningful.
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
