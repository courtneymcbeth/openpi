"""Converts retargeted episodes recorded in LW-BenchHub (UR5e or Franka Panda) to a LeRobot dataset for openpi.

The input is the output directory of LW-BenchHub's `retarget_piper_to_ur5e.py`: one folder per task with
one HDF5 file per successful episode, recorded at 20 Hz.

    uv run examples/ur5e_lwlibero/convert_lw_hdf5_to_lerobot.py --raw-dir /hdd/vla_data/ur5e_raw
    uv run examples/ur5e_lwlibero/convert_lw_hdf5_to_lerobot.py --raw-dir /hdd/vla_data/panda_raw --robot panda

The dataset is written to `$HF_LEROBOT_HOME/<repo_id>`. Features (J = 6 arm joints for the UR5e, 7 for the Panda):

    image, wrist_image   224x224x3 third-person and wrist camera
    joints               (J,) arm joint positions in radians
    gripper              (1,) gripper closure, 0 = open ... 1 = closed
    actions              (J + 1,) absolute joint position targets followed by the gripper command
    eef_pose             (7,) grasp point pose in the robot base frame [x y z qw qx qy qz]
                         (not used for joint-space training; kept for end-effector-space experiments)

`actions[t]` is the command applied after observing frame t.
"""

import json
import pathlib
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from openpi.training.config import PANDA_LWLIBERO_REPO_ID
from openpi.training.config import UR5E_LWLIBERO_REPO_ID

FPS = 20
IMAGE_SHAPE = (224, 224, 3)
# Per robot: default dataset name, the name LW-BenchHub records in the episode files, number of arm joints.
ROBOTS = {
    "ur5e": (UR5E_LWLIBERO_REPO_ID, "UR5e-Joint", 6),
    "panda": (PANDA_LWLIBERO_REPO_ID, "Panda-Joint", 7),
}


def main(
    raw_dir: str,
    *,
    robot: Literal["ur5e", "panda"] = "ur5e",
    repo_id: str | None = None,
    tasks: tuple[str, ...] = (),
    max_episodes_per_task: int | None = None,
    image_storage: Literal["video", "image"] = "video",
    push_to_hub: bool = False,
):
    """Converts all episodes below `raw_dir`.

    Args:
        raw_dir: Directory with one sub-directory of episode HDF5 files per task.
        robot: The robot the episodes were recorded with.
        repo_id: Name of the dataset below `$HF_LEROBOT_HOME` (and on the Hugging Face Hub). Default: the
            robot's dataset, `courtneymcbeth/<robot>_lwlibero`.
        tasks: Only convert these tasks (default: all).
        max_episodes_per_task: Only convert the first N episodes of each task.
        image_storage: "video" stores each camera of each episode as an AV1 video (about 2 KB per frame);
            "image" stores lossless PNG frames (about 50 KB per frame, roughly 90 GB for the full dataset).
        push_to_hub: Upload the dataset to the Hugging Face Hub as a private dataset when done.
    """
    default_repo_id, sim_robot, arm_dof = ROBOTS[robot]
    repo_id = repo_id or default_repo_id
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot,
        fps=FPS,
        features={
            "image": {"dtype": image_storage, "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": image_storage, "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
            "joints": {"dtype": "float32", "shape": (arm_dof,), "names": ["joints"]},
            "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
            "actions": {"dtype": "float32", "shape": (arm_dof + 1,), "names": ["actions"]},
            "eef_pose": {"dtype": "float32", "shape": (7,), "names": ["eef_pose"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Task directories are the ones that hold episode files (not `logs`, `asset_cache`, ...).
    task_dirs = sorted(
        p
        for p in pathlib.Path(raw_dir).iterdir()
        if p.is_dir() and (not tasks or p.name in tasks) and any(p.glob("*.hdf5"))
    )
    if not task_dirs:
        raise FileNotFoundError(f"No task directories with episodes found in {raw_dir}")
    num_episodes, num_frames = 0, 0
    # Which source demonstration every episode comes from; LW-BenchHub's `replay_ur5e_dataset.py` uses this
    # to restore the scene an episode started from (and the file's name to tell which robot to build).
    sources, task_of_lang = [], {}
    for task_dir in task_dirs:
        files = sorted(task_dir.glob("*.hdf5"))[:max_episodes_per_task]
        for file in tqdm.tqdm(files, desc=task_dir.name[:50]):
            with h5py.File(file, "r") as f:
                if abs(float(f.attrs["fps"]) - FPS) > 1e-6:
                    raise ValueError(f"{file} was recorded at {f.attrs['fps']} Hz, expected {FPS} Hz")
                if str(f.attrs["robot"]) != sim_robot:
                    raise ValueError(f"{file} was recorded with {f.attrs['robot']}, not {sim_robot}; see --robot")
                lang = str(f.attrs["lang"])
                if task_of_lang.setdefault(lang, task_dir.name) != task_dir.name:
                    raise ValueError(
                        f"Tasks {task_of_lang[lang]} and {task_dir.name} have the same instruction: {lang!r}"
                    )
                sources.append(
                    {
                        "episode_index": num_episodes,
                        "task": task_dir.name,
                        "raw_file": file.name,
                        "source_file": str(f.attrs["source_file"]),
                        "source_episode": str(f.attrs["source_episode"]),
                        "robot_root_pose": [float(v) for v in f.attrs["robot_root_pose"]],
                    }
                )
                image, wrist = f["obs/agentview_rgb"][:], f["obs/wrist_rgb"][:]
                joints, gripper = f["obs/arm_joint_pos"][:], f["obs/gripper_closure"][:]
                eef_pose, actions = f["obs/eef_pose_b"][:], f["actions"][:]
            for t in range(len(actions)):
                dataset.add_frame(
                    {
                        "image": image[t],
                        "wrist_image": wrist[t],
                        "joints": joints[t].astype(np.float32),
                        "gripper": gripper[t].astype(np.float32),
                        "actions": actions[t].astype(np.float32),
                        "eef_pose": eef_pose[t].astype(np.float32),
                        "task": lang,
                    }
                )
            dataset.save_episode()
            num_episodes += 1
            num_frames += len(actions)
    with open(output_path / "meta" / f"{robot}_source_episodes.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in sources)
    print(f"Wrote {num_episodes} episodes ({num_frames} frames) from {len(task_dirs)} tasks to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["libero", robot, "lw-benchhub"], private=True, license=None)


if __name__ == "__main__":
    tyro.cli(main)
