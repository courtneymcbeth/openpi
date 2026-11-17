"""Converts retargeted UR5e episodes recorded in LW-BenchHub to a LeRobot dataset for openpi.

The input is the output directory of LW-BenchHub's `retarget_piper_to_ur5e.py`: one folder per task with
one HDF5 file per successful episode, recorded at 20 Hz.

    uv run examples/ur5e_lwlibero/convert_lw_hdf5_to_lerobot.py --raw-dir /hdd/vla_data/ur5e_raw

The dataset is written to `$HF_LEROBOT_HOME/<repo_id>`. Features:

    image, wrist_image   224x224x3 third-person and wrist camera
    joints               (6,) arm joint positions in radians
    gripper              (1,) gripper closure, 0 = open ... 1 = closed
    actions              (7,) absolute joint position targets followed by the gripper command
    eef_pose             (7,) grasp point pose in the robot base frame [x y z qw qx qy qz]
                         (not used for joint-space training; kept for end-effector-space experiments)

`actions[t]` is the command applied after observing frame t.
"""

import json
import pathlib
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from openpi.training.config import UR5E_LWLIBERO_REPO_ID

FPS = 20
IMAGE_SHAPE = (224, 224, 3)


def main(
    raw_dir: str,
    *,
    repo_id: str = UR5E_LWLIBERO_REPO_ID,
    tasks: tuple[str, ...] = (),
    max_episodes_per_task: int | None = None,
    push_to_hub: bool = False,
):
    """Converts all episodes below `raw_dir`.

    Args:
        raw_dir: Directory with one sub-directory of episode HDF5 files per task.
        repo_id: Name of the dataset below `$HF_LEROBOT_HOME` (and on the Hugging Face Hub).
        tasks: Only convert these tasks (default: all).
        max_episodes_per_task: Only convert the first N episodes of each task.
        push_to_hub: Upload the dataset to the Hugging Face Hub as a private dataset when done.
    """
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=FPS,
        features={
            "image": {"dtype": "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
            "joints": {"dtype": "float32", "shape": (6,), "names": ["joints"]},
            "gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "eef_pose": {"dtype": "float32", "shape": (7,), "names": ["eef_pose"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    task_dirs = sorted(p for p in pathlib.Path(raw_dir).iterdir() if p.is_dir() and (not tasks or p.name in tasks))
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found in {raw_dir}")
    num_episodes, num_frames = 0, 0
    # Which source demonstration every episode comes from; LW-BenchHub's `replay_ur5e_dataset.py` uses this
    # to restore the scene an episode started from.
    sources, task_of_lang = [], {}
    for task_dir in task_dirs:
        files = sorted(task_dir.glob("*.hdf5"))[:max_episodes_per_task]
        for file in tqdm.tqdm(files, desc=task_dir.name[:50]):
            with h5py.File(file, "r") as f:
                if abs(float(f.attrs["fps"]) - FPS) > 1e-6:
                    raise ValueError(f"{file} was recorded at {f.attrs['fps']} Hz, expected {FPS} Hz")
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
    with open(output_path / "meta" / "ur5e_source_episodes.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in sources)
    print(f"Wrote {num_episodes} episodes ({num_frames} frames) from {len(task_dirs)} tasks to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["libero", "ur5e", "lw-benchhub"], private=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
