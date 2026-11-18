# pi0.5 LoRA fine-tuning for a UR5e on LW-BenchHub LIBERO tasks

Fine-tunes the pi0.5 base model with LoRA for a single-arm UR5e with a Robotiq 2F-85 gripper on the
LIBERO tasks as implemented in [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) (Isaac Lab).
The configs are sized for one 24 GB GPU.

## Train

```bash
git clone -b ur5e-lwlibero https://github.com/courtneymcbeth/openpi.git openpi && cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# The dataset is a private Hugging Face dataset: log in once, then download and unpack it (2.9 GB).
uv run huggingface-cli login
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
mkdir -p $HF_LEROBOT_HOME/courtneymcbeth
uv run huggingface-cli download courtneymcbeth/ur5e_lwlibero ur5e_lwlibero.tar --repo-type dataset \
    --local-dir $HF_LEROBOT_HOME/courtneymcbeth
tar -xf $HF_LEROBOT_HOME/courtneymcbeth/ur5e_lwlibero.tar -C $HF_LEROBOT_HOME/courtneymcbeth
rm $HF_LEROBOT_HOME/courtneymcbeth/ur5e_lwlibero.tar

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5e_lwlibero_lora \
    --exp-name=ur5e_run1 --overwrite
```

The dataset goes to `$HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`); the first training run
downloads the pi0.5 base weights (about 12 GB) to `$OPENPI_DATA_HOME` (default `~/.cache/openpi`). Set
both to a scratch file system if your home directory is small, in every shell that runs training.

Download the archive as shown: without a local copy the training code fetches the dataset file by file,
and its 3,937 files exceed Hugging Face's limit of 1,000 API requests per five minutes (free accounts),
so that download stops with "429 Too Many Requests". Checkpoints go to
`checkpoints/pi05_ur5e_lwlibero_lora/<exp-name>/` every 1,000 steps. Use `--resume` instead of
`--overwrite` to continue a run. Training logs to Weights & Biases; add `--no-wandb-enabled` to turn that
off.

Normalization statistics for the dataset are checked in
(`examples/ur5e_lwlibero/assets/ur5e_lwlibero/norm_stats.json`), so there is no separate
`compute_norm_stats` step. To recompute them after changing the dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_ur5e_lwlibero_lora
cp assets/pi05_ur5e_lwlibero_lora/<repo_id>/norm_stats.json examples/ur5e_lwlibero/assets/ur5e_lwlibero/
```

### Configs

| Config | What it is |
| --- | --- |
| `pi05_ur5e_lwlibero_lora` | Recommended. LoRA on PaliGemma and the action expert, batch size 16, 60k steps. |
| `pi05_ur5e_lwlibero_lora_frozen_vision` | Same, with the image encoder frozen as well. Use if the first runs out of memory. |
| `pi05_ur5e_lwlibero_lora_prestats` | Same as the first, but normalizes with the UR5e statistics of the pi0.5 base model instead of this dataset's. For comparison. |
| `pi05_ur5e_lwlibero_debug` | Tiny dummy model on the real dataset. Checks the data pipeline in a minute, on any machine. |
| `pi05_ur5e_lwlibero_base` | Not for training: the pi0.5 base model as it is, for a baseline in evaluation. |

If training runs out of GPU memory, in this order: `--batch-size 8`, then
`pi05_ur5e_lwlibero_lora_frozen_vision`. The memory use of pi0.5 LoRA on 24 GB has not been measured
here: the dataset machine has a 16 GB GPU.

Model settings worth knowing: 20-step action chunks (one second at 20 Hz), joint targets predicted
relative to the current joint angles, and the joint state fed to the model as discrete tokens in the
prompt (the pi0.5 default, and the only way pi0.5 sees the state).

## Dataset

`courtneymcbeth/ur5e_lwlibero` (private) in LeRobot v2.1 format, 20 Hz: 1,310 episodes, 549,000 frames
(7.6 hours). The two cameras are stored as video (AV1, one file per episode and camera); everything
else is in parquet files.

| Suite | Episodes | Tasks with at least 20 episodes (of 10) |
| --- | --- | --- |
| libero_spatial | 483 | 10 |
| libero_goal | 362 | 9 |
| libero_object | 234 | 6 |
| libero_10 | 231 | 6 |

Coverage is uneven: 31 of the 40 tasks have 20 to 50 episodes, 5 have 4 to 19, and 4 have one or none
(`LOPickUpTheKetchup...`, `LOPickUpTheSaladDressing...`, `L10L2PutBothTheCreamCheeseBoxAndTheButter...`,
`L10L6PutWhiteMugOnPlate...`). The per-task counts are in the report that `visualize_dataset.py` writes,
and the known causes are in the LW-BenchHub fork (`lw_benchhub/scripts/retarget/README.md`).

| Key | Shape | Meaning |
| --- | --- | --- |
| `image` | 224x224x3 | Third-person camera in front of the workspace, fixed to the robot base |
| `wrist_image` | 224x224x3 | Camera on the gripper |
| `joints` | 6 | Arm joint angles in radians (shoulder pan, shoulder lift, elbow, wrist 1-3) |
| `gripper` | 1 | Gripper closure, 0 = open, 1 = closed |
| `actions` | 7 | Absolute joint position targets, then the gripper command (0 = open, 1 = closed) |
| `eef_pose` | 7 | Grasp point pose in the robot base frame, `[x y z qw qx qy qz]`. Not used by these configs. |

These are the conventions of the `ur5e` embodiment in the pi0 and pi0.5 base models (see
`docs/norm_stats.md`).

Video frames are decoded with `torchcodec` if it can be imported and with `pyav` otherwise
(`src/openpi/training/data_loader.py`); `torchcodec` needs FFmpeg libraries on the machine, `pyav`
brings its own. To store PNG images instead (about 25 times larger), convert with
`--image-storage image`.

### Where the data comes from

LW-BenchHub has no UR5e and no single-arm demonstrations of its LIBERO tasks. The episodes here are
LightWheel's published bimanual Double-Piper demonstrations
(`LightwheelAI/lightwheel_tasks`, `lightwheel_libero_tasks_piper`) replayed by a UR5e that was added to
LW-BenchHub:

1. For each demonstration, the path of the Piper gripper that does the work is extracted and re-timed.
   Demonstrations that use both arms one after the other are replayed in sequence by the single arm.
   Demonstrations that need both arms at once (a hand-over in mid-air) cannot be replayed and are
   missing from the dataset.
2. UR5e joint angles are solved along the path, with the UR5e's fingertips following the Piper's.
3. The scene is rebuilt with the same objects in the same places, the UR5e executes the joint path, and
   the episode is kept only if the task's own success check passes and no other object was knocked away.
   If it fails, the same demonstration is tried again with a slightly different grasp (deeper,
   shallower, or shifted sideways onto the object).

So every episode in the dataset solved its task in simulation, and the actions are exactly the commands
the simulator executed. The code is in the `ur5e` branch of the LW-BenchHub fork, under
`lw_benchhub/scripts/retarget/` and `lw_benchhub/core/robots/universal_robots/`.

### Rebuild the dataset

On a machine with the LW-BenchHub fork set up and the raw episodes from its
`retarget_piper_to_ur5e.py`:

```bash
export HF_LEROBOT_HOME=/path/with/space/lerobot
uv run examples/ur5e_lwlibero/convert_lw_hdf5_to_lerobot.py --raw-dir /path/to/ur5e_raw
uv run examples/ur5e_lwlibero/visualize_dataset.py --out-dir /path/to/viz
```

`visualize_dataset.py` writes `report.html` (coverage per task, episode lengths, state and action
distributions, example trajectories, camera frames for every task) and sample videos.

## Evaluate

A checkpoint is evaluated closed loop in LW-BenchHub, in the environment the data was recorded in. Serve
it from this repo (inference needs 8.5 GB of GPU memory; the checkpoint's `params` and `assets`
directories are enough):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ur5e_lwlibero_lora --policy.dir=checkpoints/pi05_ur5e_lwlibero_lora/ur5e_run1/<step>
```

The simulator side (episode list, runner, report with a video of every episode) is in the `ur5e` branch
of the LW-BenchHub fork; see "Evaluating a policy" in `lw_benchhub/scripts/retarget/README.md`. The
server expects `base_rgb` and `wrist_rgb` (224x224 uint8), `joints` (6), `gripper` (1) and `prompt`, and
returns absolute joint targets plus gripper commands (`src/openpi/policies/ur5e_policy.py`).

`pi05_ur5e_lwlibero_base` is the pi0.5 base model without fine-tuning, for a zero-shot baseline: serve it
with `--policy.dir` set to the downloaded `pi05_base` checkpoint
(`$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base`).
