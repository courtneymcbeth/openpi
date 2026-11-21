# pi0.5 LoRA fine-tuning for a Franka Panda on LW-BenchHub LIBERO tasks

Fine-tunes the pi0.5 base model with LoRA for a single-arm Franka Panda with the Franka Hand on the
LIBERO tasks as implemented in [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) (Isaac Lab).
It is the Panda counterpart of [`examples/ur5e_lwlibero`](../ur5e_lwlibero/README.md): same tasks, same
source demonstrations, same cameras, same training settings, one more joint. The scripts are shared and
live in `examples/ur5e_lwlibero/`. The configs are sized for one 24 GB GPU.

## Train

```bash
git clone -b panda-lwlibero https://github.com/courtneymcbeth/openpi.git openpi && cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# The dataset is a private Hugging Face dataset: log in once, then download and unpack it (3.0 GB).
uv run huggingface-cli login
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
mkdir -p $HF_LEROBOT_HOME/courtneymcbeth
uv run huggingface-cli download courtneymcbeth/panda_lwlibero panda_lwlibero.tar --repo-type dataset \
    --local-dir $HF_LEROBOT_HOME/courtneymcbeth
tar -xf $HF_LEROBOT_HOME/courtneymcbeth/panda_lwlibero.tar -C $HF_LEROBOT_HOME/courtneymcbeth
rm $HF_LEROBOT_HOME/courtneymcbeth/panda_lwlibero.tar

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_panda_lwlibero_lora \
    --exp-name=panda_run1 --overwrite
```

In a clone that already has the UR5e branch, `git fetch origin && git checkout panda-lwlibero` replaces
the first three lines: the branch contains the UR5e configs too, and the Python environment is the same.

The dataset goes to `$HF_LEROBOT_HOME` (default `~/.cache/huggingface/lerobot`), next to the UR5e
dataset if that is there; the pi0.5 base weights (about 12 GB) are in `$OPENPI_DATA_HOME` (default
`~/.cache/openpi`) and are downloaded by the first training run if missing. Set both to a scratch file
system if your home directory is small, in every shell that runs training.

Download the archive as shown: without a local copy the training code fetches the dataset file by file,
and its 4,754 files exceed Hugging Face's limit of 1,000 API requests per five minutes (free accounts),
so that download stops with "429 Too Many Requests". Checkpoints go to
`checkpoints/pi05_panda_lwlibero_lora/<exp-name>/` every 1,000 steps. Use `--resume` instead of
`--overwrite` to continue a run. Training logs to Weights & Biases; add `--no-wandb-enabled` to turn that
off.

Normalization statistics for the dataset are checked in
(`examples/panda_lwlibero/assets/panda_lwlibero/norm_stats.json`), so there is no separate
`compute_norm_stats` step. To recompute them after changing the dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_panda_lwlibero_lora
cp assets/pi05_panda_lwlibero_lora/<repo_id>/norm_stats.json examples/panda_lwlibero/assets/panda_lwlibero/
```

### Configs

| Config | What it is |
| --- | --- |
| `pi05_panda_lwlibero_lora` | Recommended. LoRA on PaliGemma and the action expert, batch size 16, 60k steps. |
| `pi05_panda_lwlibero_lora_frozen_vision` | Same, with the image encoder frozen as well. Use if the first runs out of memory. |
| `pi05_panda_lwlibero_lora_prestats` | Same as the first, but normalizes with the `franka` statistics of the pi0.5 base model instead of this dataset's. For comparison. |
| `pi05_panda_lwlibero_debug` | Tiny dummy model on the real dataset. Checks the data pipeline in a minute, on any machine. |
| `pi05_panda_lwlibero_base` | Not for training: the pi0.5 base model as it is, for a baseline in evaluation. |

The settings are those of the UR5e configs (20-step action chunks, joint targets predicted relative to
the current joint angles, joint state as discrete tokens in the prompt); see the UR5e README for the
out-of-memory fallbacks. Only the data config differs: 7 arm joints, so 8-dimensional states and
actions. None of these configs has been trained yet: only `pi05_panda_lwlibero_debug` was run, on the
dataset machine.

## Dataset

`courtneymcbeth/panda_lwlibero` (private) in LeRobot v2.1 format, 20 Hz: 1,583 episodes, 660,335 frames
(9.2 hours). The two cameras are stored as video (AV1, one file per episode and camera); everything
else is in parquet files.

| Suite | Episodes | Tasks with at least 20 episodes (of 10) | UR5e dataset, for comparison |
| --- | --- | --- | --- |
| libero_spatial | 416 | 8 | 483 |
| libero_goal | 425 | 10 | 362 |
| libero_object | 372 | 8 | 234 |
| libero_10 | 370 | 8 | 231 |

34 of the 40 tasks have 25 to 50 episodes, 2 have 15, and 4 have none:

| Task | Episodes | Why |
| --- | --- | --- |
| `LSPickUpBlackBowlInTopDrawerOfWoodenCabinetAndPlaceItOnPlate` | 15 | The gripper closes about a centimetre short of the bowl's rim: the arm is held back there, most likely by the wide Franka Hand resting on the drawer. |
| `LSPickUpTheBlackBowlNextToTheCookieBoxAndPlaceItOnThePlate` | 15 | The bowl is out of the Panda's reach in 32 of the 50 demonstrations. |
| `LOPickUpTheKetchupAndPlaceItInTheBasket`, `LOPickUpTheSaladDressingAndPlaceItInTheBasket` | 0 | The demonstrations pass the bottle from one arm to the other in mid-air. |
| `L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket`, `L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket` | 0 | The flat cream cheese box is not picked up. |

The per-task counts are in the report that `visualize_dataset.py` writes, and the retargeting results
per task are in the LW-BenchHub fork (`lw_benchhub/scripts/retarget/README.md`, "Franka Panda").

| Key | Shape | Meaning |
| --- | --- | --- |
| `image` | 224x224x3 | Third-person camera in front of the workspace, fixed to the robot base. Same place in the scene as in the UR5e dataset. |
| `wrist_image` | 224x224x3 | Camera on the Franka Hand |
| `joints` | 7 | Arm joint angles in radians (`panda_joint1` to `panda_joint7`) |
| `gripper` | 1 | Gripper closure, 0 = open (8 cm), 1 = closed |
| `actions` | 8 | Absolute joint position targets, then the gripper command (0 = open, 1 = closed) |
| `eef_pose` | 7 | Grasp point pose in the robot base frame, `[x y z qw qx qy qz]`. Not used by these configs. |

Joints in dimensions 0 to 6 and the gripper in dimension 7 is the layout of the `franka` embodiment in
the pi0 and pi0.5 base models (see `docs/norm_stats.md`).

Two things to know about the motions:

- **Arm posture.** The Panda's elbow is free to swivel for a given gripper pose. The episodes use one
  rule: the forearm rolls away from the side of the base the gripper is on. A few tasks need other
  postures (the mug is put into the microwave with the upper arm rolled as well).
- **Pushing against furniture.** Where a demonstration pushes a drawer, the microwave door or the plate
  to its stop, the joint targets run ahead of what the arm can do, by more than 0.15 rad in 0.7% of
  the frames (128 episodes).

### Where the data comes from

The same 2,000 LightWheel Double-Piper demonstrations as the UR5e dataset, replayed by a Franka Panda
that was added to LW-BenchHub, with the same three steps (gripper path of the working arm, joint angles
along it, execution in the rebuilt scene; an episode is kept only if the task's own success check passes
and no other object was knocked away). So every episode solved its task in simulation, and the actions
are exactly the commands the simulator executed. The UR5e README describes the steps; what differs for
the Panda (joint paths for seven joints, base placement, gripper, wrist camera) is described in the
`panda` branch of the LW-BenchHub fork, under `lw_benchhub/scripts/retarget/` and
`lw_benchhub/core/robots/franka/panda_joint.py`.

### Rebuild the dataset

On a machine with the LW-BenchHub fork set up and the raw episodes from its
`retarget_piper_to_ur5e.py --robot panda`:

```bash
export HF_LEROBOT_HOME=/path/with/space/lerobot
uv run examples/ur5e_lwlibero/convert_lw_hdf5_to_lerobot.py --raw-dir /path/to/panda_raw --robot panda
uv run examples/ur5e_lwlibero/visualize_dataset.py --repo-id courtneymcbeth/panda_lwlibero --out-dir /path/to/viz
```

## Evaluate

As for the UR5e: serve a checkpoint from this repo and run the evaluation in the `panda` branch of the
LW-BenchHub fork ("Evaluating a policy" in `lw_benchhub/scripts/retarget/README.md`; its episode list
records the robot, so the runner needs no extra flag).

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_panda_lwlibero_lora --policy.dir=checkpoints/pi05_panda_lwlibero_lora/panda_run1/<step>
```

The server expects `base_rgb` and `wrist_rgb` (224x224 uint8), `joints` (7), `gripper` (1) and `prompt`,
and returns absolute joint targets plus gripper commands, 8 numbers per step
(`src/openpi/policies/ur5e_policy.py`, which serves both robots). The closed-loop evaluation has not
been run for the Panda yet.
