"""Builds a visual report of a LW-BenchHub LIBERO dataset, UR5e or Franka Panda (a LeRobot v2.1 dataset).

    uv run examples/ur5e_lwlibero/visualize_dataset.py --out-dir /hdd/vla_data/viz
    uv run examples/ur5e_lwlibero/visualize_dataset.py --out-dir /hdd/vla_data/viz_panda \
        --repo-id courtneymcbeth/panda_lwlibero

Writes to `--out-dir`:

    report.html   overview: size of the dataset, episodes per task, episode lengths, distributions of the
                  robot state and of the actions as the model is trained on them, example trajectories,
                  and a strip of camera frames for one episode of every task
    videos/       side-by-side (third-person | wrist) videos of sampled episodes of every task

The report is a single self-contained HTML file; open it in a browser. To scrub through an individual
episode interactively, use LeRobot's viewer:

    uv run python -m lerobot.scripts.visualize_dataset --repo-id <repo_id> --episode-index 0
"""

import base64
import html
import io
import json
import pathlib

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.training.config import UR5E_LWLIBERO_REPO_ID

# Chart colours (validated colour-blind-safe order) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Per `robot_type` of the dataset: display name, gripper, arm joint names, and the embodiment of the pi0.5
# base model whose value ranges the charts are compared with.
ROBOTS = {
    "ur5e": (
        "UR5e",
        "Robotiq 2F-85 gripper",
        ["shoulder pan", "shoulder lift", "elbow", "wrist 1", "wrist 2", "wrist 3"],
        "ur5e",
    ),
    "panda": ("Franka Panda", "Franka Hand", [f"joint {i}" for i in range(1, 8)], "franka"),
}
ACTION_HORIZON = 20

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    }
)


def fig_to_img_tag(fig, alt: str) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def array_to_img_tag(image: np.ndarray, alt: str) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    assert ok
    return f'<img alt="{html.escape(alt)}" src="data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}">'


def decode_image(cell: dict) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(cell["bytes"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_dataset(root: pathlib.Path):
    """Returns (info, tasks, episodes, low-dimensional arrays for all frames, parquet path per episode)."""
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        entry = json.loads(line)
        tasks[entry["task_index"]] = entry["task"]
    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    files = {int(p.stem.split("_")[-1]): p for p in sorted(root.glob("data/chunk-*/episode_*.parquet"))}
    cols = ["joints", "gripper", "actions", "episode_index", "frame_index", "task_index"]
    parts = [pq.read_table(files[i], columns=cols).to_pandas() for i in sorted(files)]
    arrays = {
        "joints": np.concatenate([np.stack(p["joints"].to_numpy()) for p in parts]).astype(np.float64),
        "gripper": np.concatenate([p["gripper"].to_numpy() for p in parts]).astype(np.float64).reshape(-1),
        "actions": np.concatenate([np.stack(p["actions"].to_numpy()) for p in parts]).astype(np.float64),
        "episode_index": np.concatenate([p["episode_index"].to_numpy() for p in parts]),
        "task_index": np.concatenate([p["task_index"].to_numpy() for p in parts]),
    }
    return info, tasks, episodes, arrays, files


def training_actions(arrays: dict, horizon: int) -> np.ndarray:
    """Actions as the model sees them: for every frame the next `horizon` actions, with the joint
    targets expressed relative to the joint positions at that frame (the gripper stays absolute).
    Past the end of an episode the last action repeats, as in the data loader."""
    out = []
    for ep in np.unique(arrays["episode_index"]):
        idx = np.flatnonzero(arrays["episode_index"] == ep)
        act, state = arrays["actions"][idx], arrays["joints"][idx]
        steps = np.minimum(np.arange(len(idx))[:, None] + np.arange(horizon)[None, :], len(idx) - 1)
        chunk = act[steps].copy()  # (T, horizon, joints + 1)
        chunk[..., :-1] -= state[:, None, :]
        out.append(chunk.reshape(-1, act.shape[-1]))
    return np.concatenate(out)


def pretrained_stats(asset_id: str) -> dict | None:
    """Norm stats of an embodiment (`ur5e`, `franka`) in the pi0.5 base checkpoint, if they can be downloaded."""
    try:
        import openpi.shared.download as download
        import openpi.shared.normalize as normalize

        return normalize.load(download.maybe_download(f"gs://openpi-assets/checkpoints/pi05_base/assets/{asset_id}"))
    except Exception as e:
        print(f"Could not load the pretrained {asset_id} norm stats ({e}); skipping that comparison.")
        return None


def wrap_label(text: str, width: int = 58) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def chart_episodes_per_task(names: list[str], counts: list[int], source_per_task: int):
    order = np.argsort(names)[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.27 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, [counts[i] for i in order], height=0.62, color=SERIES[0])
    ax.axvline(source_per_task, color=INK_2, linewidth=1)
    ax.text(
        source_per_task,
        len(names) - 0.3,
        f"{source_per_task} source demos",
        color=INK_2,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    for yi, i in zip(y, order, strict=True):
        ax.text(counts[i] + 0.6, yi, str(counts[i]), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [wrap_label(names[i]) for i in order])
    ax.set_xlim(0, max(source_per_task, *counts) * 1.08)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("episodes")
    ax.set_title("Episodes per task")
    return fig


def chart_lengths(seconds: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.hist(seconds, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
    med = float(np.median(seconds))
    ax.axvline(med, color=INK_2, linewidth=1)
    ax.text(med, ax.get_ylim()[1], f" median {med:.1f} s", color=INK_2, fontsize=8, va="top")
    ax.set_xlabel("episode length (s)")
    ax.set_ylabel("episodes")
    ax.set_title("Episode length")
    return fig


def chart_distributions(
    values: np.ndarray,
    joint_names: list[str],
    title: str,
    unit: str,
    pretrained: dict | None = None,
    pretrained_name: str = "",
):
    """Small multiples: one histogram per dimension with the 1st and 99th percentile marked."""
    names = [*joint_names, "gripper"]
    # One panel per dimension and one for the legend: 2 x 4 for the UR5e, 3 x 3 for the Panda.
    rows, cols = (2, 4) if len(names) < 8 else (3, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 2.3 * rows))
    for d, ax in enumerate(axes.flat[: len(names)]):
        v = values[:, d]
        q01, q99 = np.percentile(v, [1, 99])
        lo, hi = np.percentile(v, [0.1, 99.9])
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.hist(v, bins=60, range=(lo - pad, hi + pad), color=SERIES[0], edgecolor="none", alpha=0.9)
        for q in (q01, q99):
            ax.axvline(q, color=SERIES[0], linewidth=1.2)
        if pretrained is not None:
            for q in (pretrained.q01[d], pretrained.q99[d]):
                ax.axvline(q, color=SERIES[1], linewidth=1.2)
        ax.set_title(names[d], fontweight="normal", fontsize=9)
        ax.set_yticks([])
        ax.grid(visible=False)
        ax.set_xlabel("closure (0 open, 1 closed)" if d == len(names) - 1 else unit)
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    ax = axes.flat[len(names)]
    handles = [plt.Line2D([], [], color=SERIES[0], linewidth=1.2, label="this dataset: 1st and 99th percentile")]
    if pretrained is not None:
        handles.append(
            plt.Line2D(
                [],
                [],
                color=SERIES[1],
                linewidth=1.2,
                label=f"pi0.5 base {pretrained_name} data: 1st and 99th percentile",
            )
        )
    ax.legend(handles=handles, loc="center left", handlelength=1.4)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def chart_episode_trace(
    joints: np.ndarray, gripper: np.ndarray, actions: np.ndarray, joint_names: list[str], fps: float, title: str
):
    t = np.arange(len(joints)) / fps
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 4.2), height_ratios=[3, 1], sharex=True)
    for d, name in enumerate(joint_names):
        a1.plot(t, actions[:, d], color=SERIES[d], linewidth=1.0, alpha=0.45)
        a1.plot(t, joints[:, d], color=SERIES[d], linewidth=1.8, label=name)
    a1.set_ylabel("joint angle (rad)")
    a1.legend(ncols=len(joint_names), loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.2)
    a1.set_title(title, pad=26)
    a2.plot(t, actions[:, -1], color=INK_2, linewidth=1.0, alpha=0.6, label="command")
    a2.plot(t, gripper, color=SERIES[0], linewidth=1.8, label="measured")
    a2.set_ylabel("gripper closure")
    a2.set_xlabel("time (s)")
    a2.set_ylim(-0.08, 1.08)
    a2.legend(ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.2)
    fig.align_ylabels()
    fig.tight_layout()
    return fig


def read_frames(root: pathlib.Path, info: dict, files: dict, episode: int, key: str) -> np.ndarray:
    """All frames (T, H, W, 3) of one camera of one episode, for image- and video-based datasets."""
    if info["features"][key]["dtype"] == "video":
        import av

        path = root / info["video_path"].format(
            episode_chunk=episode // info["chunks_size"], video_key=key, episode_index=episode
        )
        with av.open(str(path)) as container:
            return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    cells = pq.read_table(files[episode], columns=[key]).column(key).to_pylist()
    return np.stack([decode_image(c) for c in cells])


def frame_strip(image: np.ndarray, wrist: np.ndarray, n: int) -> np.ndarray:
    """Third-person frames on top, wrist frames below, `n` evenly spaced moments of one episode."""
    idx = np.linspace(0, len(image) - 1, n).round().astype(int)
    return np.concatenate([np.concatenate(list(image[idx]), axis=1), np.concatenate(list(wrist[idx]), axis=1)], axis=0)


def write_video(image: np.ndarray, wrist: np.ndarray, path: pathlib.Path, fps: float):
    """Side-by-side (third-person | wrist) H.264 video at twice the recorded resolution."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=round(fps))
        stream.width, stream.height, stream.pix_fmt = 2 * 2 * image.shape[2], 2 * image.shape[1], "yuv420p"
        for a, b in zip(image, wrist, strict=True):
            frame = cv2.resize(np.concatenate([a, b], axis=1), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


CSS = f"""
body {{ background: {SURFACE}; color: {INK}; font: 15px/1.5 system-ui, sans-serif; margin: 0; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 40px 0 6px; }}
p {{ color: {INK_2}; max-width: 78ch; margin: 4px 0 12px; }}
.tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 18px 0; }}
.tile {{ border: 1px solid {GRID}; border-radius: 8px; padding: 10px 16px; min-width: 120px; }}
.tile b {{ display: block; font-size: 24px; }} .tile span {{ color: {INK_2}; font-size: 13px; }}
img {{ max-width: 100%; height: auto; }}
.task {{ margin: 18px 0 26px; }} .task h3 {{ font-size: 14px; margin: 0 0 2px; }}
.task small {{ color: {INK_2}; }}
table {{ border-collapse: collapse; font-size: 13px; }} td, th {{ padding: 3px 12px 3px 0; text-align: left; }}
th {{ color: {INK_2}; font-weight: 600; border-bottom: 1px solid {GRID}; }} td.n {{ text-align: right; }}
details {{ margin: 8px 0; }} summary {{ cursor: pointer; color: {INK_2}; }}
"""


def main(
    out_dir: str,
    *,
    repo_id: str = UR5E_LWLIBERO_REPO_ID,
    source_demos_per_task: int = 50,
    videos_per_task: int = 2,
    frames_per_strip: int = 8,
    compare_pretrained: bool = True,
):
    """Writes the report for the dataset at `$HF_LEROBOT_HOME/<repo_id>`.

    Args:
        out_dir: Where to write `report.html` and `videos/`.
        repo_id: Dataset name below `$HF_LEROBOT_HOME`.
        source_demos_per_task: Number of source demonstrations per task (reference line in the first chart).
        videos_per_task: Episodes per task to export as videos (0 to skip).
        frames_per_strip: Camera frames shown per task in the report.
        compare_pretrained: Mark the value ranges of the pi0.5 base model's data for this kind of robot (its
            `ur5e` embodiment for the UR5e, `franka` for the Panda) in the charts.
    """
    root = HF_LEROBOT_HOME / repo_id
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info, tasks, episodes, arrays, files = load_dataset(root)
    fps = float(info["fps"])
    rng = np.random.default_rng(0)
    robot, gripper_name, joint_names, pretrained_id = ROBOTS[info["robot_type"]]

    ep_task = {e["episode_index"]: e["tasks"][0] for e in episodes}
    ep_len = {e["episode_index"]: e["length"] for e in episodes}
    task_names = sorted(set(ep_task.values()))
    by_task = {name: [i for i, t in ep_task.items() if t == name] for name in task_names}
    counts = [len(by_task[name]) for name in task_names]
    seconds = np.array([ep_len[i] / fps for i in sorted(ep_len)])

    state = np.concatenate([arrays["joints"], arrays["gripper"][:, None]], axis=1)
    deltas = training_actions(arrays, ACTION_HORIZON)
    pretrained = pretrained_stats(pretrained_id) if compare_pretrained else None

    parts = [
        f"<h1>{robot} demonstrations on LW-BenchHub LIBERO tasks</h1><p>Dataset <code>{html.escape(repo_id)}</code>, "
        f"recorded at {fps:g} Hz. Every episode is a Double-Piper demonstration replayed by a {robot} with a "
        f"{gripper_name} in simulation and kept only if the task was solved.</p>",
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><b>{v}</b><span>{k}</span></div>'
            for k, v in [
                ("episodes", f"{len(episodes):,}"),
                ("tasks", len(task_names)),
                ("frames", f"{len(state):,}"),
                ("hours of robot time", f"{len(state) / fps / 3600:.1f}"),
                ("median episode", f"{np.median(seconds):.1f} s"),
            ]
        )
        + "</div>",
        f"<h2>Coverage</h2><p>Successful {robot} episodes per task. Each task has "
        f"{source_demos_per_task} source demonstrations; the shortfall is demonstrations the {robot} could not "
        "reproduce.</p>",
        fig_to_img_tag(chart_episodes_per_task(task_names, counts, source_demos_per_task), "Episodes per task"),
        "<details><summary>Table: episodes per task</summary><table><tr><th>task</th><th>episodes</th>"
        "<th>median length (s)</th></tr>"
        + "".join(
            f"<tr><td>{html.escape(n)}</td><td class='n'>{c}</td>"
            f"<td class='n'>{np.median([ep_len[i] / fps for i in by_task[n]]):.1f}</td></tr>"
            for n, c in zip(task_names, counts, strict=True)
        )
        + "</table></details>",
        fig_to_img_tag(chart_lengths(seconds), "Histogram of episode lengths"),
        "<h2>Robot state</h2><p>Measured arm joint angles and gripper closure over all frames. These are the "
        "model's proprioceptive input.</p>",
        fig_to_img_tag(
            chart_distributions(
                state, joint_names, "State", "rad", pretrained and pretrained.get("state"), pretrained_id
            ),
            "State distributions",
        ),
        f"<h2>Actions as trained</h2><p>The model predicts {ACTION_HORIZON} actions ({ACTION_HORIZON / fps:g} s) at a "
        "time. Joint targets are expressed relative to the joint angles at the start of the chunk; the gripper "
        "command stays absolute. The histograms cover every action of every chunk.</p>",
        fig_to_img_tag(
            chart_distributions(
                deltas,
                joint_names,
                "Actions (relative joint targets)",
                "rad",
                pretrained and pretrained.get("actions"),
                pretrained_id,
            ),
            "Action distributions",
        ),
        "<h2>Example trajectories</h2><p>Thick lines are measured joint angles, thin lines the commanded "
        "targets (the actions). The measured angles follow the targets with a short lag.</p>",
    ]
    for ep in rng.choice(sorted(ep_len), size=min(3, len(ep_len)), replace=False):
        idx = np.flatnonzero(arrays["episode_index"] == ep)
        title = f"Episode {ep}: {ep_task[int(ep)]}"
        fig = chart_episode_trace(
            arrays["joints"][idx], arrays["gripper"][idx], arrays["actions"][idx], joint_names, fps, title
        )
        parts.append(fig_to_img_tag(fig, title))

    parts.append(
        "<h2>Camera views</h2><p>One episode per task, evenly spaced from first to last frame. Top row: "
        "third-person camera. Bottom row: wrist camera.</p>"
    )
    for name in task_names:
        eps = by_task[name]
        slug = "".join(c if c.isalnum() else "_" for c in name.lower())[:60].strip("_")
        shown = False
        picks = [
            eps[0],
            *[int(e) for e in rng.choice(eps[1:], size=min(videos_per_task - 1, len(eps) - 1), replace=False)],
        ]
        for ep in picks[: max(1, videos_per_task)]:
            image = read_frames(root, info, files, ep, "image")
            wrist = read_frames(root, info, files, ep, "wrist_image")
            if not shown:
                strip = array_to_img_tag(frame_strip(image, wrist, frames_per_strip), f"Camera frames: {name}")
                parts.append(
                    f'<div class="task"><h3>{html.escape(name)}</h3><small>episode {ep}, {len(eps)} episodes in '
                    f"total</small><br>{strip}</div>"
                )
                shown = True
            if videos_per_task > 0:
                write_video(image, wrist, out / "videos" / f"{slug}__ep{ep:05d}.mp4", fps)

    (out / "report.html").write_text(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
        f"initial-scale=1'><title>{robot} LIBERO dataset</title><style>{CSS}</style></head><body><main>"
        + "\n".join(parts)
        + "</main></body></html>"
    )
    print(f"Wrote {out / 'report.html'}" + (f" and videos to {out / 'videos'}" if videos_per_task else ""))


if __name__ == "__main__":
    tyro.cli(main)
