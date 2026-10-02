"""Screen recordings replayed as an agent's observation stream.

Frames are sampled at a fixed interval (an agent that takes a screenshot every N seconds),
downscaled so the long edge is at most 1280px, and fed to the same history policies as the
Mind2Web replay. Source: HumynLabs/flight-booking-screen-recording (CC-BY-4.0), people
booking flights on Expedia on desktop and phone. Videos are referred to by index only.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from bench.replay.mind2web import Step, Trajectory

REPO = "HumynLabs/flight-booking-screen-recording"


def download() -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".mp4")
    )
    return [Path(hf_hub_download(REPO, f, repo_type="dataset")) for f in files]


def sample_frames(
    path: Path, every_s: float = 2.0, long_edge: int = 1280, max_frames: int = 120
) -> list[Image.Image]:
    import av

    frames: list[Image.Image] = []
    next_t = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.time is None or frame.time < next_t:
                continue
            im = frame.to_image().convert("RGB")
            s = long_edge / max(im.size)
            if s < 1:
                im = im.resize(
                    (round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS
                )
            frames.append(im)
            next_t += every_s
            if len(frames) >= max_frames:
                break
    return frames


def load(every_s: float = 2.0, max_frames: int = 120) -> list[Trajectory]:
    trajs = []
    for i, path in enumerate(download(), 1):
        frames = sample_frames(path, every_s, max_frames=max_frames)
        if len(frames) >= 3:
            steps = [Step(k, "observe", f, None) for k, f in enumerate(frames)]
            trajs.append(Trajectory(f"video_{i}", "expedia", "book a flight", steps))
    return trajs
