"""
Stitch per-model planning gifs into ONE side-by-side comparison GIF.

After running `eval_only` planning on several checkpoints (e.g. the 1/5/12-epoch
models), each writes a gif per room (episode) showing its agent trying to reach
the goal. Because the env's room generator is deterministic, room `e` is the SAME
maze for every model — so we can lay them side by side and watch which model
solves it.

Per room we show frames until the FIRST model solves (or the step budget runs
out), then switch to the next room — exactly the "race" framing.

Usage (paths are the per-model checkpoint folders that contain plan_eval/):
    python -m examples.ac_video_jepa.compare_gifs \
        --models "evidence/1epoch_X/model,evidence/5epoch_Y/model,evidence/12epoch_Z/model" \
        --labels "1 epoch,5 epochs,12 epochs" \
        --out evidence/compare/solve_comparison.gif \
        --num_rooms 10
"""

from glob import glob
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

UP = 4          # upscale factor per panel (65x65 -> 260x260)
HOLD = 6        # extra frames held on the last frame of each room
FPS = 6
PAD = 4         # black gutter between panels
LABEL_H = 26    # header strip height (px, after upscale)


def _find_ep_gif(model_dir, ep):
    """Glob the per-episode gif for one model. Returns (path, solved:bool) or (None, False)."""
    hits = glob(f"{model_dir}/plan_eval/*/ep_{ep}/agent_steps_*.gif")
    if not hits:
        return None, False
    p = sorted(hits)[-1]  # most recent eval folder if several
    return p, ("succ" in Path(p).name)


def _load_rgb(path):
    """Read a gif -> [n, H, W, 3] uint8."""
    arr = np.asarray(iio.imread(path, index=None))
    if arr.ndim == 3:  # single frame [H,W,3] or [n,H,W]
        arr = arr[None] if arr.shape[-1] in (3, 4) else arr[..., None].repeat(3, -1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:  # grayscale [n,H,W]
        arr = arr[..., None].repeat(3, -1)
    return arr.astype(np.uint8)


def _panel(frame, up=UP):
    """Upscale one [H,W,3] frame by nearest-neighbour."""
    return np.repeat(np.repeat(frame, up, 0), up, 1)


def _blank(h, w):
    img = np.full((h, w, 3), 60, np.uint8)
    return img


def compare_gifs(models, labels, out, num_rooms=10, max_steps=120):
    model_dirs = [m.strip() for m in models.split(",")]
    names = [l.strip() for l in labels.split(",")]
    assert len(model_dirs) == len(names), "models and labels count mismatch"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []
    room_report = []
    for ep in range(num_rooms):
        gifs, solved = [], []
        for md in model_dirs:
            p, ok = _find_ep_gif(md, ep)
            gifs.append(_load_rgb(p) if p else None)
            solved.append(ok)
        if all(g is None for g in gifs):
            continue  # this room wasn't evaluated by anyone

        # Common panel size (use the first available frame's H,W).
        ref = next(g for g in gifs if g is not None)
        H, W = ref.shape[1], ref.shape[2]
        lengths = [len(g) for g in gifs if g is not None]
        # Room length = first solver's solve time (= its gif length) else min full length,
        # capped. min() over models implements "switch when the first one solves".
        L = min(min(lengths), max_steps)

        # Who won this room (shortest SUCCESSFUL gif).
        winners = [(len(gifs[i]), names[i]) for i in range(len(gifs))
                   if gifs[i] is not None and solved[i]]
        win = min(winners)[1] if winners else "none"
        room_report.append((ep, win, [solved[i] for i in range(len(gifs))]))

        pw, ph = W * UP, H * UP
        for i in range(L):
            panels = []
            for k, g in enumerate(gifs):
                if g is None:
                    panel = _blank(ph, pw)
                    status = "N/A"
                else:
                    fr = g[min(i, len(g) - 1)]
                    panel = _panel(fr)
                    # green tick once this model has solved (its gif ended)
                    status = "SOLVED" if (solved[k] and i >= len(g) - 1) else f"step {i}"
                panels.append((panel, names[k], status, solved[k]))
            all_frames.append(_compose_row(panels, ep, win, ph, pw))
        # hold the final frame so the result is readable
        all_frames += [all_frames[-1]] * HOLD

    if not all_frames:
        raise SystemExit("No episode gifs found — did eval_only run for these models?")

    iio.imwrite(out, np.stack(all_frames), duration=int(1000 / FPS), loop=0)
    print(f"[compare_gifs] wrote {out}  ({len(all_frames)} frames, {num_rooms} rooms)")
    print("Room winners:")
    for ep, win, sv in room_report:
        print(f"  room {ep}: winner={win}  solved={dict(zip(names, sv))}")


def _compose_row(panels, ep, win, ph, pw):
    """Lay panels horizontally with a header strip + per-panel labels."""
    n = len(panels)
    total_w = n * pw + (n - 1) * PAD
    canvas = Image.new("RGB", (total_w, ph + LABEL_H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), f"Room {ep}   (first to solve: {win})", fill=(255, 255, 255))
    x = 0
    for panel, name, status, solved in panels:
        canvas.paste(Image.fromarray(panel), (x, LABEL_H))
        col = (90, 220, 90) if (solved and status == "SOLVED") else (220, 220, 220)
        draw.text((x + 6, LABEL_H + 4), f"{name}", fill=(255, 255, 0))
        draw.text((x + 6, LABEL_H + 16), status, fill=col)
        x += pw + PAD
    return np.asarray(canvas)


if __name__ == "__main__":
    import fire

    fire.Fire(compare_gifs)
