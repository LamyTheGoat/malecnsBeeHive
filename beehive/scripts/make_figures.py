"""Render every figure in the BeeHive presentation from the model itself.

Nothing here is drawn by hand or mocked: each panel is produced by running the
same code path the CLI uses, so the presentation cannot drift away from what
the model actually does.

    python3 scripts/make_figures.py [output_dir]

Writes PNG panels plus figures.json (the numeric series for the charts).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import Fly  # noqa: E402
from flybrain.eye import CHANNELS  # noqa: E402
from flybrain.synthetic import LABEL_NOISE, TASTE, make_profiles  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
SUPERSAMPLE = 3
PANEL = 300

# Channel panels carry their own dark ground, the way a confocal channel is
# conventionally displayed, so they read identically in either page theme.
PANEL_BG = (18, 20, 26)

SEQUENTIAL = ["#12141a", "#173a63", "#1f5da3", "#3987e5", "#86b6ef", "#e3edfb"]
DIVERGING_NEG = ["#2e2f34", "#8a3a3c", "#e5605f"]
DIVERGING_POS = ["#2e2f34", "#2f5c9e", "#4e93ea"]


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def ramp(stops: list[str], t: np.ndarray) -> np.ndarray:
    """Piecewise-linear colour ramp; t in [0, 1]."""
    cols = np.stack([hex_to_rgb(s) for s in stops])
    pos = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - lo)[:, None]
    return cols[lo] * (1 - frac) + cols[lo + 1] * frac


def sequential(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return ramp(SEQUENTIAL, (values - lo) / (hi - lo + 1e-9))


def diverging(values: np.ndarray) -> np.ndarray:
    scale = float(np.abs(values).max()) + 1e-9
    t = values / scale
    neg = ramp(DIVERGING_NEG, np.abs(np.minimum(t, 0.0)))
    pos = ramp(DIVERGING_POS, np.maximum(t, 0.0))
    return np.where((t < 0)[:, None], neg, pos)


def hex_panel(eye, values: np.ndarray, colours: np.ndarray, path: Path) -> None:
    """Draw one value per ommatidium as a hexagon on the lattice."""
    size = PANEL * SUPERSAMPLE
    img = Image.new("RGB", (size, size), PANEL_BG)
    draw = ImageDraw.Draw(img)
    unit = eye.lattice.spacing * size
    radius = unit / math.sqrt(3) * 0.96
    angles = [math.radians(a) for a in (90, 150, 210, 270, 330, 30)]
    for (x, y), colour in zip(eye.lattice.xy, colours):
        cx, cy = x * size, y * size
        draw.polygon(
            [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles],
            fill=tuple(int(round(c)) for c in colour),
        )
    img.resize((PANEL, PANEL), Image.LANCZOS).save(path, optimize=True)


def save_stimulus(image: np.ndarray, path: Path, size: int = 300) -> None:
    """Save a linear-light stimulus as a display-gamma JPEG."""
    srgb = np.clip(image, 0.0, 1.0) ** (1 / 2.2)
    Image.fromarray((srgb * 255).astype(np.uint8)).resize(
        (size, size), Image.LANCZOS
    ).save(path, quality=88, optimize=True)


def grating(grid: int, cycles: float) -> np.ndarray:
    """A vertical sinusoidal grating, the standard acuity stimulus."""
    x = np.arange(grid) / grid
    wave = 0.5 + 0.45 * np.sin(2 * np.pi * cycles * x)
    return np.repeat(wave[None, :, None], grid, axis=0).repeat(3, axis=2).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fly = Fly()
    eye, grid = fly.eye, fly.eye.params.grid
    figures: dict = {
        "model": {
            "ommatidia": eye.n_ommatidia,
            "vpn": eye.n_vpn,
            "pools": eye.n_pools,
            "channels": list(CHANNELS),
            "kc": fly.mb.params.n_kc,
            "claws": fly.mb.params.claws,
            "sparsity": fly.mb.params.sparsity,
            "active_kc": int(round(fly.mb.params.sparsity * fly.mb.params.n_kc)),
            "label_noise": LABEL_NOISE,
            "taste": [float(t) for t in TASTE],
        }
    }

    # ---- stimuli -------------------------------------------------------
    train = make_profiles(400, grid, seed=1)
    test = make_profiles(150, grid, seed=901)
    # Identical attributes and labels, rendered large: make_profiles draws its
    # random numbers in a grid-independent order.
    train_big = make_profiles(400, 300, seed=1)

    scores = np.array([p.attributes @ TASTE for p in train])
    order = np.argsort(scores)
    picks = [int(order[i]) for i in (3, 40, 150, 250, 370, 396)]

    figures["profiles"] = []
    for rank, idx in enumerate(picks):
        profile, big = train[idx], train_big[idx]
        save_stimulus(big.image, OUT / f"stim_{rank}.jpg")
        hex_panel(eye, None, sequential(eye.look(profile.image)["lum"]),
                  OUT / f"retina_{rank}.png")
        percept = fly.percept_from_vpn(eye.vpn(profile.image))
        figures["profiles"].append({
            "stimulus": f"stim_{rank}.jpg",
            "retina": f"retina_{rank}.png",
            "attributes": [round(float(a), 2) for a in profile.attributes],
            "score": round(float(scores[idx]), 2),
            "swipe": profile.swipe,
            "active_kc": int((percept.kc > 0).sum()),
        })

    # ---- channel decomposition ----------------------------------------
    hero = train[picks[4]]
    response = eye.look(hero.image)
    signed = {"opponent"}
    figures["channels"] = []
    for name in CHANNELS:
        values = response[name]
        colours = diverging(values) if name in signed else sequential(values)
        hex_panel(eye, values, colours, OUT / f"chan_{name}.png")
        figures["channels"].append({
            "name": name,
            "file": f"chan_{name}.png",
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
            "signed": name in signed,
        })
    figures["hero"] = {"stimulus": "stim_4.jpg", "swipe": hero.swipe}

    # ---- acuity --------------------------------------------------------
    figures["acuity"] = []
    for cycles in (4, 8, 14, 28):
        stim = grating(grid, cycles)
        big = grating(300, cycles)
        save_stimulus(big, OUT / f"grating_{cycles}.jpg")
        seen = eye.look(stim)["lum"]
        hex_panel(eye, seen, sequential(seen), OUT / f"grating_seen_{cycles}.png")
        contrast = float(eye.look(stim)["on"].max())
        figures["acuity"].append({
            "cycles": cycles,
            "degrees_per_cycle": round(28 / cycles * 5.0, 1),
            "stimulus": f"grating_{cycles}.jpg",
            "seen": f"grating_seen_{cycles}.png",
            "contrast": round(contrast, 3),
        })

    # ---- projection vector --------------------------------------------
    vpn = eye.vpn(hero.image)
    figures["vpn"] = {
        "values": [round(float(v), 3) for v in vpn],
        "pools": eye.n_pools,
        "channels": list(CHANNELS),
    }

    # ---- Kenyon cell code ---------------------------------------------
    a_idx, b_idx = picks[4], picks[1]
    kc_a = fly.mb.kenyon_cells(eye.vpn(train[a_idx].image))
    kc_b = fly.mb.kenyon_cells(eye.vpn(train[b_idx].image))
    act_a, act_b = kc_a > 0, kc_b > 0
    raster = np.zeros((40, 50, 3), dtype=np.uint8)
    raster[:] = PANEL_BG
    both = act_a & act_b
    flat = raster.reshape(-1, 3)
    flat[np.where(act_a & ~both)[0]] = hex_to_rgb("#3987e5")
    flat[np.where(act_b & ~both)[0]] = hex_to_rgb("#e66767")
    flat[np.where(both)[0]] = hex_to_rgb("#e3edfb")
    Image.fromarray(raster).resize((50 * 12, 40 * 12), Image.NEAREST).save(
        OUT / "kc_raster.png", optimize=True
    )
    # One pair proves nothing, so measure the overlap over many pairs, and
    # compare it against how similar the inputs were to begin with.  This is
    # what says whether the expansion is actually decorrelating anything.
    sample = [eye.vpn(p.image) for p in train[:120]]
    codes = [fly.mb.kenyon_cells(v) > 0 for v in sample]
    units = [v / (np.linalg.norm(v) + 1e-9) for v in sample]
    overlaps, similarities = [], []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            overlaps.append((codes[i] & codes[j]).sum() / codes[i].sum())
            similarities.append(float(units[i] @ units[j]))
    overlaps_arr = np.asarray(overlaps)
    figures["kenyon"] = {
        "total": int(fly.mb.params.n_kc),
        "a_active": int(act_a.sum()),
        "b_active": int(act_b.sum()),
        "shared": int(both.sum()),
        "a_stimulus": "stim_4.jpg",
        "b_stimulus": "stim_1.jpg",
        "pairs": len(overlaps),
        "overlap_median": round(float(np.median(overlaps_arr)), 3),
        "overlap_p10": round(float(np.percentile(overlaps_arr, 10)), 3),
        "overlap_p90": round(float(np.percentile(overlaps_arr, 90)), 3),
        "input_similarity_median": round(float(np.median(similarities)), 3),
        "overlap_hist": np.histogram(overlaps_arr, bins=np.linspace(0, 1, 21))[0].tolist(),
    }

    # ---- learning ------------------------------------------------------
    train_vpn = [eye.vpn(p.image) for p in train]
    test_vpn = [eye.vpn(p.image) for p in test]
    truth = [p.swipe for p in test]

    def accuracy() -> float:
        return sum(fly.swipe_vpn(v).swipe == t for v, t in zip(test_vpn, truth)) / len(truth)

    def valences() -> dict[str, list[float]]:
        out: dict[str, list[float]] = {"right": [], "left": []}
        for v, t in zip(test_vpn, truth):
            out[t].append(round(float(fly.percept_from_vpn(v).valence), 4))
        return out

    figures["valence_before"] = valences()
    edges = np.linspace(0, len(train), 17).astype(int)
    curve = [accuracy()]
    seen_n = [0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        for i in range(lo, hi):
            fly.train_vpn(train_vpn[i], train[i].swipe)
        curve.append(accuracy())
        seen_n.append(int(hi))
    figures["valence_after"] = valences()
    figures["learning"] = {
        "seen": seen_n,
        "accuracy": [round(c, 4) for c in curve],
        "chance": 0.5,
        "noise_ceiling": round(1 - LABEL_NOISE, 3),
    }

    # Least-squares read-out straight off the projection channels: a yardstick
    # for how much of the hidden taste survives the optics at all.
    x = np.hstack([np.stack(train_vpn + test_vpn),
                   np.ones((len(train) + len(test), 1), dtype=np.float32)])
    y = np.array([1.0 if p.swipe == "right" else -1.0 for p in train + test])
    split = len(x) * 2 // 3
    w = np.linalg.solve(x[:split].T @ x[:split] + np.eye(x.shape[1]), x[:split].T @ y[:split])
    figures["learning"]["optics_ceiling"] = round(
        float(np.mean(np.sign(x[split:] @ w) == y[split:])), 4
    )

    # ---- synapses ------------------------------------------------------
    hist_edges = np.linspace(fly.mb.params.w_min, fly.mb.params.w0, 25)
    figures["weights"] = {
        "edges": [round(float(e), 3) for e in hist_edges],
        "approach": np.histogram(fly.mb.w_out[0], bins=hist_edges)[0].tolist(),
        "avoid": np.histogram(fly.mb.w_out[1], bins=hist_edges)[0].tolist(),
        "load": {k: round(v, 4) for k, v in fly.mb.memory_load().items()},
    }

    # ---- the turn ------------------------------------------------------
    figures["races"] = []
    for label, idx in (("right", a_idx), ("left", b_idx)):
        percept = fly.percept_from_vpn(eye.vpn(train[idx].image))
        runs = []
        for _ in range(6):
            decision = fly.steering.decide(percept.valence)
            runs.append([round(float(v), 4) for v in decision.trace])
        left, right = fly.steering.descending(percept.valence)
        figures["races"].append({
            "expected": label,
            "valence": round(float(percept.valence), 4),
            "dna02": [round(left, 3), round(right, 3)],
            "threshold": fly.steering.params.threshold,
            "runs": runs,
        })

    (OUT / "figures.json").write_text(json.dumps(figures, indent=1), encoding="utf-8")
    print(f"{len(list(OUT.glob('*.png'))) + len(list(OUT.glob('*.jpg')))} panel, "
          f"figures.json -> {OUT}")


if __name__ == "__main__":
    main()
