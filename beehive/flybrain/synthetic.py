"""Synthetic "profiles" so you can watch the fly learn without any photos.

Each stimulus is built from four latent attributes -- warmth, contrast,
busyness and brightness -- and a hidden taste rule decides the ground-truth
swipe.  Ten percent of labels are flipped on purpose: real people are not
consistent either, and a model that cannot cope with that is useless here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Hidden preference used by the demo: warm tones and contrast good, visual
#: clutter bad.  The fly is never told this.
TASTE = np.array([1.2, 0.9, -1.0, 0.15])
LABEL_NOISE = 0.10


@dataclass
class Profile:
    image: np.ndarray
    attributes: np.ndarray  # warmth, contrast, busyness, brightness
    swipe: str              # ground truth, with noise applied


def render(attrs: np.ndarray, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a stimulus whose latent attributes are actually visible.

    Every attribute is expressed as something a 5-degree-resolution eye can
    pick up: a hue shift, a luminance step, a count of blobs.  Fine detail is
    pointless -- the ommatidial lattice would average it away.
    """
    warmth, contrast, busyness, brightness = attrs
    yy, xx = np.mgrid[0:grid, 0:grid] / grid

    level = 0.15 + 0.55 * brightness
    warm = np.array([1.0, 0.62, 0.30], dtype=np.float32)
    cool = np.array([0.30, 0.58, 1.0], dtype=np.float32)
    hue = warmth * warm + (1.0 - warmth) * cool
    base = np.ones((grid, grid, 3), dtype=np.float32) * level * hue

    # Blob count carries busyness.  Blobs go on a jittered grid so that they
    # do not merge into one another: if they merged, edge density would stop
    # tracking the count and the attribute would become invisible to the eye.
    n_blobs = int(1 + round(busyness * 11))
    step = 0.12 + 0.75 * contrast
    side = int(np.ceil(np.sqrt(n_blobs)))
    cell = 1.0 / side
    slots = rng.permutation(side * side)[:n_blobs]
    for i, slot in enumerate(slots):
        cx = (slot % side + 0.5) * cell + rng.uniform(-0.15, 0.15) * cell
        cy = (slot // side + 0.5) * cell + rng.uniform(-0.15, 0.15) * cell
        r = cell * rng.uniform(0.30, 0.42)
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        sign = 1.0 if i % 2 == 0 else -1.0
        tint = hue * (1.0 + 0.3 * rng.uniform(-1.0, 1.0, size=3).astype(np.float32))
        base[mask] = np.clip(level * (1.0 + sign * step), 0.02, 1.0) * tint

    base += rng.normal(0.0, 0.01, size=base.shape).astype(np.float32)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def make_profiles(n: int, grid: int, seed: int = 0) -> list[Profile]:
    rng = np.random.default_rng(seed)
    attrs = rng.uniform(0.0, 1.0, size=(n, 4)).astype(np.float32)
    scores = attrs @ TASTE
    cutoff = float(np.median(scores))

    profiles = []
    for i in range(n):
        liked = scores[i] > cutoff
        if rng.random() < LABEL_NOISE:
            liked = not liked
        profiles.append(
            Profile(
                image=render(attrs[i], grid, rng),
                attributes=attrs[i],
                swipe="right" if liked else "left",
            )
        )
    return profiles
