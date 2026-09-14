"""A caricature of the Drosophila visual front end.

The real animal samples the world with ~750 ommatidia per eye arranged on a
hexagonal lattice, with an interommatidial angle of ~5 deg and a comparable
acceptance angle.  That is roughly 1/100th of the angular resolution of a
human fovea: a fly looking at a phone screen does *not* see a face, it sees a
coarse blob of colour and contrast.

We keep the parts of the anatomy that actually matter for downstream learning:

  * hexagonal ommatidial lattice with Gaussian acceptance functions
  * R1-6 (broadband, luminance) and R7/R8 pale / yellow spectral channels
  * lamina L1 / L2 ON and OFF half-wave rectified contrast
  * medulla-style orientation contrast along the three principal hex axes
  * blue/green spectral opponency (Dm9-mediated in the real medulla)
  * lobula-style pooling into a small number of "visual projection neuron"
    (VPN) channels, which is what the mushroom body actually receives

Everything here is a static-image approximation: no motion channels (T4/T5),
no looming (LPLC2), because a dating profile photo does not move.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# "odd-r" offset coordinates: every odd row is shifted half a step to the
# right, which is how a hexagonal lattice is stored in a rectangular array.
# Neighbour offsets therefore depend on the parity of the row.  Directions are
# listed in angular order -- E, SE, SW, W, NW, NE -- so that entries i and
# i + 3 are always opposite, which is what makes HEX_AXES below valid.
OFFSET_NEIGHBOURS = {
    0: ((1, 0), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)),   # even rows
    1: ((1, 0), (1, 1), (0, 1), (-1, 0), (0, -1), (1, -1)),     # odd rows
}
HEX_AXES = ((0, 3), (1, 4), (2, 5))  # opposite-neighbour pairs => 3 axes

# Channel names in the order produced by FlyEye.look().
CHANNELS = (
    "lum",       # R1-6, broadband luminance
    "pale",      # R7p/R8p, short wavelength
    "yellow",    # R7y/R8y, long wavelength
    "on",        # L1, ON contrast
    "off",       # L2, OFF contrast
    "axis_h",    # medulla orientation contrast, horizontal hex axis
    "axis_p",    # medulla orientation contrast, "p" hex axis
    "axis_q",    # medulla orientation contrast, "q" hex axis
    "opponent",  # blue/green spectral opponency
)


@dataclass
class EyeParams:
    """Optics of one eye.

    Defaults give 756 ommatidia, close to the ~750-800 of a real *Drosophila
    melanogaster* eye.
    """

    cols: int = 28
    rows: int = 27
    #: Acceptance function width as a fraction of the interommatidial spacing.
    #: Real Drosophila has delta_rho ~= delta_phi, i.e. neighbouring
    #: acceptance functions overlap heavily.
    acceptance: float = 0.55
    #: Resolution of the internal image grid the lattice samples from.
    grid: int = 72
    #: Lobula pooling: pools_per_side ** 2 retinotopic patches per channel.
    pools_per_side: int = 4
    #: Photoreceptor dark current: sets where log compression starts.
    dark_current: float = 0.02
    #: Saturation point of the lobula gain control.
    pool_gain: float = 1.5


@dataclass
class Lattice:
    axial: np.ndarray      # (n, 2) int axial coordinates
    xy: np.ndarray         # (n, 2) float positions in [0, 1]
    neighbours: np.ndarray  # (n, 6) int indices, -1 where the lattice ends
    spacing: float         # interommatidial spacing in [0, 1] units


def build_lattice(cols: int, rows: int) -> Lattice:
    """Build a hexagonal ommatidial lattice with axial coordinates."""
    axial = []
    for r in range(rows):
        for q in range(cols):
            axial.append((q, r))
    axial_arr = np.asarray(axial, dtype=np.int32)

    # "odd-r" offset layout: every other row is shifted by half a spacing.
    q = axial_arr[:, 0].astype(float)
    r = axial_arr[:, 1].astype(float)
    x = q + 0.5 * (r % 2.0)
    y = r * (np.sqrt(3.0) / 2.0)

    # Normalise into the unit square, keeping the aspect ratio of the lattice.
    span = max(x.max() - x.min(), y.max() - y.min())
    xy = np.stack([(x - x.min()) / span, (y - y.min()) / span], axis=1)
    # Centre whichever axis is shorter.
    xy[:, 0] += (1.0 - (xy[:, 0].max())) / 2.0
    xy[:, 1] += (1.0 - (xy[:, 1].max())) / 2.0
    spacing = 1.0 / span

    index = {(int(a), int(b)): i for i, (a, b) in enumerate(axial_arr)}
    neighbours = np.full((len(axial_arr), 6), -1, dtype=np.int32)
    for i, (qq, rr) in enumerate(axial_arr):
        for k, (dq, dr) in enumerate(OFFSET_NEIGHBOURS[int(rr) % 2]):
            neighbours[i, k] = index.get((int(qq) + dq, int(rr) + dr), -1)
    return Lattice(axial=axial_arr, xy=xy, neighbours=neighbours, spacing=spacing)


def _sampling_matrix(lattice: Lattice, grid: int, acceptance: float) -> np.ndarray:
    """Gaussian acceptance function of every ommatidium over an image grid.

    Returns an (n_ommatidia, grid * grid) row-normalised matrix, so sampling an
    image is a single matrix-vector product.
    """
    axis = (np.arange(grid) + 0.5) / grid
    gx, gy = np.meshgrid(axis, axis, indexing="xy")
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)

    sigma = acceptance * lattice.spacing
    d2 = (
        (lattice.xy[:, None, 0] - pix[None, :, 0]) ** 2
        + (lattice.xy[:, None, 1] - pix[None, :, 1]) ** 2
    )
    w = np.exp(-d2 / (2.0 * sigma ** 2))
    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32)


def _neighbour_mean(values: np.ndarray, neighbours: np.ndarray) -> np.ndarray:
    """Mean over existing hex neighbours of each ommatidium."""
    valid = neighbours >= 0
    idx = np.where(valid, neighbours, 0)
    gathered = values[idx] * valid
    count = valid.sum(axis=1)
    return gathered.sum(axis=1) / np.maximum(count, 1)


class FlyEye:
    """One compound eye plus the optic lobe up to the lobula."""

    def __init__(self, params: EyeParams | None = None):
        self.params = params or EyeParams()
        self.lattice = build_lattice(self.params.cols, self.params.rows)
        self._matrix = _sampling_matrix(
            self.lattice, self.params.grid, self.params.acceptance
        )
        self._pool_index, self.n_pools = self._build_pools()

    # -- geometry ---------------------------------------------------------
    @property
    def n_ommatidia(self) -> int:
        return len(self.lattice.axial)

    @property
    def n_vpn(self) -> int:
        """Number of visual projection channels handed to the mushroom body."""
        return self.n_pools * len(CHANNELS)

    def _build_pools(self):
        """Assign each ommatidium to one retinotopic lobula pool."""
        p = self.params.pools_per_side
        xy = self.lattice.xy
        col = np.clip((xy[:, 0] * p).astype(int), 0, p - 1)
        row = np.clip((xy[:, 1] * p).astype(int), 0, p - 1)
        return row * p + col, p * p

    # -- transduction -----------------------------------------------------
    def sample(self, image: np.ndarray) -> np.ndarray:
        """Sample a linear-light RGB image onto the ommatidial lattice.

        Parameters
        ----------
        image
            ``(grid, grid, 3)`` float array in [0, 1], linear light.

        Returns
        -------
        ``(n_ommatidia, 3)`` array of per-ommatidium RGB.
        """
        g = self.params.grid
        if image.shape[:2] != (g, g):
            raise ValueError(f"expected a {g}x{g} image, got {image.shape[:2]}")
        flat = image.reshape(g * g, 3).astype(np.float32)
        return self._matrix @ flat

    def look(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Full optic-lobe response: one value per channel per ommatidium.

        Two properties of real fly vision shape the maths here.  First,
        photoreceptors respond to the logarithm of intensity, so a doubling of
        light is a fixed increment rather than a doubling of response.  Second,
        the lamina reports *Weber* contrast -- the local difference divided by
        the local mean -- which makes the contrast channels invariant to how
        brightly the photo happened to be exposed while preserving how
        contrasty the scene actually is.
        """
        rgb = self.sample(image)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

        # R1-6 are broadband with a green-dominated spectral sensitivity, and
        # respond compressively to intensity.
        intensity = 0.25 * r + 0.60 * g + 0.15 * b
        lum = np.log1p(intensity / self.params.dark_current)
        floor = self.params.dark_current + intensity.mean() * 0.05

        # R7/R8 chromaticity: colour independent of overall exposure.
        total = intensity + 1e-6
        pale = b / total       # Rh5 / Rh3, short wavelength
        yellow = g / total     # Rh6 / Rh4, long wavelength

        # Lamina: L1 and L2 carry the ON and OFF half-waves of Weber contrast.
        surround = _neighbour_mean(intensity, self.lattice.neighbours)
        weber = (intensity - surround) / (surround + floor)
        on = np.maximum(weber, 0.0)
        off = np.maximum(-weber, 0.0)

        # Medulla: contrast along each of the three principal hex axes.
        axes = []
        for a, b_ in HEX_AXES:
            na, nb = self.lattice.neighbours[:, a], self.lattice.neighbours[:, b_]
            va = np.where(na >= 0, intensity[np.where(na >= 0, na, 0)], intensity)
            vb = np.where(nb >= 0, intensity[np.where(nb >= 0, nb, 0)], intensity)
            axes.append(np.abs(va - vb) / (0.5 * (va + vb) + floor))

        # Spectral opponency, as computed across R7/R8 in the medulla.
        opponent = (b - g) / (b + g + 1e-6)

        return {
            "lum": lum,
            "pale": pale,
            "yellow": yellow,
            "on": on,
            "off": off,
            "axis_h": axes[0],
            "axis_p": axes[1],
            "axis_q": axes[2],
            "opponent": opponent,
        }

    def vpn(self, image: np.ndarray) -> np.ndarray:
        """Lobula output: channels pooled over retinotopic patches.

        This is the vector that reaches the mushroom body calyx, standing in
        for the ME-MB / VPN-MB neurons that feed the visual (gamma-d) Kenyon
        cells in the real animal.
        """
        resp = self.look(image)
        out = np.zeros(self.n_vpn, dtype=np.float32)
        pools = self._pool_index
        for c, name in enumerate(CHANNELS):
            vals = resp[name]
            sums = np.bincount(pools, weights=vals, minlength=self.n_pools)
            counts = np.bincount(pools, minlength=self.n_pools)
            out[c * self.n_pools : (c + 1) * self.n_pools] = sums / np.maximum(counts, 1)

        # Compressive gain control by the wide-field medulla interneurons.
        # Unlike a divisive normalisation by the image mean, this keeps
        # differences *between* photos, which is the whole signal we want to
        # learn from, while stopping any one channel from dominating.
        return np.tanh(out / self.params.pool_gain).astype(np.float32)

    # -- inspection -------------------------------------------------------
    def ascii_view(self, image: np.ndarray, ramp: str = " .:-=+*#%@") -> str:
        """Render what the eye sees, on the hex lattice, as text."""
        lum = self.look(image)["lum"]
        lo, hi = float(lum.min()), float(lum.max())
        norm = (lum - lo) / (hi - lo + 1e-9)
        chars = np.asarray(list(ramp))[
            np.clip((norm * (len(ramp) - 1)).round().astype(int), 0, len(ramp) - 1)
        ]
        lines = []
        for r in range(self.params.rows):
            mask = self.lattice.axial[:, 1] == r
            row = "".join(f"{c} " for c in chars[mask])
            lines.append((" " if r % 2 else "") + row.rstrip())
        return "\n".join(lines)
