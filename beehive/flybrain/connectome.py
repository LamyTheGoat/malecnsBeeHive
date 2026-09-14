"""Where the numbers come from.

The model runs out of the box on published population statistics.  If you have
a neuprint token you can replace them with counts measured directly in the
male CNS reconstruction that this repository wraps: run
``beehive/scripts/export_mb_connectome.R`` and point ``--connectome`` at the
directory it writes.

Nothing here invents connectivity.  Either a number came from a paper (and says
so) or it came from a CSV you exported yourself.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .mushroom_body import MBON_CELL_TYPES, MBParams

#: Population statistics used when no local export is available.
PUBLISHED = {
    "n_kc": (2000, "~2000 Kenyon cells/hemisphere (hemibrain v1.2; Li et al. 2020)"),
    "claws": (6, "6-7 dendritic claws per KC (Caron et al. 2013)"),
    "sparsity": (0.08, "~5-10% of KCs active per stimulus, APL-enforced (Lin et al. 2014)"),
    "n_mbon": (2, "2 of ~35 MBON types/hemisphere modelled (Aso et al. 2014)"),
    "n_ommatidia": (756, "~750-800 ommatidia per eye (Ready et al. 1976)"),
}


@dataclass
class ConnectomeSource:
    name: str
    n_kc: int
    claws: int
    mbon_scale: dict[str, float]
    provenance: str

    def apply(self, params: MBParams) -> MBParams:
        return MBParams(
            n_input=params.n_input,
            n_kc=self.n_kc,
            claws=self.claws,
            sparsity=params.sparsity,
            eta=params.eta,
            recovery=params.recovery,
            w0=params.w0,
            w_min=params.w_min,
            seed=params.seed,
        )

    def describe(self) -> str:
        lines = [f"Konnektom kaynagi: {self.name}", f"  {self.provenance}"]
        lines.append(f"  Kenyon hucresi sayisi : {self.n_kc}")
        lines.append(f"  KC basina pençe (claw): {self.claws}")
        for k, v in self.mbon_scale.items():
            lines.append(f"  KC->{k} bagil agirlik : {v:.3f}")
        return "\n".join(lines)


def default_source() -> ConnectomeSource:
    return ConnectomeSource(
        name="yayinlanmis istatistikler (canli veri yok)",
        n_kc=PUBLISHED["n_kc"][0],
        claws=PUBLISHED["claws"][0],
        mbon_scale={name: 1.0 for name in MBON_CELL_TYPES},
        provenance="; ".join(src for _, src in PUBLISHED.values()),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_export(directory: str | Path) -> ConnectomeSource:
    """Load an export produced by ``scripts/export_mb_connectome.R``.

    Expects ``mb_neurons.csv`` (bodyid, type, mbclass) and ``mb_kc_mbon.csv``
    (bodyid_pre, bodyid_post, type_post, weight).
    """
    directory = Path(directory)
    neurons = _read_csv(directory / "mb_neurons.csv")
    edges = _read_csv(directory / "mb_kc_mbon.csv")

    kcs = {row["bodyid"] for row in neurons if row.get("mbclass") == "KC"}
    n_kc = len(kcs) or PUBLISHED["n_kc"][0]

    claw_rows = [row for row in neurons if row.get("mbclass") == "KC" and row.get("claws")]
    if claw_rows:
        claws = int(round(sum(float(r["claws"]) for r in claw_rows) / len(claw_rows)))
    else:
        claws = PUBLISHED["claws"][0]

    totals: dict[str, float] = {}
    for row in edges:
        totals[row["type_post"]] = totals.get(row["type_post"], 0.0) + float(row["weight"])
    mean_total = (sum(totals.values()) / len(totals)) if totals else 1.0
    scale = {k: v / mean_total for k, v in totals.items()} or {
        name: 1.0 for name in MBON_CELL_TYPES
    }

    return ConnectomeSource(
        name=f"yerel export ({directory})",
        n_kc=n_kc,
        claws=max(claws, 1),
        mbon_scale=scale,
        provenance=f"{len(neurons)} noron, {len(edges)} KC->MBON kenari okundu",
    )


def resolve(directory: str | Path | None) -> ConnectomeSource:
    """Use a local export if one is given and readable, else the defaults."""
    if directory is None:
        return default_source()
    try:
        return load_export(directory)
    except (OSError, KeyError, ValueError) as exc:
        fallback = default_source()
        fallback.provenance += f" [export okunamadi: {exc}]"
        return fallback
