"""The mushroom body: where a fly stores what it has learned to like.

This is the one part of the model that is not a caricature.  The circuit motif
below is the textbook Drosophila associative memory circuit:

  * A small number of projection neurons (here: visual projection neurons)
    diverge onto a large population of Kenyon cells (~2000 per hemisphere in
    the male CNS / hemibrain reconstructions).
  * Each Kenyon cell samples a handful of inputs at random -- about 6-7 claws
    per cell, with no discernible structure (Caron et al. 2013).  This random
    expansion turns overlapping inputs into near-orthogonal sparse codes.
  * The APL neuron provides global feedback inhibition, so only ~5-10% of
    Kenyon cells fire for any one stimulus.
  * Kenyon cells converge onto a few dozen mushroom body output neurons
    (MBONs), each confined to one compartment of the lobes.  MBON populations
    are biased towards approach or avoidance.
  * Dopaminergic neurons (DANs) tile the same compartments.  When a DAN fires
    close in time to a Kenyon cell, the KC->MBON synapse *depresses*.  Learning
    in the fly is subtractive: you start with everything looking good and carve
    away the parts that turned out badly.

So "training a fly on your Bumble taste" is literally: show a stimulus, fire
the reward DAN or the punishment DAN depending on which way you swiped, and
let the KC->MBON synapses depress.

References worth reading before trusting any of this:
Aso et al. 2014 eLife (compartment map); Hige et al. 2015 Neuron (depression);
Cohn et al. 2015 Cell (dopaminergic gating); Handler et al. 2019 Cell
(timing-dependent sign); Modi, Shuai & Turner 2020 Annu Rev Neurosci (review).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The two output channels we model, named after real compartments.
#: MBON-gamma1pedc>alpha/beta promotes approach and is depressed by the
#: aversive PPL1-gamma1pedc DAN; MBON-gamma4>gamma1gamma2 promotes avoidance
#: and is depressed by the appetitive PAM-gamma4 DAN.
MBON_CHANNELS = ("approach", "avoid")
MBON_CELL_TYPES = ("MBON-g1pedc>a/b", "MBON-g4>g1g2")
DAN_CELL_TYPES = ("PPL1-g1pedc", "PAM-g4<g1g2")


@dataclass
class MBParams:
    n_input: int = 144        # visual projection channels arriving in the calyx
    n_kc: int = 2000          # Kenyon cells per hemisphere
    claws: int = 6            # dendritic claws per KC (Caron et al. 2013)
    sparsity: float = 0.08    # fraction of KCs the APL lets through
    eta: float = 0.60         # depression per DAN-KC pairing
    recovery: float = 0.001   # slow drift back to baseline (forgetting)
    w0: float = 1.0           # naive KC->MBON weight
    w_min: float = 0.02       # synapses depress, they do not invert
    seed: int = 20250914


class MushroomBody:
    """KC expansion coding plus dopamine-gated synaptic depression."""

    def __init__(self, params: MBParams | None = None):
        self.params = params or MBParams()
        rng = np.random.default_rng(self.params.seed)
        p = self.params

        # Random claw connectivity.  Each KC samples `claws` inputs without
        # replacement; weights are lognormal, as synapse counts tend to be.
        self.kc_inputs = np.stack(
            [rng.choice(p.n_input, size=p.claws, replace=False) for _ in range(p.n_kc)]
        ).astype(np.int32)
        self.kc_weights = rng.lognormal(0.0, 0.35, size=(p.n_kc, p.claws)).astype(
            np.float32
        )

        # KC -> MBON weights start uniform: a naive fly finds everything
        # equally worth approaching.
        self.w_out = np.full((len(MBON_CHANNELS), p.n_kc), p.w0, dtype=np.float32)
        self.n_pairings = 0

    # -- encoding ---------------------------------------------------------
    def kenyon_cells(self, vpn: np.ndarray) -> np.ndarray:
        """Sparse Kenyon cell code for one stimulus.

        Returns a length ``n_kc`` vector that is zero for all but roughly
        ``sparsity * n_kc`` cells.
        """
        p = self.params
        if vpn.shape[0] != p.n_input:
            raise ValueError(f"expected {p.n_input} inputs, got {vpn.shape[0]}")
        drive = (self.kc_weights * vpn[self.kc_inputs]).sum(axis=1)

        # APL: raise a global inhibitory threshold until only k cells survive.
        k = max(1, int(round(p.sparsity * p.n_kc)))
        cutoff = np.partition(drive, -(k + 1))[-(k + 1)]
        kc = np.maximum(drive - cutoff, 0.0)
        total = kc.sum()
        return (kc / total if total > 0 else kc).astype(np.float32)

    # -- read-out ---------------------------------------------------------
    def mbon(self, kc: np.ndarray) -> np.ndarray:
        """MBON firing rates: ``(approach, avoid)``."""
        return self.w_out @ kc

    def valence(self, kc: np.ndarray) -> float:
        """Net approach drive in [-1, 1].

        The fly does not read absolute MBON rates, it reads the *balance*
        between opposing compartments -- the standard "MBON see-saw".
        """
        approach, avoid = self.mbon(kc)
        return float((approach - avoid) / (approach + avoid + 1e-6))

    # -- plasticity -------------------------------------------------------
    def teach(self, kc: np.ndarray, dan: np.ndarray) -> None:
        """Apply dopamine-gated depression.

        Parameters
        ----------
        kc
            Kenyon cell activity for the stimulus that was just presented.
        dan
            Dopaminergic drive per compartment, same order as
            ``MBON_CHANNELS``.  ``[0, 1]`` is reward (depress the avoidance
            MBON), ``[1, 0]`` is punishment (depress the approach MBON).
        """
        p = self.params
        active = kc / (kc.max() + 1e-9)  # eligibility, normalised per stimulus
        depression = p.eta * np.outer(np.asarray(dan, dtype=np.float32), active)
        self.w_out *= 1.0 - depression

        # Dopamine-dependent forgetting: unreinforced synapses creep back to
        # baseline, which is why a fly's memory decays over hours.
        self.w_out += p.recovery * (p.w0 - self.w_out)
        np.clip(self.w_out, p.w_min, p.w0, out=self.w_out)
        self.n_pairings += 1

    # -- diagnostics ------------------------------------------------------
    def memory_load(self) -> dict[str, float]:
        """How far each compartment has been carved away from naive."""
        p = self.params
        depressed = (self.w_out < 0.9 * p.w0).mean(axis=1)
        return {
            name: float(val) for name, val in zip(MBON_CHANNELS, depressed)
        } | {"pairings": float(self.n_pairings)}

    def state(self) -> dict:
        return {
            "kc_inputs": self.kc_inputs,
            "kc_weights": self.kc_weights,
            "w_out": self.w_out,
            "n_pairings": np.asarray(self.n_pairings),
        }

    def load_state(self, state: dict) -> None:
        self.kc_inputs = np.asarray(state["kc_inputs"], dtype=np.int32)
        self.kc_weights = np.asarray(state["kc_weights"], dtype=np.float32)
        self.w_out = np.asarray(state["w_out"], dtype=np.float32)
        self.n_pairings = int(np.asarray(state["n_pairings"]))
