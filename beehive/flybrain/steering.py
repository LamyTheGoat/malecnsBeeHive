"""From MBON balance to a left or right turn.

MBON axons converge in the crepine and superior medial protocerebrum onto
pathways that reach the lateral accessory lobe, which is the fly's steering
hub.  From there a small set of descending neurons carries the command to the
ventral nerve cord.  DNa02 is the best characterised of these: it is tuned to
turning, its activity predicts turn magnitude, and unilateral activation makes
the fly turn towards that side (Rayshubskiy et al.).

We model the last stage as a two-sided race.  The MBON see-saw biases left and
right DNa02 asymmetrically, noise is added at every step (flies are stochastic
turners, this is not a defect of the model), and evidence accumulates until one
side wins.  A swipe right is a turn towards the stimulus; a swipe left is a
turn away from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SteeringParams:
    #: How strongly MBON balance biases the descending neurons.
    gain: float = 8.0
    #: Turn noise per timestep, in units of accumulated evidence.
    noise: float = 0.25
    #: Maximum number of 20 ms timesteps before the fly commits anyway.
    max_steps: int = 40
    #: Evidence needed to commit to a turn.
    threshold: float = 1.0
    seed: int = 7


@dataclass
class Decision:
    swipe: str              # "right" (like) or "left" (nope)
    valence: float          # MBON balance that drove it, in [-1, 1]
    confidence: float       # |accumulated evidence| at commitment, 0..1
    steps: int              # how long the fly dithered
    dna02: tuple[float, float]  # (left, right) descending neuron drive
    trace: np.ndarray       # evidence over time, for plotting


class Steering:
    """MBON balance -> LAL -> DNa02 -> turn."""

    def __init__(self, params: SteeringParams | None = None):
        self.params = params or SteeringParams()
        self.rng = np.random.default_rng(self.params.seed)

    def descending(self, valence: float) -> tuple[float, float]:
        """Left and right DNa02 drive for a given MBON balance."""
        bias = np.tanh(self.params.gain * valence)
        # Turning towards the stimulus is driven by one side, away by the
        # other; baseline drive is symmetric so a naive fly goes straight.
        right = 0.5 * (1.0 + bias)
        left = 0.5 * (1.0 - bias)
        return float(left), float(right)

    def decide(self, valence: float) -> Decision:
        p = self.params
        left, right = self.descending(valence)
        drift = (right - left) / p.max_steps * p.threshold * 2.0

        evidence = 0.0
        trace = []
        for step in range(1, p.max_steps + 1):
            evidence += drift + self.rng.normal(0.0, p.noise / np.sqrt(p.max_steps))
            trace.append(evidence)
            if abs(evidence) >= p.threshold:
                break

        swipe = "right" if evidence > 0 else "left"
        confidence = float(min(abs(evidence) / p.threshold, 1.0))
        return Decision(
            swipe=swipe,
            valence=float(valence),
            confidence=confidence,
            steps=step,
            dna02=(left, right),
            trace=np.asarray(trace, dtype=np.float32),
        )
