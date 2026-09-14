"""One whole (simulated) fly: eye -> mushroom body -> descending neurons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .connectome import ConnectomeSource, resolve
from .dataset import load_image
from .eye import EyeParams, FlyEye
from .mushroom_body import MBON_CHANNELS, MBParams, MushroomBody
from .steering import Decision, Steering, SteeringParams

#: DAN drive per outcome, ordered like MBON_CHANNELS = (approach, avoid).
#: A right swipe is a reward: PAM DANs depress the avoidance MBON.
#: A left swipe is a punishment: PPL1 DANs depress the approach MBON.
DAN_DRIVE = {"right": (0.0, 1.0), "left": (1.0, 0.0)}


@dataclass
class Percept:
    vpn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    valence: float


class Fly:
    def __init__(
        self,
        eye_params: EyeParams | None = None,
        mb_params: MBParams | None = None,
        steering_params: SteeringParams | None = None,
        connectome: ConnectomeSource | None = None,
    ):
        self.eye = FlyEye(eye_params)
        self.connectome = connectome or resolve(None)
        base = mb_params or MBParams()
        base = MBParams(**{**vars(base), "n_input": self.eye.n_vpn})
        self.mb = MushroomBody(self.connectome.apply(base))
        self.steering = Steering(steering_params)

    # -- perception -------------------------------------------------------
    def perceive(self, image: np.ndarray) -> Percept:
        return self.percept_from_vpn(self.eye.vpn(image))

    def percept_from_vpn(self, vpn: np.ndarray) -> Percept:
        """Percept from an already-computed visual projection vector.

        Optic lobe processing does not change with learning, so anything that
        shows the same stimuli repeatedly should compute ``eye.vpn`` once and
        come in through here.
        """
        kc = self.mb.kenyon_cells(vpn)
        return Percept(vpn=vpn, kc=kc, mbon=self.mb.mbon(kc), valence=self.mb.valence(kc))

    def look_at(self, path: str | Path) -> np.ndarray:
        return load_image(path, self.eye.params.grid)

    # -- behaviour --------------------------------------------------------
    def swipe(self, image: np.ndarray) -> Decision:
        return self.steering.decide(self.perceive(image).valence)

    def swipe_vpn(self, vpn: np.ndarray) -> Decision:
        return self.steering.decide(self.percept_from_vpn(vpn).valence)

    def train(self, image: np.ndarray, swipe: str) -> Decision:
        """Show a stimulus, let the fly commit, then fire the DANs.

        The decision is taken *before* the reinforcement, which is how you get
        an honest online accuracy estimate: the fly never sees the answer
        before it answers.
        """
        return self.train_vpn(self.eye.vpn(image), swipe)

    def train_vpn(self, vpn: np.ndarray, swipe: str) -> Decision:
        """As :meth:`train`, starting from a cached visual projection vector."""
        if swipe not in DAN_DRIVE:
            raise ValueError("swipe 'left' veya 'right' olmali")
        percept = self.percept_from_vpn(vpn)
        decision = self.steering.decide(percept.valence)
        self.mb.teach(percept.kc, DAN_DRIVE[swipe])
        return decision

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "eye": asdict(self.eye.params),
            "mb": asdict(self.mb.params),
            "steering": asdict(self.steering.params),
            "connectome": self.connectome.name,
            "mbon_channels": list(MBON_CHANNELS),
        }
        np.savez_compressed(path, meta=json.dumps(meta), **self.mb.state())

    @classmethod
    def load(cls, path: str | Path) -> "Fly":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        fly = cls(
            eye_params=EyeParams(**meta["eye"]),
            mb_params=MBParams(**meta["mb"]),
            steering_params=SteeringParams(**meta["steering"]),
        )
        fly.mb.load_state({k: data[k] for k in ("kc_inputs", "kc_weights", "w_out", "n_pairings")})
        return fly
