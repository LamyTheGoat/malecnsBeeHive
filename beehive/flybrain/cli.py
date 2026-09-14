"""Command line front end.

    python -m flybrain demo
    python -m flybrain eye  --image foto.jpg
    python -m flybrain train --photos ./fotolar
    python -m flybrain swipe --photos ./yeni --brain fly.npz
    python -m flybrain info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .connectome import resolve
from .dataset import LabelStore, list_photos
from .fly import Fly
from .mushroom_body import DAN_CELL_TYPES, MBON_CELL_TYPES
from .synthetic import LABEL_NOISE, make_profiles

BAR = "=" * 64


def _sparkline(values, width: int = 48) -> str:
    ramp = "▁▂▃▄▅▆▇█"
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return ""
    if len(vals) > width:
        edges = np.linspace(0, len(vals), width + 1).astype(int)
        vals = np.array([vals[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    lo, hi = vals.min(), vals.max()
    scaled = np.zeros_like(vals) if hi - lo < 1e-9 else (vals - lo) / (hi - lo)
    return "".join(ramp[int(round(v * (len(ramp) - 1)))] for v in scaled)


def _blocks(correct: list[bool], n_blocks: int = 8) -> list[float]:
    """Accuracy in consecutive non-overlapping blocks of trials.

    Trial-by-trial accuracy on a noisy task is unreadable; blocks are how
    learning curves are actually reported in the behaviour literature.
    """
    if not correct:
        return []
    edges = np.linspace(0, len(correct), min(n_blocks, len(correct)) + 1).astype(int)
    return [
        sum(correct[a:b]) / (b - a)
        for a, b in zip(edges[:-1], edges[1:]) if b > a
    ]


def _optics_ceiling(fly, profiles) -> float:
    """How much of the hidden taste survives the fly's optics at all.

    A least-squares read-out straight off the visual projection channels --
    not something the fly does, just a yardstick.  The gap between this and
    the label-noise ceiling is what the 5-degree eye threw away; the gap
    between this and the fly's own score is what the learning rule cost.
    """
    import numpy as np

    x = np.stack([fly.eye.vpn(p.image) for p in profiles])
    x = np.hstack([x, np.ones((len(x), 1), dtype=np.float32)])
    y = np.array([1.0 if p.swipe == "right" else -1.0 for p in profiles])
    split = len(x) * 2 // 3
    w = np.linalg.solve(
        x[:split].T @ x[:split] + np.eye(x.shape[1]), x[:split].T @ y[:split]
    )
    return float(np.mean(np.sign(x[split:] @ w) == y[split:]))


def _make_fly(args) -> Fly:
    return Fly(connectome=resolve(getattr(args, "connectome", None)))


# -- commands -------------------------------------------------------------
def cmd_info(args) -> int:
    fly = _make_fly(args)
    print(BAR)
    print("BeeHive: sanal meyve sinegi beyni")
    print(BAR)
    print(fly.connectome.describe())
    print()
    print(f"  Ommatidyum / goz      : {fly.eye.n_ommatidia}")
    print(f"  Gorsel projeksiyon k. : {fly.eye.n_vpn}")
    print(f"  Kenyon hucresi        : {fly.mb.params.n_kc}")
    print(f"  Ayni anda aktif KC    : ~{int(fly.mb.params.sparsity * fly.mb.params.n_kc)}"
          f" (%{fly.mb.params.sparsity * 100:.0f}, APL geri besleme inhibisyonu)")
    print(f"  MBON kanallari        : {', '.join(MBON_CELL_TYPES)}")
    print(f"  Ogretmen dopamin non. : {', '.join(DAN_CELL_TYPES)}")
    print(f"  Inis hucresi          : DNa02 (sol/sag donus)")
    return 0


def cmd_demo(args) -> int:
    fly = _make_fly(args)
    grid = fly.eye.params.grid
    train = make_profiles(args.n, grid, seed=args.seed)
    test = make_profiles(max(120, args.n // 3), grid, seed=args.seed + 999)

    print(BAR)
    print(f"Egitim: {len(train)} sentetik profil (gizli bir zevk kurali + "
          f"%{int(LABEL_NOISE * 100)} etiket gurultusu)")
    print(f"Test  : {len(test)} hic gorulmemis profil")
    print(BAR)

    # The optic lobe does not change with learning, so look at every stimulus
    # exactly once and reuse the projection vectors.
    train_vpn = [fly.eye.vpn(p.image) for p in train]
    test_vpn = [fly.eye.vpn(p.image) for p in test]
    test_truth = [p.swipe for p in test]

    def evaluate() -> float:
        return sum(
            fly.swipe_vpn(v).swipe == truth for v, truth in zip(test_vpn, test_truth)
        ) / len(test_vpn)

    n_points = 16
    edges = np.linspace(0, len(train), n_points + 1).astype(int)
    curve, online = [evaluate()], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for i in range(lo, hi):
            decision = fly.train_vpn(train_vpn[i], train[i].swipe)
            online.append(decision.swipe == train[i].swipe)
        curve.append(evaluate())

    print(f"Test dogrulugu ({len(curve)} olcum, egitim boyunca):")
    print(f"  {_sparkline(curve, width=len(curve))}")
    print("  " + " ".join(f"{c * 100:.0f}" for c in curve))
    print(f"  naif sinek %{curve[0] * 100:.0f}  ->  egitilmis sinek %{curve[-1] * 100:.0f}")
    print(f"Egitim sirasinda online dogruluk: %{sum(online) / len(online) * 100:.0f}")

    print()
    print(BAR)
    print(f"Gorulmemis {len(test)} profilde dogruluk    : %{curve[-1] * 100:.0f}")
    print(f"  sans seviyesi                       : %50")
    print(f"  sineğin optigi neye izin veriyor    : "
          f"%{_optics_ceiling(fly, train + test) * 100:.0f}")
    print(f"  etiket gurultusu tavani             : %{(1 - LABEL_NOISE) * 100:.0f}")
    print()
    load = fly.mb.memory_load()
    print(f"Depresyona ugramis KC->MBON sinapsi: yaklas %{load['approach'] * 100:.0f}, "
          f"kac %{load['avoid'] * 100:.0f} ({int(load['pairings'])} eslestirme)")

    if args.brain:
        fly.save(args.brain)
        print(f"Beyin kaydedildi: {args.brain}")
    return 0


def cmd_eye(args) -> int:
    fly = _make_fly(args)
    image = fly.look_at(args.image)
    print(BAR)
    print(f"{Path(args.image).name} -> {fly.eye.n_ommatidia} ommatidyum, ~5 derece cozunurluk")
    print(BAR)
    print(fly.eye.ascii_view(image))
    percept = fly.perceive(image)
    print()
    print(f"aktif Kenyon hucresi : {int((percept.kc > 0).sum())}")
    print(f"MBON (yaklas, kac)   : {percept.mbon[0]:.3f}, {percept.mbon[1]:.3f}")
    print(f"valans               : {percept.valence:+.3f}")
    return 0


def _ask(name: str) -> str | None:
    while True:
        answer = input(f"{name}  [j=sol/nope, l=sag/like, s=atla, q=cik]: ").strip().lower()
        if answer in ("j", "l", "s", "q"):
            return {"j": "left", "l": "right", "s": None, "q": "quit"}[answer]
        print("  j, l, s veya q girin.")


def cmd_train(args) -> int:
    fly = Fly.load(args.brain) if args.brain and Path(args.brain).exists() else _make_fly(args)
    photos = list_photos(args.photos)
    if not photos:
        print(f"{args.photos} icinde resim yok.", file=sys.stderr)
        return 1

    store = LabelStore(args.labels or Path(args.photos) / "swipes.json")
    correct = []
    for path in photos:
        label = store.get(path.name)
        if label is None:
            if args.batch:
                continue
            image = fly.look_at(path)
            print(BAR)
            print(fly.eye.ascii_view(image))
            guess = fly.swipe(image)
            print(f"sineğin tahmini: {guess.swipe} (guven %{guess.confidence * 100:.0f})")
            label = _ask(path.name)
            if label == "quit":
                break
            if label is None:
                continue
            store.set(path.name, label)
            store.save()
        image = fly.look_at(path)
        decision = fly.train(image, label)
        correct.append(decision.swipe == label)

    if not correct:
        print("Etiketlenmis profil yok; once etiketle.", file=sys.stderr)
        return 1

    print(BAR)
    blocks = _blocks(correct, n_blocks=6)
    print(f"{len(correct)} profilde egitim yapildi.")
    print(f"Ogrenme egrisi: {_sparkline(blocks, width=len(blocks))}  "
          + "  ".join(f"%{b * 100:.0f}" for b in blocks))
    out = args.brain or "fly.npz"
    fly.save(out)
    print(f"Beyin kaydedildi: {out}")
    return 0


def cmd_swipe(args) -> int:
    if not args.brain or not Path(args.brain).exists():
        print("Once 'train' ile bir beyin egitip kaydedin.", file=sys.stderr)
        return 1
    fly = Fly.load(args.brain)
    photos = list_photos(args.photos)
    print(f"{'profil':40s} {'karar':>6s} {'valans':>8s} {'guven':>7s}")
    print(BAR)
    for path in photos:
        decision = fly.swipe(fly.look_at(path))
        arrow = "-> SAG" if decision.swipe == "right" else "<- SOL"
        print(f"{path.name[:40]:40s} {arrow:>6s} {decision.valence:+8.3f} "
              f"{decision.confidence * 100:6.0f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flybrain",
        description="Konnektom esinli bir sinek beynini kendi zevkinize gore egitin.",
    )
    parser.add_argument("--connectome", help="export_mb_connectome.R ciktisinin klasoru")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="model ve konnektom kaynagini goster")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("demo", help="sentetik profillerle ogrenmeyi izle")
    p.add_argument("-n", type=int, default=300, help="egitim profili sayisi")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--brain", help="egitilmis beyni bu dosyaya kaydet")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("eye", help="sineğin bir fotografi nasil gordugunu goster")
    p.add_argument("--image", required=True)
    p.set_defaults(func=cmd_eye)

    p = sub.add_parser("train", help="fotograf klasorunde egit")
    p.add_argument("--photos", required=True)
    p.add_argument("--labels", help="swipes.json yolu")
    p.add_argument("--brain", help="beyin dosyasi (varsa yuklenir, sonra kaydedilir)")
    p.add_argument("--batch", action="store_true", help="sadece etiketli olanlari kullan, soru sorma")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("swipe", help="egitilmis beyinle yeni profilleri degerlendir")
    p.add_argument("--photos", required=True)
    p.add_argument("--brain", required=True)
    p.set_defaults(func=cmd_swipe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
