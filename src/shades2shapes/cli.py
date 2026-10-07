"""Command-line interface.

    shades2shapes diagnose IMAGE [options]
    shades2shapes discriminate IMAGE IMAGE [IMAGE ...] [options]

(`s2s` is a short alias.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _add_common(p):
    g = p.add_argument_group("diagnosis options")
    g.add_argument("--channel", default="auto",
                   help="band to analyse: auto (default), r, g, b, gray. For discriminate, "
                        "one value for all images or a comma-separated list, one per image.")
    g.add_argument("--dark-ridges", action="store_true",
                   help="filaments are darker than the background")
    g.add_argument("--mask-method", choices=["otsu", "local"], default="otsu",
                   help="filament thresholding: global Otsu (default) or local/adaptive")
    g.add_argument("--tensor-sigma", type=float, default=4.0,
                   help="structure-tensor scale in px (default 4)")
    g.add_argument("--no-eddies", action="store_true", help="skip eddy detection")
    g.add_argument("--eddy-min-alignment", type=float, default=0.55)
    g.add_argument("--eddy-min-significance", type=float, default=1.5)
    g.add_argument("--pixel-size", type=float, default=None,
                   help="physical pixel size, to report lengths in physical units")
    g.add_argument("--unit", default="km", help="unit of --pixel-size (default km)")
    p.add_argument("-o", "--out", default=None, help="output directory")
    p.add_argument("--no-plot", action="store_true", help="do not write figures")


def _diag_kwargs(a):
    return dict(bright_ridges=not a.dark_ridges, mask_method=a.mask_method,
                tensor_sigma=a.tensor_sigma, detect_eddies=not a.no_eddies,
                eddy_min_alignment=a.eddy_min_alignment,
                eddy_min_significance=a.eddy_min_significance,
                pixel_size=a.pixel_size, unit=a.unit)


def cmd_diagnose(a):
    from .diagnose import diagnose

    out = Path(a.out or "s2s_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    for img in a.images:
        d = diagnose(img, channel=a.channel, **_diag_kwargs(a))
        stem = Path(img).stem
        print(d.summary())
        print()
        d.to_json(out / f"{stem}_diagnosis.json")
        (out / f"{stem}_summary.txt").write_text(d.summary())
        if not a.no_plot:
            d.plot(out / f"{stem}_diagnosis.png")
    print(f"Results written to {out.resolve()}")


def cmd_discriminate(a):
    from .discriminate import discriminate

    n = len(a.images)
    ch = a.channel.split(",")
    if len(ch) == 1:
        ch = ch * n
    if len(ch) != n:
        sys.exit("--channel: give one value or one per image")
    labels = a.labels.split(",") if a.labels else None
    if labels and len(labels) != n:
        sys.exit("--labels: give one label per image")
    res = discriminate(a.images, labels=labels, channels=ch, tile=a.tile, overlap=a.overlap,
                       n_select=a.n_select, **_diag_kwargs(a))
    out = Path(a.out or "s2s_discrimination")
    res.save(out, plots=not a.no_plot)
    if a.save_diagnoses:
        for d in res.diagnoses:
            stem = Path(d.source).stem
            d.to_json(out / f"{stem}_diagnosis.json")
            if not a.no_plot:
                d.plot(out / f"{stem}_diagnosis.png")
    print(res.summary())
    print(f"\nResults written to {out.resolve()}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="shades2shapes",
        description="Diagnose and discriminate spatial patterns (filaments, networks, "
                    "texture, eddies) in images.")
    p.add_argument("--version", action="version", version=f"shades2shapes {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pd_ = sub.add_parser("diagnose", help="diagnose one or more images independently")
    pd_.add_argument("images", nargs="+")
    _add_common(pd_)
    pd_.set_defaults(func=cmd_diagnose)

    pc = sub.add_parser("discriminate", help="discriminate two or more images or groups")
    pc.add_argument("images", nargs="+")
    pc.add_argument("--labels", default=None,
                    help="comma-separated class label per image; repeat a label to group "
                         "images (default: one class per image)")
    pc.add_argument("--tile", type=int, default=64, help="tile size in px (default 64)")
    pc.add_argument("--overlap", type=float, default=0.5, help="tile overlap (default 0.5)")
    pc.add_argument("--n-select", type=int, default=4,
                    help="size of the recommended feature set (default 4)")
    pc.add_argument("--save-diagnoses", action="store_true",
                    help="also write each image's diagnosis JSON and figure")
    _add_common(pc)
    pc.set_defaults(func=cmd_discriminate)

    a = p.parse_args(argv)
    if a.command == "discriminate" and len(a.images) < 2:
        p.error("discriminate needs at least two images")
    a.func(a)


if __name__ == "__main__":
    main()
