"""Command-line interface.

    shades2shapes diagnose IMAGE [options]
    shades2shapes discriminate IMAGE IMAGE [IMAGE ...] [options]
    shades2shapes pyramid STORE.zarr [options]

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
                   help="band to analyse: auto (default), r, g, b, gray, or a band or variable "
                        "name of a GeoTIFF/NetCDF file. For discriminate, one value for all "
                        "images or a comma-separated list, one per image.")
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
                   help="physical pixel size, to report lengths in physical units "
                        "(default: from the georeferencing of GeoTIFF/NetCDF files)")
    g.add_argument("--unit", default="km", help="unit of --pixel-size (default km)")
    p.add_argument("--save-maps", action="store_true",
                   help="also write the maps (mask, ridges, orientation...) as NetCDF, "
                        "georeferenced for GeoTIFF/NetCDF inputs (needs the geo extra)")
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
        if a.save_maps:
            d.to_xarray().to_netcdf(out / f"{stem}_maps.nc")
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
            if a.save_maps:
                d.to_xarray().to_netcdf(out / f"{stem}_maps.nc")
    print(res.summary())
    print(f"\nResults written to {out.resolve()}")


def cmd_pyramid(a):
    from .multiscale import diagnose_pyramid

    kw = _diag_kwargs(a)
    kw.pop("pixel_size")
    if a.pixel_size:
        sys.exit("pyramid: the pixel size is read from each level; drop --pixel-size")
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    if bbox is not None and len(bbox) != 4:
        sys.exit("--bbox: give xmin,ymin,xmax,ymax")
    res = diagnose_pyramid(a.store, levels=a.levels.split(",") if a.levels else None,
                           variable=a.variable, bbox=bbox, bbox_crs=a.bbox_crs, mask=a.mask,
                           fill_nodata_pixels=not a.no_fill, max_nodata=a.max_nodata,
                           transform=a.transform, min_size=a.min_size, channel=a.channel, **kw)
    out = Path(a.out or "s2s_pyramid")
    res.save(out, plots=not a.no_plot, maps=a.save_maps)
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
    pd_.add_argument("images", nargs="+",
                     help="image files (PNG, JPEG, TIFF...; GeoTIFF and NetCDF with the geo extra)")
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
                    help="also write each image's diagnosis JSON and figure (and maps, "
                         "with --save-maps)")
    _add_common(pc)
    pc.set_defaults(func=cmd_discriminate)

    pp = sub.add_parser("pyramid", help="diagnose the same area at every level of a Zarr pyramid")
    pp.add_argument("store", help="Zarr store with levels 0, 1, 2... (needs the geo extra and zarr)")
    pp.add_argument("--variable", default=None, help="variable to analyse, e.g. Rrs")
    pp.add_argument("--levels", default=None, help="comma-separated levels (default: all)")
    pp.add_argument("--bbox", default=None,
                    help="xmin,ymin,xmax,ymax of the area (map coordinates, or --bbox-crs)")
    pp.add_argument("--bbox-crs", default=None, help="CRS of --bbox, e.g. EPSG:4326")
    pp.add_argument("--mask", default=None,
                    help="variable flagging invalid pixels (non-zero = invalid), e.g. mask")
    pp.add_argument("--no-fill", action="store_true",
                    help="skip levels with no-data pixels instead of filling them")
    pp.add_argument("--max-nodata", type=float, default=0.25,
                    help="skip levels with more no-data than this fraction (default 0.25)")
    pp.add_argument("--transform", choices=["log10"], default=None,
                    help="transform the image first (log10 for reflectances)")
    pp.add_argument("--min-size", type=int, default=64,
                    help="skip levels smaller than this, in px (default 64)")
    _add_common(pp)
    pp.set_defaults(func=cmd_pyramid)

    a = p.parse_args(argv)
    if a.command == "discriminate" and len(a.images) < 2:
        p.error("discriminate needs at least two images")
    a.func(a)


if __name__ == "__main__":
    main()
