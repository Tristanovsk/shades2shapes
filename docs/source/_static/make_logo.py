"""Generate the shades2shapes logo.

The background is the Sato ridge response of the example eddy image, computed
with shades2shapes itself. The wordmark merges 'p' (shapes, red) and 'd'
(shades, green) into one glyph, so it reads both ways.

    python make_logo.py   ->  logo.svg/png, logo_dark.svg/png, icon.svg/png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib._text_helpers import layout

from shades2shapes import diagnose
from shades2shapes.features import ridge_response
from skimage import exposure
from skimage.transform import rescale

HERE = Path(__file__).parent
FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
FP = FontProperties(fname=FONT)
RED = "#D62F2F"
GREEN = "#3FAE49"
AMBER = "#E39A1E"   # midpoint of the p -> d gradient

# ----------------------------------------------------------------- background
UP = 3  # ridge filter run on a 3x upsampled image: sharper filaments when enlarged
_d = diagnose(HERE.parent / "examples" / "single_eddy.png", channel="g")
_up = rescale(_d.band, UP, order=3)
_r = ridge_response(_up, sigmas=(1.5, 3, 4.5))
_r = np.clip(_r / np.percentile(_r, 99.7), 0, 1)
# local contrast brings out the faint inner spiral arms; blending with the raw
# ridge map keeps the large arms dominant over the fine mesh
_c = exposure.equalize_adapthist(_r, kernel_size=_r.shape[0] // 6, clip_limit=0.02)
_r = 0.55 * _r ** 0.7 + 0.45 * _c
_r = np.clip((_r - np.percentile(_r, 50)) / (np.percentile(_r, 99.8) - np.percentile(_r, 50)), 0, 1)
RIDGES = _r ** 1.1
CX, CY = _d.eddies[0].x * UP, _d.eddies[0].y * UP


# ----------------------------------------------------------------- colour
def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def _to_oklch(hex_):
    lab = _M2 @ np.cbrt(_M1 @ _srgb_to_lin(to_rgb(hex_)))
    return np.array([lab[0], np.hypot(lab[1], lab[2]), np.arctan2(lab[2], lab[1])])


def _oklch_to_lin(L, C, h):
    lms = np.linalg.solve(_M2, np.array([L, C * np.cos(h), C * np.sin(h)])) ** 3
    return np.linalg.solve(_M1, lms)


def _from_oklch(lch):
    """OKLCH -> sRGB, reducing chroma (keeping L and hue) until in gamut."""
    L, C, h = lch
    lin = _oklch_to_lin(L, C, h)
    if np.all((lin >= -1e-6) & (lin <= 1 + 1e-6)):
        return _lin_to_srgb(lin)
    lo, hi = 0.0, C
    for _ in range(30):
        mid = (lo + hi) / 2
        lin = _oklch_to_lin(L, mid, h)
        if np.all((lin >= -1e-6) & (lin <= 1 + 1e-6)):
            lo = mid
        else:
            hi = mid
    return _lin_to_srgb(_oklch_to_lin(L, lo, h))


def _to_oklab(hex_):
    return _M2 @ np.cbrt(_M1 @ _srgb_to_lin(to_rgb(hex_)))


def _from_oklab(lab):
    lin = np.linalg.solve(_M1, np.linalg.solve(_M2, lab) ** 3)
    return _lin_to_srgb(lin)


def gradient(stops, n=256):
    """n colours through the given colour stops, interpolated in OKLab.

    Piecewise OKLab between in-gamut stops stays vivid and continuous, unlike a
    plain RGB blend (muddy middle) or an OKLCH arc (clips at the gamut edge)."""
    labs = np.array([_to_oklab(c) for c in stops])
    pos = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    lab = np.stack([np.interp(t, pos, labs[:, k]) for k in range(3)], axis=1)
    return np.array([_from_oklab(v) for v in lab])


def glyphs(text, size):
    """[(char, x_offset)] using the font's advances and kerning."""
    font = FT2Font(FONT)
    font.set_size(size, 72)
    return [(it.char, it.x) for it in layout(text, font)]


def _counter_right_edge(path, baseline):
    """Right edge x(y) of the p counter (the hole of the bowl)."""
    polys = path.to_polygons()
    counter = min((q for q in polys if q[:, 1].min() > baseline - 1),
                  key=lambda q: np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
    xc = counter[:, 0].mean()
    right = counter[counter[:, 0] >= xc]
    order = np.argsort(right[:, 1])
    return right[order, 1], right[order, 0]


def _halo(ax, artist_path_or_poly, halo, width, z):
    if halo is None:
        return
    if isinstance(artist_path_or_poly, np.ndarray):
        a = plt.Polygon(artist_path_or_poly, closed=True, fc=halo, ec=halo, lw=width,
                        joinstyle="round", zorder=z, alpha=0.9)
    else:
        a = PathPatch(artist_path_or_poly, fc=halo, ec=halo, lw=width, joinstyle="round",
                      zorder=z, alpha=0.9)
    ax.add_patch(a)


def draw_pd(ax, x, baseline, size, z=3, halo=None, halo_w=10):
    """Merged glyph: red 'p' (shapes) + green ascender of a 'd' (shades).

    The green stem has the weight of the p stem, is flush with the right edge of
    the bowl and runs from the baseline to the ascender, over the red bowl. Its
    inner edge follows the counter, so the hole of the bowl stays open.
    Returns the bowl centre (eddy anchor).
    """
    p = TextPath((x, baseline), "p", size=size, prop=FP)
    pv = np.vstack(p.to_polygons())             # outline points only (no dummy 0,0)
    desc = pv[pv[:, 1] < baseline - 0.1 * size]
    stem_l, stem_r = desc[:, 0].min(), desc[:, 0].max()
    sw = stem_r - stem_l                        # stem weight of the p
    xh = TextPath((0, 0), "x", size=size, prop=FP).vertices[:, 1].max()
    asc = TextPath((0, 0), "d", size=size, prop=FP).vertices[:, 1].max()
    right = pv[:, 0].max()
    cy, cx = _counter_right_edge(p, baseline)
    y_lo, y_hi = baseline, baseline + asc
    ys = np.linspace(y_lo, y_hi, 200)
    left = np.full_like(ys, right - sw)
    inside = (ys >= cy.min()) & (ys <= cy.max())
    left[inside] = np.maximum(left[inside], np.interp(ys[inside], cy, cx))
    poly = np.r_[np.c_[left, ys], [[right, y_hi], [right, y_lo]]]
    _halo(ax, p, halo, halo_w, z - 1)
    _halo(ax, poly, halo, halo_w, z - 1)
    # p filled with a horizontal gradient: solid red up to the p stem, red -> green
    # across the bowl, solid green from the d bar onwards
    x_min = pv[:, 0].min()
    x_a, x_b = stem_r, right - sw
    n = 512
    xs = np.linspace(x_min, right, n)
    t = np.clip((xs - x_a) / (x_b - x_a), 0, 1)
    ramp = gradient([RED, AMBER, GREEN], 256)
    cols = ramp[np.round(t * 255).astype(int)]
    img = np.repeat(cols[None, :, :], 2, axis=0)
    im = ax.imshow(img, extent=[x_min, right, pv[:, 1].min(), pv[:, 1].max()],
                   origin="lower", interpolation="bilinear", zorder=z, aspect="auto")
    clip = PathPatch(p, transform=ax.transData)
    im.set_clip_path(clip)
    ax.add_patch(plt.Polygon(poly, closed=True, fc=GREEN, ec="none", zorder=z + 1))
    return (pv[:, 0].min() + sw + right) / 2, baseline + xh / 2


def draw_wordmark(ax, size, cx, baseline, ink, z=3, halo=None, halo_w=10):
    """Draw 'sha[p/d]es' horizontally centred on cx. Returns the p/d bowl centre."""
    full = np.vstack(TextPath((0, 0), "shapes", size=size, prop=FP).to_polygons())
    x0 = cx - (full[:, 0].min() + full[:, 0].max()) / 2
    anchor = None
    for ch, dx in glyphs("shapes", size):
        if ch == "p":
            anchor = draw_pd(ax, x0 + dx, baseline, size, z, halo, halo_w)
        else:
            tp = TextPath((x0 + dx, baseline), ch, size=size, prop=FP)
            _halo(ax, tp, halo, halo_w, z - 1)
            ax.add_patch(PathPatch(tp, fc=ink, ec="none", zorder=z))
    return anchor


def draw_background(ax, W, H, dark, centre=None, zoom=1.25, strength=None, fade_r=0.62):
    """Ridge response of the eddy image, eddy centre placed at `centre`."""
    h, w = RIDGES.shape
    ox, oy = centre if centre is not None else (W / 2, H / 2)
    scale = H * zoom / h
    ext = [ox - CX * scale, ox + (w - CX) * scale, oy - (h - CY) * scale, oy + CY * scale]
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot((xx - CX) / (fade_r * w), (yy - CY) / (fade_r * h))
    fade = np.clip(1.2 - rr, 0, 1) ** 1.4
    strength = strength or (1.0 if dark else 0.95)
    rgba = np.zeros((h, w, 4))
    rgba[..., :3] = to_rgb("#7BE08A" if dark else "#1F2A27")
    rgba[..., 3] = RIDGES * fade * strength
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear", zorder=1)


def canvas(W, H, bg):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    return fig, ax


def render_logo(name, dark=False, W=1600, H=1600, size=380, zoom=1.0):
    bg = "#0E1A17" if dark else "#FFFFFF"
    fig, ax = canvas(W, H, bg)
    xh = TextPath((0, 0), "x", size=size, prop=FP).vertices[:, 1].max()
    anchor = draw_wordmark(ax, size, W / 2, H / 2 - xh / 2, "#F2F2F2" if dark else "#151515",
                           halo=bg, halo_w=18)
    draw_background(ax, W, H, dark, centre=anchor, zoom=zoom)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", facecolor=bg)
    plt.close(fig)


def render_icon(name, dark=False, S=1024, size=700):
    """Square icon: merged p/d glyph at the centre of the eddy."""
    bg = "#0E1A17" if dark else "#FFFFFF"
    fig, ax = canvas(S, S, bg)
    p = np.vstack(TextPath((0, 0), "p", size=size, prop=FP).to_polygons())
    d = np.vstack(TextPath((0, 0), "d", size=size, prop=FP).to_polygons())
    x = S / 2 - (p[:, 0].min() + p[:, 0].max()) / 2
    y0, y1 = p[:, 1].min(), d[:, 1].max()
    baseline = S / 2 - (y0 + y1) / 2
    anchor = draw_pd(ax, x, baseline, size, halo=bg, halo_w=22)
    draw_background(ax, S, S, dark, centre=anchor, zoom=1.9, fade_r=0.45,
                    strength=0.9 if dark else 0.7)
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"{name}.{ext}", facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    render_logo("logo")
    render_logo("logo_dark", dark=True)
    render_icon("icon")
    render_icon("icon_dark", dark=True)
    print("written:", sorted(p.name for p in HERE.glob("*.png")))
