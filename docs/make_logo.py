"""shades2shapes logo: point-symmetric wordmark around a vortex.

'shades' and 'shapes' mirror each other through the '2' (a 'd' turned 180 deg is a 'p').
The stem of the 'd' rises and curls into a vortex centred on the '2'; the stem of the 'p'
is its exact 180-degree image, so the two opposite bars form the two arms of the vortex.

    python docs/make_logo.py [--dark] [output.png]

Default output: docs/source/_static/shades2shapes.png, or shades2shapes_dark.png (light
letters, for dark backgrounds) with --dark.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Polygon
from matplotlib.textpath import TextPath, text_to_path

FONT = FontProperties(fname="/usr/share/fonts/truetype/gfonts/montserrat/montserrat-v14-latin-700.ttf")
SIZE = 100
D_COL = "#13a89e"   # teal: green / cyanobacteria bloom
P_COL = "#e8553f"   # red-orange: red tide
WORD = "shades2shapes"
args = [a for a in sys.argv[1:] if a != "--dark"]
dark = "--dark" in sys.argv
INK = "#e6edf3" if dark else "#1d2b3a"
out = args[0] if args else (Path(__file__).parent / "source/_static"
                            / ("shades2shapes_dark.png" if dark else "shades2shapes.png"))

# glyph x positions straight from the font layout
font = text_to_path._get_font(FONT)
font.set_size(text_to_path.FONT_SCALE, text_to_path.DPI)
glyph_info, _, _ = text_to_path.get_glyphs_with_font(font, WORD, return_new_glyphs_only=False)
scale = SIZE / text_to_path.FONT_SCALE
xs = [g[1] * scale for g in glyph_info]


def glyph(i):
    return TextPath((xs[i], 0), WORD[i], size=SIZE, prop=FONT)


ext_of = lambda s: TextPath((0, 0), s, size=SIZE, prop=FONT).get_extents()
stem_w = ext_of("l").width
asc = ext_of("l").y1
xh = ext_of("x").y1

i_d, i_p = WORD.index("d"), WORD.index("p")
gd, gp = glyph(i_d).get_extents(), glyph(i_p).get_extents()
d_x = gd.x1 - stem_w / 2            # stem centre lines
p_x = gp.x0 + stem_w / 2
cx, cy = (d_x + p_x) / 2, xh / 2    # centre of symmetry

fig = plt.figure(dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_aspect("equal")
ax.axis("off")


# ---------------------------------------------------------------- vortex
def ribbon(xy, w):
    """Filled ribbon of half-width w (array) along polyline xy."""
    d = np.gradient(xy, axis=0)
    n = np.c_[-d[:, 1], d[:, 0]] / np.hypot(*d.T)[:, None]
    return np.r_[xy + n * w[:, None], (xy - n * w[:, None])[::-1]]


C = np.array([cx, cy])
SX = 1.5                                    # horizontal stretch of the vortex
Y_BAR = 1.22 * asc                          # straight part of the d stem ends here
PHI_A, R_A = np.deg2rad(100), 1.42 * asc - cy  # where the arm joins the spiral (top)
DECAY = 0.30                                # radius factor per turn (tightness)
TURNS = 2.3


def spiral(u, phase=0.0, r_scale=1.0):
    """Clockwise inward spiral, u in [0, 1] from the outer point A to the centre."""
    phi = PHI_A + phase - 2 * np.pi * TURNS * u
    r = R_A * r_scale * DECAY ** (TURNS * u)
    return np.c_[C[0] + SX * r * np.cos(phi), C[1] + r * np.sin(phi)]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


# d arm = stem (up to B) + clockwise arc + straight tangent segment + spiral from A.
# The arc only turns one way, so the arm never changes bend direction.
sp = spiral(np.linspace(0, 1, 900))
A = sp[0]
tA = (sp[1] - sp[0]) / np.linalg.norm(sp[1] - sp[0])
B = np.array([d_x, Y_BAR])
# arc centre O = B + (r, 0); its tangent line in direction tA must pass through A
r1 = -cross(tA, B - A) / (cross(tA, [1.0, 0.0]) + 1)
O = B + [r1, 0]
psi_T = np.arctan2(tA[1], tA[0])
psi = np.linspace(np.pi / 2, psi_T, 120)
arc = O - r1 * np.c_[np.sin(psi), -np.cos(psi)]
assert r1 > 0 and np.dot(A - arc[-1], tA) > 0, (r1, "junction not reachable")
line = arc[-1] + np.linspace(0, 1, 60)[:, None] * (A - arc[-1])
stem = np.c_[np.full(60, d_x), np.linspace(stem_w / 2, Y_BAR, 60)]
d_arm = np.r_[stem[:-1], arc[:-1], line[:-1], sp]
n_s = len(stem) - 1 + len(arc) - 1 + len(line) - 1
w = np.r_[np.full(n_s, stem_w / 2), (stem_w / 2) * (1 - 0.88 * np.linspace(0, 1, len(sp)) ** 0.8)]
p_arm = 2 * C - d_arm                       # 180-degree rotation: the p arm

for arm, col in [(d_arm, D_COL), (p_arm, P_COL)]:
    rib = ribbon(arm, w)
    ax.add_patch(Polygon(rib, closed=True, fc=col, ec="none", zorder=2))
    # stem and curl again above the letters, so the bars read in colour
    m = n_s + 25
    ax.add_patch(Polygon(ribbon(arm[:m], w[:m]), closed=True, fc=col, ec="none", zorder=5))
    # rounded cap at the foot of the bar
    ax.add_patch(plt.Circle(arm[0], stem_w / 2, fc=col, ec="none", zorder=5))

# thin filaments shadowing each arm inside the vortex
for k in range(1, 5):
    u = np.linspace(0.02 * k, 0.75, 500)
    f = spiral(u, phase=-0.16 * k, r_scale=1 + 0.06 * k)
    wf = stem_w * 0.07 * np.sin(np.pi * (u - u[0]) / (u[-1] - u[0])) + 0.2
    for xy, col in [(f, D_COL), (2 * C - f, P_COL)]:
        ax.add_patch(Polygon(ribbon(xy, wf), closed=True, fc=col, ec="none",
                             alpha=0.5 - 0.08 * k, zorder=1))

top = d_arm[:, 1].max() + stem_w / 2      # the p arm reaches xh - top

# ---------------------------------------------------------------- letters
for i in range(len(WORD)):
    ax.add_patch(PathPatch(glyph(i), fc=INK, ec="none", zorder=4))

W = xs[-1] + ext_of(WORD[-1]).x1
m = 0.03 * W
y0, y1 = xh - top - 0.05 * asc, top + 0.05 * asc
ax.set_xlim(-m, W + m)
ax.set_ylim(y0, y1)
fig.set_size_inches(10, 10 * (y1 - y0) / (W + 2 * m))
fig.savefig(out, transparent=True)
print(out)
