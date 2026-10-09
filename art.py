"""
Generative stipple artwork for SentimentScope.

The hero image is DECORATIVE ART built from your data:
  - the dotted hills are a smoothed distribution of the sentiment scores
  - the circular figure is a radial histogram (teal = positive, orange = negative)
Real charts elsewhere in the app always sit on solid black panels.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

W, H = 1600, 900
INK, SAGE, CREAM, ORANGE, TEAL, SLATE = "#0A2230", "#9CC5BC", "#F6EFC9", "#F2A31B", "#2F8F83", "#8FA3AD"


def _rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def _density(scores, grid, bandwidth=0.14):
    """Smooth 0-1 density of the scores along the -1..+1 axis."""
    if len(scores) == 0:
        return np.full_like(grid, 0.4)
    d = np.exp(-0.5 * ((grid[:, None] - scores[None, :]) / bandwidth) ** 2).sum(axis=1)
    return d / d.max()


def _wiggle(x, rng):
    """Smooth random hills made from a few sine waves."""
    y = sum(rng.uniform(0.3, 1) * np.sin(x / rng.uniform(120, 340) + rng.uniform(0, 6.28)) for _ in range(4))
    return (y - y.min()) / (y.max() - y.min())


def make_hero(scores, pct_pos=0.4, pct_neu=0.2, pct_neg=0.4, seed=7):
    """Return the hero artwork as a base64 JPEG string."""
    rng = np.random.default_rng(seed)
    scores = np.clip(np.asarray(scores, float), -1, 1)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # ---- soft mint background ----
    top, bottom = _rgb("#DCEBE5"), _rgb("#F6F4E2")
    t = np.linspace(0, 1, 64)[:, None, None]
    ax.imshow(top * (1 - t) + bottom * t, extent=[0, W, H, 0], aspect="auto", zorder=0)

    # ---- the circular figure: a radial histogram of sentiment ----
    cx, cy, R = 1160, 285, 165
    n = 7500
    r = R * np.sqrt(rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    px, py = cx + r * np.cos(a), cy + r * np.sin(a)
    light = 1 - ((px - cx) + (py - cy)) / (2 * R) / 2  # brighter dots toward the upper-left
    keep = rng.random(n) < (0.25 + 0.75 * np.clip(light, 0, 1))
    ax.scatter(px[keep], py[keep], s=rng.uniform(2, 7, keep.sum()), c=SAGE, alpha=0.75, linewidths=0, zorder=2)

    bins = 26
    counts, _ = np.histogram(scores, bins=bins, range=(-1, 1)) if len(scores) else (np.ones(bins), None)
    share = counts / max(counts.max(), 1)
    for i in range(bins):
        mid = -1 + (i + 0.5) * 2 / bins
        angle = np.radians(215 - (i + 0.5) / bins * 250)  # a 250-degree arc over the top
        colour = ORANGE if mid < -0.05 else TEAL if mid > 0.05 else SLATE
        length = 14 + 150 * share[i]
        dx, dy = np.cos(angle), -np.sin(angle)
        for step in np.arange(R + 16, R + 16 + length, 7.5):
            for side in (-5, 0, 5):
                bx = cx + dx * step - dy * side
                by = cy + dy * step + dx * side
                ax.scatter(bx, by, s=rng.uniform(5, 11), c=colour, linewidths=0, zorder=3)

    # ---- small white "data card" floating like the inspiration ----
    ax.add_patch(FancyBboxPatch((1428, 300), 150, 118, boxstyle="round,pad=0,rounding_size=14",
                                fc="#000000", alpha=0.10, ec="none", zorder=3))
    ax.add_patch(FancyBboxPatch((1420, 290), 150, 118, boxstyle="round,pad=0,rounding_size=14",
                                fc="white", ec="none", zorder=4))
    for k, (share_k, colour) in enumerate([(pct_pos, TEAL), (pct_neu, SLATE), (pct_neg, ORANGE)]):
        h = 14 + 70 * share_k
        for row in range(int(h // 9)):
            for col in range(2):
                ax.add_patch(plt.Rectangle((1444 + k * 38 + col * 10, 384 - row * 9), 7, 7, fc=colour, ec="none", zorder=5))

    # ---- stippled hills built from the real score distribution ----
    x = np.arange(0, W)
    density = _density(scores, np.linspace(-1, 1, W))
    layers = [(660, 190, "#BBD6CC", 0.6), (725, 240, "#6FA79D", 0.75), (810, 210, "#25606A", 0.85)]
    for i, (base, amp, fill, mix) in enumerate(layers):
        h = mix * np.roll(density, int((i - 1) * 140)) + (1 - mix) * _wiggle(x, rng)
        h = np.convolve(np.pad(h, 60, mode="edge"), np.ones(60) / 60, mode="same")[60:-60]
        ridge = base - amp * h
        ax.fill_between(x, ridge, H, color=fill, zorder=6 + i * 2, linewidth=0)
        m = 32000
        dot_x = rng.uniform(0, W, m)
        depth = rng.random(m) ** 0.55
        dot_y = ridge[dot_x.astype(int)] + depth * (H - ridge[dot_x.astype(int)])
        shade = np.clip(depth * 1.1 + rng.normal(0, 0.08, m), 0, 1)[:, None]
        colours = _rgb(CREAM) * (1 - shade) ** 1.5 + _rgb("#08202D") * (1 - (1 - shade) ** 1.5)
        ax.scatter(dot_x, dot_y, s=rng.uniform(2, 7, m), c=colours, linewidths=0, zorder=7 + i * 2)

    # ---- fade into the dark page below ----
    fade = np.zeros((64, 1, 4))
    fade[..., :3] = _rgb(INK)
    fade[..., 3] = np.linspace(0, 1, 64)[:, None] ** 1.6
    ax.imshow(fade, extent=[0, W, H, H - 220], aspect="auto", zorder=20)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=100, pil_kwargs={"quality": 86})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    demo = np.concatenate([np.random.default_rng(1).normal(0.45, 0.25, 300), np.random.default_rng(2).normal(-0.5, 0.25, 220)])
    open("hero_preview.jpg", "wb").write(base64.b64decode(make_hero(demo, 0.5, 0.15, 0.35)))
    print("Saved hero_preview.jpg")
