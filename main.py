import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time, os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from block1_centroid    import load_frame, find_all_centroids
from block2_slope       import get_slopes, load_ref
from block3_reconstruct import (lenslet_polar_coords,
                                 build_D_matrix, reconstruct)
from block4_turbulence  import (slopes_timeseries,
                                 fried_r0, coherence_tau0)
from block5_actuator    import (influence_matrix,
                                 wavefront_to_dm)

DATA     = "data"
OUT      = "output"
N_FRAMES = 50

os.makedirs(OUT, exist_ok=True)

print("=" * 52)
print("  SH-WFS Pipeline  |  BAH 2026  |  Problem 09")
print("=" * 52)

# ── setup ─────────────────────────────────────────────────
t0  = time.time()
ref = load_ref()

D_path = os.path.join(DATA, "D_matrix.npy")
if os.path.exists(D_path):
    D = np.load(D_path)
    print("[setup] D matrix loaded from cache")
else:
    print("[setup] building D matrix (~20s)...")
    pts = lenslet_polar_coords()
    D   = build_D_matrix(pts)
    np.save(D_path, D)
print(f"[setup] D shape: {D.shape}")

# ── turbulence stats ──────────────────────────────────────
print("\n[turb]  computing r0 and tau0...")
SX, SY = slopes_timeseries()
r0, sig2 = fried_r0(SX, SY)
tau0, ac = coherence_tau0(SX, SY)
print(f"[turb]  r0   = {r0*100:.4f} cm")
print(f"[turb]  tau0 = {tau0*1e3:.4f} ms")

# ── per-frame pipeline ────────────────────────────────────
print(f"\n[pipe]  running {N_FRAMES} frames...")

WF_all  = []
ACT_all = []
ZC_all  = []
t_per   = []

for i in range(N_FRAMES):
    tf = time.time()

    fr            = load_frame(i)
    cents, valids = find_all_centroids(fr)
    sx, sy, _     = get_slopes(cents, ref, valids)
    WF, zc        = reconstruct(sx, sy, D)
    amap, _       = wavefront_to_dm(WF)

    WF_all.append(WF)
    ACT_all.append(amap)
    ZC_all.append(zc)
    t_per.append(time.time() - tf)

    if i % 10 == 0:
        print(f"  frame {i+1:02d}/{N_FRAMES}"
              f"  {(time.time()-tf)*1e3:.1f} ms")

avg_ms = np.mean(t_per) * 1e3
fps    = 1.0 / np.mean(t_per)
print(f"\n[pipe]  avg {avg_ms:.2f} ms/frame  |  {fps:.0f} fps")
print(f"[pipe]  total: {time.time()-t0:.1f} s")

# ── final dashboard ───────────────────────────────────────
print("\n[dash]  building dashboard...")

fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.5, wspace=0.38)

BG = "#161b22"
TC = "#c9d1d9"
AC = "#58a6ff"

def ax_style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=AC, fontsize=9,
                 fontweight="bold", pad=5)
    ax.tick_params(colors=TC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

# title row
ax0 = fig.add_subplot(gs[0, :])
ax0.axis("off")
ax0.set_facecolor("#0d1117")
ax0.text(0.5, 0.70,
         "SHACK-HARTMANN WAVEFRONT SENSOR PIPELINE",
         ha="center", va="center",
         color=AC, fontsize=17, fontweight="bold",
         fontfamily="monospace",
         transform=ax0.transAxes)
ax0.text(0.5, 0.25,
         "Bharatiya Antariksh Hackathon 2026   |   Problem 09",
         ha="center", va="center",
         color=TC, fontsize=10,
         fontfamily="monospace",
         transform=ax0.transAxes)

# raw frame
ax1 = fig.add_subplot(gs[1, 0])
ax1.imshow(load_frame(0), cmap="hot", origin="upper")
ax_style(ax1, "raw SH-WFS frame")

# wavefront map
ax2  = fig.add_subplot(gs[1, 1])
im2  = ax2.imshow(WF_all[0], cmap="RdYlBu",
                   origin="upper", interpolation="bilinear")
plt.colorbar(im2, ax=ax2).ax.tick_params(colors=TC, labelsize=6)
ax_style(ax2, "W(x,y) phase map")

# actuator map
ax3  = fig.add_subplot(gs[1, 2])
im3  = ax3.imshow(ACT_all[0], cmap="coolwarm",
                   origin="upper", interpolation="bilinear")
plt.colorbar(im3, ax=ax3).ax.tick_params(colors=TC, labelsize=6)
ax_style(ax3, "DM actuator strokes (µm)")

# zernike bar
ax4  = fig.add_subplot(gs[1, 3])
zc   = ZC_all[0]
cols = ["#e05c5c" if v < 0 else "#4ecdc4" for v in zc]
ax4.bar(range(len(zc)), zc * 1e9, color=cols, alpha=0.85)
ax4.axhline(0, color=TC, lw=0.8)
ax4.set_xlabel("Zernike mode", color=TC, fontsize=7)
ax4.set_ylabel("coeff (nm)",   color=TC, fontsize=7)
ax_style(ax4, "Zernike decomposition")

# WF RMS over time
ax5   = fig.add_subplot(gs[2, 0])
rms_t = [np.nanstd(w) for w in WF_all]
ax5.plot(rms_t, color=AC, lw=1.5)
ax5.fill_between(range(len(rms_t)), rms_t, alpha=0.18, color=AC)
ax5.set_xlabel("frame", color=TC, fontsize=7)
ax5.set_ylabel("RMS",   color=TC, fontsize=7)
ax_style(ax5, "wavefront RMS over time")

# autocorrelation
ax6   = fig.add_subplot(gs[2, 1])
taxis = np.arange(len(ac)) * 5   # ms
ax6.plot(taxis, ac, color=AC, lw=1.5)
ax6.axhline(1/np.e,       color="#e05c5c", ls="--", lw=1.2)
ax6.axvline(tau0 * 1e3,   color="#4ecdc4", ls="--", lw=1.2)
ax6.set_xlabel("lag (ms)", color=TC, fontsize=7)
ax6.set_ylabel("autocorr", color=TC, fontsize=7)
ax_style(ax6, "temporal autocorrelation")

# actuator RMS over time
ax7     = fig.add_subplot(gs[2, 2])
act_rms = [np.sqrt(np.mean(a**2)) for a in ACT_all]
ax7.plot(act_rms, color="#f0c040", lw=1.5)
ax7.fill_between(range(len(act_rms)), act_rms,
                 alpha=0.18, color="#f0c040")
ax7.set_xlabel("frame",       color=TC, fontsize=7)
ax7.set_ylabel("RMS (µm)",    color=TC, fontsize=7)
ax_style(ax7, "actuator RMS stroke over time")

# summary
ax8 = fig.add_subplot(gs[2, 3])
ax8.axis("off")
ax8.set_facecolor(BG)
if   r0 > 0.15: grade = "WEAK"
elif r0 > 0.08: grade = "MODERATE"
else:           grade = "STRONG"

summary = (
    f"r0      {r0*100:.3f} cm\n"
    f"tau0    {tau0*1e3:.3f} ms\n"
    f"turb    {grade}\n\n"
    f"frames  {N_FRAMES}\n"
    f"speed   {avg_ms:.2f} ms/frame\n"
    f"fps     {fps:.0f}\n\n"
    f"acts    {8*8} (8x8)\n"
    f"coupling 15%\n"
    f"modes   15 Zernike\n"
)
ax8.text(0.08, 0.93, summary,
         transform=ax8.transAxes,
         color=TC, fontsize=9,
         fontfamily="monospace",
         verticalalignment="top",
         bbox=dict(fc="#0d1117", ec="#30363d",
                   boxstyle="round", alpha=0.9))

plt.savefig(os.path.join(OUT, "dashboard.png"),
            dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()

print("\n" + "=" * 52)
print("  pipeline complete")
print(f"  outputs in: {OUT}/")
print("=" * 52)