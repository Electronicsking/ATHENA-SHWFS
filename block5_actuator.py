import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import os, sys, math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from block1_centroid    import load_frame, find_all_centroids
from block2_slope       import get_slopes, load_ref
from block3_reconstruct import (lenslet_polar_coords,
                                 build_D_matrix,
                                 reconstruct,
                                 NOLL, N_MODES)

# Block 5 — DM actuator map generation
# reconstructed wavefront → conjugate → actuator strokes
# accounts for inter-actuator mechanical coupling

DATA       = "data"
OUT        = "output"
N_ACT      = 8        # 8x8 actuator grid
STROKE_LIM = 5.0      # max actuator displacement (micrometers)
COUPLING   = 0.15     # fraction of stroke leaking to neighbours
LAMBDA     = 500e-9   # wavelength for phase→length conversion

def influence_matrix():
    # each actuator pushes itself (1.0) and
    # nudges immediate neighbours (COUPLING)
    # shape: (64, 64)
    N   = N_ACT * N_ACT
    IFM = np.zeros((N, N))
    for i in range(N):
        ri, ci = divmod(i, N_ACT)
        for j in range(N):
            rj, cj = divmod(j, N_ACT)
            dr = abs(ri - rj)
            dc = abs(ci - cj)
            if   dr == 0 and dc == 0: IFM[i, j] = 1.0
            elif dr <= 1 and dc <= 1: IFM[i, j] = COUPLING
    return IFM

def wavefront_to_dm(WF):
    # Step 1 — conjugate: mirror must be opposite of wavefront
    conj = -WF.copy()

    # Step 2 — sample conjugate at 8x8 actuator grid positions
    G    = WF.shape[0]   # 64
    idxs = np.linspace(0, G-1, N_ACT, dtype=int)
    want = np.zeros((N_ACT, N_ACT))
    for i, gi in enumerate(idxs):
        for j, gj in enumerate(idxs):
            v = conj[gi, gj]
            want[i, j] = 0.0 if np.isnan(v) else v

    # convert phase (radians) to physical stroke (micrometers)
    scale = (LAMBDA / (2 * np.pi)) * 1e6
    want_um = want.flatten() * scale

    # Step 3 — coupling correction
    # IFM * strokes = desired_surface  →  strokes = IFM^-1 * desired
    IFM     = influence_matrix()
    strokes, _, _, _ = lstsq(IFM, want_um)

    # Step 4 — clip to stroke limits
    strokes = np.clip(strokes, -STROKE_LIM, STROKE_LIM)

    return strokes.reshape(N_ACT, N_ACT), want

def show_actuator_map(WF, act_map, want, frame_idx=0):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"DM actuator map — frame {frame_idx:03d}",
                 fontweight="bold")

    axes[0].imshow(WF, cmap="RdYlBu",
                   origin="upper", interpolation="bilinear")
    axes[0].set_title("W(x,y) — reconstructed wavefront")

    axes[1].imshow(want, cmap="RdYlBu_r",
                   origin="upper", interpolation="bilinear")
    axes[1].set_title("conjugate at actuator grid")

    im2 = axes[2].imshow(act_map, cmap="coolwarm",
                          origin="upper", interpolation="bilinear")
    axes[2].set_title(f"actuator strokes (µm)\n±{STROKE_LIM}µm limit")
    plt.colorbar(im2, ax=axes[2], label="stroke (µm)")
    for i in range(N_ACT):
        for j in range(N_ACT):
            axes[2].text(j, i, f"{act_map[i,j]:.1f}",
                         ha="center", va="center",
                         fontsize=5, color="black")

    flat = act_map.flatten()
    cols = ["#e05c5c" if v < 0 else "#4a90d9" for v in flat]
    axes[3].bar(range(len(flat)), flat, color=cols, alpha=0.8)
    axes[3].axhline(0,           color="black", lw=0.8)
    axes[3].axhline( STROKE_LIM, color="tomato", ls="--", lw=1)
    axes[3].axhline(-STROKE_LIM, color="tomato", ls="--", lw=1)
    axes[3].set_xlabel("actuator index")
    axes[3].set_ylabel("stroke (µm)")
    axes[3].set_title("all 64 actuator commands")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT,
                f"actuator_frame{frame_idx:03d}.png"), dpi=130)
    plt.show()

# ── run ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Block 5 — actuator map generation")

    D_path = os.path.join(DATA, "D_matrix.npy")
    if os.path.exists(D_path):
        D = np.load(D_path)
        print("  D matrix loaded from cache")
    else:
        print("  building D matrix (~20s)...")
        pts = lenslet_polar_coords()
        D   = build_D_matrix(pts)
        np.save(D_path, D)

    ref          = load_ref()
    fr           = load_frame(0)
    cents, valids = find_all_centroids(fr)
    sx, sy, _    = get_slopes(cents, ref, valids)
    WF, coeffs   = reconstruct(sx, sy, D)

    act_map, want = wavefront_to_dm(WF)
    flat          = act_map.flatten()

    print(f"  actuators   : {len(flat)}")
    print(f"  max push    : +{flat.max():.3f} µm")
    print(f"  max pull    : {flat.min():.3f} µm")
    print(f"  rms stroke  : {np.sqrt(np.mean(flat**2)):.3f} µm")
    clipped = (np.abs(flat) >= STROKE_LIM * 0.99).sum()
    print(f"  at limit    : {clipped} actuators")

    show_actuator_map(WF, act_map, want, frame_idx=0)

    np.save(os.path.join(DATA, "act_map_000.npy"), act_map)
    print("  saved act_map_000.npy")