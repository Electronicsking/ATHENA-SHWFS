import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import math
import os

# Block 3 — modal wavefront reconstruction
# slopes → Zernike coefficients → phase map W(x,y)
# using Noll-ordered Zernike polynomials on unit circle

DATA    = "data"
OUT     = "output"
N_LENS  = 10
CAMPIX  = 512
N_MODES = 15   # how many Zernike terms to fit

# Noll ordering (n, m) for first 15 modes
NOLL = [
    (0,  0),   # piston
    (1, -1),   # tilt y
    (1,  1),   # tilt x
    (2, -2),   # astig 45
    (2,  0),   # defocus
    (2,  2),   # astig 0
    (3, -3),   # trefoil y
    (3, -1),   # coma y
    (3,  1),   # coma x
    (3,  3),   # trefoil x
    (4, -4),
    (4, -2),
    (4,  0),   # spherical
    (4,  2),
    (4,  4),
]

MODE_NAMES = [
    "piston", "tilt_y", "tilt_x",
    "astig45", "defocus", "astig0",
    "trefoil_y", "coma_y", "coma_x", "trefoil_x",
    "Z11", "Z12", "spherical", "Z14", "Z15"
]

def zernike_R(n, m, rho):
    # radial polynomial R^m_n(rho)
    R = np.zeros_like(rho, dtype=float)
    for s in range((n - abs(m)) // 2 + 1):
        num   = (-1)**s * math.factorial(n - s)
        den   = (math.factorial(s)
               * math.factorial((n + abs(m))//2 - s)
               * math.factorial((n - abs(m))//2 - s))
        R    += (num / den) * rho**(n - 2*s)
    return R

def Z(n, m, rho, theta):
    # full Zernike polynomial
    R = zernike_R(n, abs(m), rho)
    if   m > 0: return R * np.cos( m * theta)
    elif m < 0: return R * np.sin(-m * theta)
    else:       return R

def lenslet_polar_coords():
    # map 10x10 lenslet grid onto unit circle
    cx = N_LENS / 2.0
    cy = N_LENS / 2.0
    pts = []
    for r in range(N_LENS):
        for c in range(N_LENS):
            x  = (c + 0.5 - cx) / (N_LENS / 2.0)
            y  = (r + 0.5 - cy) / (N_LENS / 2.0)
            rh = np.sqrt(x**2 + y**2)
            th = np.arctan2(y, x)
            pts.append((rh, th, x, y))
    return pts

def build_D_matrix(pts):
    # interaction matrix D: (2N_lens, N_modes)
    # D * coefficients = slopes
    # we need D_inv later to go slopes → coefficients
    N  = len(pts)
    M  = N_MODES
    h  = 1e-5   # finite difference step

    Dx = np.zeros((N, M))
    Dy = np.zeros((N, M))

    for j, (n, m) in enumerate(NOLL[:M]):
        for i, (rho, theta, x, y) in enumerate(pts):
            if rho >= 1.0:
                continue

            # dZ/dx numerically
            rpx = np.sqrt((x+h)**2 + y**2)
            rmx = np.sqrt((x-h)**2 + y**2)
            tpx = np.arctan2(y, x+h)
            tmx = np.arctan2(y, x-h)
            Dx[i,j] = (Z(n,m,rpx,tpx) - Z(n,m,rmx,tmx)) / (2*h)

            # dZ/dy numerically
            rpy = np.sqrt(x**2 + (y+h)**2)
            rmy = np.sqrt(x**2 + (y-h)**2)
            tpy = np.arctan2(y+h, x)
            tmy = np.arctan2(y-h, x)
            Dy[i,j] = (Z(n,m,rpy,tpy) - Z(n,m,rmy,tmy)) / (2*h)

    return np.vstack([Dx, Dy])   # (200, 15)

def reconstruct(sx, sy, D):
    # least-squares fit: D * a = s
    s_vec = np.concatenate([sx, sy])
    a, _, _, _ = lstsq(D, s_vec)   # Zernike coefficients

    # evaluate W(x,y) on a 64x64 grid
    G  = 64
    WF = np.zeros((G, G))
    xs = np.linspace(-1, 1, G)
    ys = np.linspace(-1, 1, G)

    for gi in range(G):
        for gj in range(G):
            x  = xs[gj]
            y  = ys[gi]
            rh = np.sqrt(x**2 + y**2)
            if rh > 1.0:
                WF[gi, gj] = np.nan
                continue
            th  = np.arctan2(y, x)
            val = sum(a[k] * Z(n,m,rh,th)
                      for k,(n,m) in enumerate(NOLL[:N_MODES]))
            WF[gi, gj] = val

    return WF, a

def show_reconstruction(WF, coeffs):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Wavefront reconstruction — frame 000",
                 fontweight="bold")

    im = axes[0].imshow(WF, cmap="RdYlBu",
                         origin="upper", interpolation="bilinear")
    axes[0].set_title("W(x,y) phase map")
    plt.colorbar(im, ax=axes[0], label="phase (waves)")

    cols = ["#e05c5c" if c < 0 else "#4a90d9" for c in coeffs]
    axes[1].bar(range(N_MODES), coeffs * 1e9, color=cols, alpha=0.8)
    axes[1].set_xticks(range(N_MODES))
    axes[1].set_xticklabels(MODE_NAMES, rotation=45,
                             ha="right", fontsize=7)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("coefficient (nm)")
    axes[1].set_title("Zernike modal decomposition")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "wavefront_frame000.png"), dpi=130)
    plt.show()

# ── run ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Block 3 — wavefront reconstruction")

    sx = np.load(os.path.join(DATA, "sx_000.npy"))
    sy = np.load(os.path.join(DATA, "sy_000.npy"))

    D_path = os.path.join(DATA, "D_matrix.npy")
    if os.path.exists(D_path):
        print("  loading cached D matrix...")
        D = np.load(D_path)
    else:
        print("  building D matrix (may take ~20s)...")
        pts = lenslet_polar_coords()
        D   = build_D_matrix(pts)
        np.save(D_path, D)
    print(f"  D matrix shape: {D.shape}")

    print("  reconstructing...")
    WF, coeffs = reconstruct(sx, sy, D)

    print("  Zernike coefficients (nm):")
    for i, name in enumerate(MODE_NAMES):
        print(f"    {name:<12} {coeffs[i]*1e9:+.4f} nm")

    wf_vals = WF[~np.isnan(WF)]
    print(f"  WF PtV : {wf_vals.max()-wf_vals.min():.4f}")
    print(f"  WF RMS : {wf_vals.std():.4f}")

    show_reconstruction(WF, coeffs)

    np.save(os.path.join(DATA, "WF_000.npy"),     WF)
    np.save(os.path.join(DATA, "zcoeffs_000.npy"), coeffs)
    print("  saved WF_000.npy, zcoeffs_000.npy")