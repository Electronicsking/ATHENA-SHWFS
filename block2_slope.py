import numpy as np
import matplotlib.pyplot as plt
import os

# Block 2 — wavefront slope calculation
# spot shift in pixels → local wavefront tilt in radians
# slope = shift * pixel_size / focal_length

DATA   = "data"
OUT    = "output"
N_LENS = 10
CAMPIX = 512

# sensor geometry — typical lab AO setup
F_LENS   = 18.0e-3   # lenslet focal length (m)
PIX_SIZE = 5.5e-6    # camera pixel pitch (m)

SUB = CAMPIX // N_LENS

def load_cents(idx):
    c = np.load(os.path.join(DATA, f"cents_{idx:03d}.npy"))
    v = np.load(os.path.join(DATA, f"valids_{idx:03d}.npy"))
    return c, v

def load_ref():
    return np.load(os.path.join(DATA, "ref_positions.npy"))

def get_slopes(cents, ref, valids):
    # pixel displacement of each spot from reference
    dpix = cents - ref               # (100, 2) pixels

    # physical displacement on detector
    dmm  = dpix * PIX_SIZE           # meters

    # wavefront slope = d / focal_length (radians)
    sx = dmm[:, 0] / F_LENS
    sy = dmm[:, 1] / F_LENS

    # zero out bad lenslets
    sx[~valids] = 0.0
    sy[~valids] = 0.0
    return sx, sy, dpix

def to_grid(arr):
    return arr.reshape(N_LENS, N_LENS)

def show_slopes(sx, sy, dpix):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Wavefront slopes — frame 000", fontweight="bold")

    im0 = axes[0].imshow(to_grid(sx), cmap="RdBu", origin="upper")
    axes[0].set_title("sx (rad)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(to_grid(sy), cmap="RdBu", origin="upper")
    axes[1].set_title("sy (rad)")
    plt.colorbar(im1, ax=axes[1])

    # quiver plot — direction of local tilt
    g  = np.arange(N_LENS)
    X, Y = np.meshgrid(g, g)
    axes[2].quiver(X, Y, to_grid(sx), to_grid(sy),
                   scale=5e-3, color="steelblue")
    axes[2].set_title("tilt direction per lenslet")
    axes[2].invert_yaxis()
    axes[2].set_aspect("equal")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "slopes_frame000.png"), dpi=130)
    plt.show()

# ── run ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Block 2 — slope calculation")
    cents, valids = load_cents(0)
    ref           = load_ref()
    sx, sy, dpix  = get_slopes(cents, ref, valids)

    print(f"  sx range : {sx.min():.4e} to {sx.max():.4e} rad")
    print(f"  sy range : {sy.min():.4e} to {sy.max():.4e} rad")
    print(f"  max pixel shift: {np.abs(dpix).max():.2f} px")

    show_slopes(sx, sy, dpix)

    np.save(os.path.join(DATA, "sx_000.npy"), sx)
    np.save(os.path.join(DATA, "sy_000.npy"), sy)
    print("  saved sx_000.npy, sy_000.npy")