import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from block1_centroid import load_frame, find_all_centroids
from block2_slope    import get_slopes, load_ref

# Block 4 — turbulence characterization
# compute Fried parameter r0 and coherence time tau0
# from the full time series of slope measurements

DATA      = "data"
OUT       = "output"
N_FRAMES  = 50
N_LENS    = 10
LAMBDA    = 500e-9     # wavelength (500 nm green)
D_SUB     = 0.05       # lenslet physical diameter (m)
DT        = 0.005      # time between frames (5 ms)

def slopes_timeseries():
    # run blocks 1+2 on every frame to get slopes
    ref = load_ref()
    SX, SY = [], []
    print(f"  processing {N_FRAMES} frames for turbulence stats...")
    for i in range(N_FRAMES):
        fr = load_frame(i)
        c, v = find_all_centroids(fr)
        sx, sy, _ = get_slopes(c, ref, v)
        SX.append(sx)
        SY.append(sy)
        if i % 10 == 0:
            print(f"    frame {i+1}/{N_FRAMES}")
    return np.array(SX), np.array(SY)   # (50, 100)

def fried_r0(SX, SY):
    # variance of slopes → r0 via Kolmogorov theory
    # sigma^2 = 0.162 * (lambda/r0)^2 * (D_sub/r0)^(-1/3)
    sig2 = (np.var(SX) + np.var(SY)) / 2.0
    r0   = LAMBDA * ((0.162 / sig2) * D_SUB**(-1/3))**(3/5)
    return r0, sig2

def coherence_tau0(SX, SY):
    # temporal autocorrelation of mean slope magnitude
    # tau0 = lag where autocorr drops to 1/e
    sig = np.sqrt(SX.mean(axis=1)**2 + SY.mean(axis=1)**2)
    sig = sig - sig.mean()
    N   = len(sig)
    ac  = np.correlate(sig, sig, mode="full")[N-1:]
    ac /= ac[0]

    tau0 = N * DT   # default if never drops below 1/e
    for k in range(1, len(ac)):
        if ac[k] < 1.0/np.e:
            frac = (ac[k-1] - 1/np.e) / (ac[k-1] - ac[k])
            tau0 = (k - 1 + frac) * DT
            break
    return tau0, ac

def show_turbulence(SX, SY, r0, tau0, ac, sig2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Turbulence characterization", fontweight="bold")

    # variance map
    vmap = np.var(SX, axis=0).reshape(N_LENS, N_LENS)
    im   = axes[0].imshow(vmap, cmap="hot", origin="upper")
    axes[0].set_title("slope variance per lenslet")
    plt.colorbar(im, ax=axes[0], label="var (rad²)")

    # autocorrelation
    taxis = np.arange(len(ac)) * DT * 1e3   # ms
    axes[1].plot(taxis, ac, color="royalblue", lw=2)
    axes[1].axhline(1/np.e,      color="tomato",     ls="--", lw=1.5,
                    label=f"1/e = {1/np.e:.3f}")
    axes[1].axvline(tau0 * 1e3,  color="seagreen",   ls="--", lw=1.5,
                    label=f"τ₀ = {tau0*1e3:.2f} ms")
    axes[1].set_xlabel("lag (ms)")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_title("temporal autocorrelation of slopes")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.3, 1.1)

    # text summary
    axes[2].axis("off")
    if   r0 > 0.15: grade, gc = "WEAK",     "seagreen"
    elif r0 > 0.08: grade, gc = "MODERATE", "goldenrod"
    else:           grade, gc = "STRONG",   "tomato"

    txt = (
        f"r0   = {r0*100:.4f} cm\n"
        f"tau0 = {tau0*1e3:.4f} ms\n"
        f"sig2 = {sig2:.3e} rad²\n\n"
        f"turbulence: {grade}"
    )
    axes[2].text(0.1, 0.6, txt,
                 transform=axes[2].transAxes,
                 fontsize=13, fontfamily="monospace",
                 color=gc,
                 bbox=dict(fc="lightyellow", ec="gray",
                           boxstyle="round", alpha=0.85))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "turbulence.png"), dpi=130)
    plt.show()

# ── run ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Block 4 — turbulence characterization")

    SX, SY = slopes_timeseries()
    print(f"  slope array: {SX.shape}")

    r0,  sig2 = fried_r0(SX, SY)
    tau0, ac  = coherence_tau0(SX, SY)

    print(f"  r0   = {r0*100:.4f} cm")
    print(f"  tau0 = {tau0*1e3:.4f} ms")
    print(f"  sig2 = {sig2:.3e} rad²")

    show_turbulence(SX, SY, r0, tau0, ac, sig2)

    np.save(os.path.join(DATA, "r0.npy"),   np.array([r0]))
    np.save(os.path.join(DATA, "tau0.npy"), np.array([tau0]))
    np.save(os.path.join(DATA, "SX.npy"),   SX)
    np.save(os.path.join(DATA, "SY.npy"),   SY)
    print("  saved r0, tau0, SX, SY")