import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Block 1 — centroid detection
# for each lenslet subaperture find the spot center
# using intensity-weighted center of mass (standard in AO)

DATA   = "data"
OUT    = "output"
N_LENS = 10
CAMPIX = 512
THRESH = 20    # below this = noise, ignore

SUB = CAMPIX // N_LENS

def load_frame(idx):
    p = os.path.join(DATA, f"frame_{idx:03d}.bmp")
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img

def subaperture(frame, row, col):
    # crop lenslet region from full frame
    y0, y1 = row * SUB, (row + 1) * SUB
    x0, x1 = col * SUB, (col + 1) * SUB
    return frame[y0:y1, x0:x1]

def spot_centroid(sub, row, col):
    # intensity-weighted CoM inside one subaperture
    # returns position in full-frame pixel coords
    patch = sub.astype(np.float32)
    patch[patch < THRESH] = 0.0

    I_tot = patch.sum()
    if I_tot < 1e-6:
        # no spot found — fall back to subaperture center
        return (col * SUB + SUB // 2,
                row * SUB + SUB // 2, False)

    xs = np.arange(SUB)
    ys = np.arange(SUB)

    # local centroid
    lx = (patch * xs[np.newaxis, :]).sum() / I_tot
    ly = (patch * ys[:, np.newaxis]).sum() / I_tot

    # convert to full frame coords
    gx = col * SUB + lx
    gy = row * SUB + ly
    return gx, gy, True

def find_all_centroids(frame):
    cents  = []
    valids = []
    for r in range(N_LENS):
        for c in range(N_LENS):
            sub = subaperture(frame, r, c)
            gx, gy, ok = spot_centroid(sub, r, c)
            cents.append([gx, gy])
            valids.append(ok)
    return np.array(cents), np.array(valids)

def show_centroids(frame, cents, valids):
    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for i, (cx, cy) in enumerate(cents):
        col = (0, 255, 0) if valids[i] else (0, 0, 255)
        cv2.circle(rgb, (int(cx), int(cy)), 3, col, -1)
    plt.figure(figsize=(7, 7))
    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.title("Centroid detection — green=valid, red=fallback")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "centroids_frame000.png"))
    plt.show()

# ── run ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Block 1 — centroid detection")
    frame = load_frame(0)
    cents, valids = find_all_centroids(frame)

    n_ok  = valids.sum()
    print(f"  {n_ok}/{len(cents)} valid spots found")
    print(f"  first 3 centroids: {cents[:3]}")

    show_centroids(frame, cents, valids)

    np.save(os.path.join(DATA, "cents_000.npy"),  cents)
    np.save(os.path.join(DATA, "valids_000.npy"), valids)
    print("  saved cents_000.npy, valids_000.npy")