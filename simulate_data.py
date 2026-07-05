import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# SH-WFS simulation for BAH2026 Problem 09
# generates fake but physically reasonable spot frames
# camera: 512x512, 10x10 lenslet array

OUTDIR     = "data"
N_FRAMES   = 50
CAM_PIX    = 512
N_LENS     = 10
SPOT_WIDTH = 3.0      # gaussian sigma for each spot
CAM_NOISE  = 5        # detector read noise level
TURB_STR   = 2.5      # how much spots shift due to turbulence
os.makedirs(OUTDIR, exist_ok=True)

SUB = CAM_PIX // N_LENS   # pixels per lenslet = 51

np.random.seed(7)   # reproducible

def ref_grid():
    # ideal spot positions when wavefront is flat
    # center of each subaperture
    pts = []
    for r in range(N_LENS):
        for c in range(N_LENS):
            px = c * SUB + SUB // 2
            py = r * SUB + SUB // 2
            pts.append([px, py])
    return np.array(pts)   # (100, 2)

def turb_shifts(strength):
    # random spot displacements mimicking atmospheric tilt
    # nearby lenslets should shift similarly — smooth field
    raw = np.random.randn(N_LENS, N_LENS, 2) * strength
    out = []
    for r in range(N_LENS):
        for c in range(N_LENS):
            out.append([raw[r, c, 0], raw[r, c, 1]])
    return np.array(out)   # (100, 2)

def draw_spot(img, cx, cy, sigma=SPOT_WIDTH, peak=255):
    # paint a 2D gaussian onto img at position (cx, cy)
    # only touch a small window around the spot center
    x0 = max(0, int(cx) - 12)
    x1 = min(CAM_PIX, int(cx) + 12)
    y0 = max(0, int(cy) - 12)
    y1 = min(CAM_PIX, int(cy) + 12)
    for px in range(x0, x1):
        for py in range(y0, y1):
            r2 = (px - cx)**2 + (py - cy)**2
            img[py, px] += peak * np.exp(-r2 / (2 * sigma**2))

def make_frame(refs, shifts):
    img = np.zeros((CAM_PIX, CAM_PIX), dtype=np.float32)
    for i in range(len(refs)):
        sx = refs[i, 0] + shifts[i, 0]
        sy = refs[i, 1] + shifts[i, 1]
        draw_spot(img, sx, sy)
    # add detector noise
    img += np.random.rand(CAM_PIX, CAM_PIX) * CAM_NOISE
    return np.clip(img, 0, 255).astype(np.uint8)

# ── run ──────────────────────────────────────────────────
refs = ref_grid()
np.save(os.path.join(OUTDIR, "ref_positions.npy"), refs)

print(f"Simulating {N_FRAMES} SH-WFS frames ({N_LENS}x{N_LENS} lenslets)...")

for i in range(N_FRAMES):
    sh = turb_shifts(TURB_STR)
    fr = make_frame(refs, sh)
    cv2.imwrite(os.path.join(OUTDIR, f"frame_{i:03d}.bmp"), fr)
    if i % 10 == 0:
        print(f"  frame {i+1}/{N_FRAMES}")

print("Done.\n")

# quick look at frame 0
sample = cv2.imread(os.path.join(OUTDIR, "frame_000.bmp"), 0)
plt.figure(figsize=(6, 6))
plt.imshow(sample, cmap="hot")
plt.title("Simulated SH-WFS frame 000")
plt.colorbar(label="intensity (ADU)")
os.makedirs("output", exist_ok=True)
plt.savefig("output/sim_preview.png")
plt.show()