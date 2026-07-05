# ATHENA
### Adaptive Turbulence correction and Hartmann wavefront rEconstruction with Neural Analysis

> BAH 2026 — Problem 09 | Shack-Hartmann Wavefront Sensor Pipeline

---

## What This Is

A real-time adaptive optics pipeline that takes Shack-Hartmann 
Wavefront Sensor (SH-WFS) frames distorted by atmospheric turbulence 
and reconstructs the wavefront shape, characterizes turbulence 
strength, and computes deformable mirror actuator commands.

---

## Pipeline Blocks

| File | What it does |
|---|---|
| `simulate_data.py` | Generates Kolmogorov-correlated SH-WFS frames |
| `block1_centroid.py` | Detects spot centroids using weighted CoM |
| `block2_slope.py` | Converts spot shifts to wavefront slopes |
| `block3_reconstruct.py` | Zernike modal reconstruction, weighted least squares |
| `block4_turbulence.py` | Computes Fried parameter r0 and coherence time τ0 |
| `block5_actuator.py` | Generates DM actuator stroke map with Gaussian coupling |
| `main.py` | Runs full pipeline + live dashboard |

---

## Outputs

- Reconstructed wavefront phase maps W(x,y) per frame
- Fried parameter r0 (turbulence strength)
- Coherence time τ0 (turbulence speed)
- Deformable mirror actuator maps A(x,y) per frame
- Live animated dashboard

---

## Setup

```bash
pip install numpy scipy matplotlib opencv-python
```

Run simulation first:
```bash
python simulate_data.py
```

Then run full pipeline:
```bash
python main.py
```

---

## Tech Stack

- Python 3.13
- NumPy, SciPy — numerical core
- OpenCV — BMP frame I/O
- Matplotlib — visualization

---

## Team

TKM COLLEGE OF ENGINEERING(TKMCE)
Bharatiya Antariksh Hackathon 2026