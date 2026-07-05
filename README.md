**\# ATHENA**  
**\#\#\# Adaptive Turbulence correction and Hartmann wavefront rEconstruction with Neural Analysis**

\> \*\*BAH 2026 — Problem 09\*\* | Shack-Hartmann Wavefront Sensor Pipeline | TKM COLLEGE OF ENGINEERING


\---

**\#\# What is ATHENA?**

A real-time adaptive optics pipeline that takes Shack-Hartmann Wavefront Sensor (SH-WFS) frames distorted by atmospheric turbulence and:  
\- Reconstructs the wavefront shape frame by frame  
\- Characterizes turbulence strength (Fried parameter r0) and speed (coherence time τ0)  
\- Computes deformable mirror actuator commands to correct the distortion in real time

\---

**\#\# Pipeline Architecture**

Atmosphere (turbulence)  
 ↓  
 SH-WFS Camera (.bmp frames)  
 ↓  
 Block 1 — Centroid Detection (weighted center of mass per lenslet)  
 ↓  
 Block 2 — Slope Calculation (spot shift → wavefront tilt in radians)  
 ↓  
 Block 3 — Wavefront Reconstruction (15-mode Zernike fit, weighted least squares)  
 ↓  
 Block 4 — Turbulence Stats (Fried parameter r0, coherence time τ0)  
 ↓  
 Block 5 — Actuator Map (conjugate wavefront, Gaussian coupling)  
 ↓  
 Deformable Mirror Correction (\< 10ms real-time loop)

\---

\#\# Project Structure

| File | Description |  
|---|---|  
| \`simulate\_data.py\` | Generates Kolmogorov-correlated SH-WFS BMP frames |  
| \`block1\_centroid.py\` | Detects spot centroids using intensity-weighted CoM |  
| \`block2\_slope.py\` | Converts spot pixel shifts to wavefront slopes (rad) |  
| \`block3\_reconstruct.py\` | Zernike modal reconstruction via weighted least squares |  
| \`block4\_turbulence.py\` | Computes r0 and τ0 from slope time series |  
| \`block5\_actuator.py\` | Generates DM actuator stroke map with Gaussian coupling |  
| \`main.py\` | Runs full pipeline end to end with live dashboard |

\---

\#\# Key Results

| Parameter | Value |  
|---|---|  
| Lenslet array | 10×10 \= 100 subapertures |  
| Camera resolution | 512×512 pixels |  
| Pixel size | 5.5 µm |  
| Lenslet focal length | 18 mm |  
| Wavelength | 500 nm |  
| Zernike modes | 15 (Noll ordering) |  
| Actuators | 8×8 \= 64 (Gaussian coupling 15%) |  
| Pipeline speed | \~1.2 ms per frame |  
| Throughput | \~830 fps |  
| Fried parameter r0 | 0.17 cm |  
| Coherence time τ0 | 2.59 ms |  
| Max actuator stroke | ±5 µm |

\---

\#\# Outputs

\- Reconstructed wavefront phase maps \*\*W(xi, yi)\*\* per SH-WFS frame  
\- Turbulence strength — \*\*Fried parameter r0\*\*  
\- Turbulence speed — \*\*Coherence time τ0\*\*  
\- Deformable mirror commands — \*\*Actuator maps A(xi, yi)\*\* per frame  
\- Live animated dashboard showing all outputs in real time

\---

\#\# Setup

Install dependencies:  
\`\`\`bash  
pip install numpy scipy matplotlib opencv-python  
\`\`\`

Generate simulated SH-WFS frames:  
\`\`\`bash  
python simulate\_data.py  
\`\`\`

Run the full pipeline:  
\`\`\`bash  
python main.py  
\`\`\`

Run live animation demo:  
\`\`\`bash  
python demo\_live.py  
\`\`\`

\---

\#\# Tech Stack

| Layer | Tool |  
|---|---|  
| Language | Python 3.13 |  
| Numerical core | NumPy, SciPy |  
| Image processing | OpenCV |  
| Visualization | Matplotlib |  
| Algorithm | Zernike modal reconstruction, Kolmogorov turbulence |

\---

\#\# About

\*\*ATHENA\*\* was built for the \*\*Bharatiya Antariksh Hackathon 2026 (BAH 2026)\*\*, Problem Statement 09 —  
\*Developing and optimizing algorithms for Wavefront reconstruction and turbulence characterization using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data.\*

\*\*Team:\*\* TKM COLLEGE OF ENGINEERING, Kerala  
