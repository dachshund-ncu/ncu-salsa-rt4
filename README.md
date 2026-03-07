# NCU SALSA 💃
**Nicolaus Copernicus University Spectroscopic Autocorrelator Library for Signal Analysis**

NCU SALSA is a high-performance Python library developed for processing of the spectroscopic data from autocorrelator correlators. It provides a specialized pipeline for transforming raw autocorrelation functions (ACF) into calibrated, Doppler-corrected power spectra.



## 🌌 Core Purpose
In digital radio spectroscopy, signals are digitized and quantized (typically 2-level or 3-level). This process introduces non-linear distortions in the correlation. NCU SALSA implements the **Van Vleck Correction** to recover the true analog correlation from quantized samples, alongside precise Earth-rotation and orbital velocity compensations.

## ✨ Key Features

* **Vectorized Van Vleck Reconstruction:** Optimized 2D-vectorized implementations of the Borkowski (A2S) legacy code. It handles both low and high-bias regimes without Python loops.
* **Precision Doppler Engine:**
    * Barycentric velocity correction via `barycorrpy`.
    * Local Standard of Rest (LSR) motion calculation.
    * Automatic coordinate precession to JNOW (FK5).
* **Spectroscopic Rotation:** Complex-domain ACF rotation to align spectral lines with target rest frequencies before Fourier Transformation.
* **Calibration Suite:**
    * $T_{sys}$ (System Temperature) scaling to mK.
    * Automated statistical estimation of digitization bias (Nmax/multiple).



## 🚀 Performance Design
NCU SALSA is engineered to solve the "Python Bottleneck" in radio astronomy data reduction:

* **2D Broadcasting:** By replacing scalar Python loops with NumPy 2D broadcasting, the library processes multiple Base Band Converters (BBC) simultaneously in the C-layer.
* **Serialization Optimization:** The pipeline favors high-speed sequential vectorization over Multiprocessing to avoid the "Pickling Tax" (the overhead of moving large ACF arrays between CPU cores).
* **Horner's Method:** Polynomials for Van Vleck corrections are evaluated using nested multiplications to minimize floating-point operations.

## 🛠 Installation

Manage dependencies and environments efficiently using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/dachshund-ncu/ncu-salsa-rt4.git
cd ncu-salsa-rt4
python3 -m pip install .
```

## 📖 Quick start

```python
from ncu_salsa_rt4 import ScanSet
scan_set = ScanSet(
    archive_filename=tar_bz2_archive, 
    on_off=False, 
    debug=False, 
    use_optimized_methods=False, 
    use_multithreaded_utils=False)
for scan in scan_set.scans:
    final_spectrum = scan.spectr_bbc_final
```

## 🏫 Institutional Context
Developed for the Institute of Astronomy, Nicolaus Copernicus University (Toruń, Poland). This library is a modern Python port and optimization of long-standing radio spectroscopy algorithms used in and single-dish observations.