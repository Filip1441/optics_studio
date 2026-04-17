# Optics Studio 🌌

A professional hybrid optics simulator built in Python and PyQt6. Designed for advanced study of geometric and wave optics, with a focus on diffraction, Fourier processing, and high-precision bench prototyping.

## 🚀 Key Features

- **Hybrid Simulation Engine:** 
  - **Geometric Ray Tracing:** Fast computation for systems with mirrors, lenses, and complex paths.
  - **ASM Wave Engine:** Physically accurate propagation using the **Angular Spectrum Method (ASM)** with zero-padding to eliminate boundary artifacts.
- **Fourier Optics & Spectral Analysis:**
  - Real-time **Fourier Plane** visualization on detectors.
  - High-pass (DC-block) filters for edge enhancement and spatial frequency filtering.
- **Dynamic Optical Axis:** The main optical axis follows the light path through mirrors and refractions. No more manual coordinate calculation.
- **Intelligent Grating Logic:** 
  - **Auto-Footprint Sensing:** Gratings automatically detect incoming beam diameter and regenerate high-density diffraction orders for both 1D and 2D (Cross) gratings.
- **Professional Engineering View:**
  - Sidebar for precise parameter entry (focal length, line density, radius).
  - Interactive rotation handles with snapping.
- **Save/Load System:** Full JSON-based persistence for sharing and archiving your optical setups.

## 🛠️ Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/your-username/optics-studio.git
   cd optics-studio
   ```

2. **Setup environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. **Run the studio:**
   ```bash
   python gui_app.py
   ```

## 🎮 UI Controls

- **Left Click:** Select / Drag component.
- **Blue Handle:** Rotate component (Hold **Ctrl** for 45° snapping).
- **Control Panel:** Real-time updates for focal length, wavelength, grating density, and spectral modes.
- **Middle Click:** Pan the scene.
- **Mouse Wheel:** Zoom.

## 📄 License

MIT
