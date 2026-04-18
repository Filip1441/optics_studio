import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import logging

logger = logging.getLogger("WaveEngine")

class WaveEngine:
    """
    Advanced Wave Optics Engine using Angular Spectrum Method (ASM).
    Refined for better paraxial focus handling and sampling.
    """
    def __init__(self, res=512, size=0.02):
        self.res = res
        self.size = size 
        self.dx = size / res
        self.wavelength = 532e-9
        self.padding_factor = 2 
        self.setup_grids(res, size)

    def setup_grids(self, res, size):
        self.res = res
        self.size = size
        self.dx = size / res
        x = np.linspace(-size / 2, size / 2, res)
        self.X, self.Y = np.meshgrid(x, x)
        
        pad_res = res * self.padding_factor
        f = np.fft.fftfreq(pad_res, d=self.dx)
        FX, FY = np.meshgrid(f, f)
        self.f_sq = FX**2 + FY**2

    def propagate(self, field, z):
        """ASM Propagation."""
        if abs(z) < 1e-7: return field
        
        k = 2 * np.pi / self.wavelength
        pad = self.res // 2
        
        # 1. Padding
        padded_field = np.pad(field, ((pad, pad), (pad, pad)), mode='constant')
        
        # 2. Transfer Function
        # We cap kz to avoid evanescent aliases
        arg = k**2 - (4 * np.pi**2 * self.f_sq)
        kz = np.sqrt(np.maximum(arg, 0.0) + 0j)
        H = np.exp(1j * kz * z)
        
        # 3. FFT
        F = fft2(padded_field)
        propagated = ifft2(F * H)
        
        return propagated[pad:-pad, pad:-pad]

    def generate_source(self, type="point", r=0.005, z_div=0.01):
        """
        type: 'point' or 'planar'
        r: radius of beam (for planar)
        z_div: virtual distance for divergence (for point)
        """
        r_sq = self.X**2 + self.Y**2
        k = 2 * np.pi / self.wavelength
        
        if type == "point":
            # Diverging wave (spherical)
            z = max(z_div, 1e-4)
            # Phase: exp(i k r^2 / 2z)
            field = (1.0 / z) * np.exp(1j * k * z) * np.exp(1j * k * r_sq / (2 * z))
            # Aperture limit
            field *= np.exp(-(r_sq / (0.9 * self.size / 2)**2)**10)
        else:
            # Planar Super-Gaussian
            field = np.exp(-(r_sq / r**2)**10).astype(complex)
            
        return field

    def apply_lens(self, field, f, r):
        r_sq = self.X**2 + self.Y**2
        k = 2 * np.pi / self.wavelength
        mask = (np.sqrt(r_sq) <= r).astype(float)
        phase = np.exp(-1j * k / (2 * f) * r_sq)
        return field * mask * phase

    def apply_grating(self, field, lines_mm, pattern="Linear Cosine"):
        d = 1e-3 / lines_mm
        if "Zebra" in pattern:
            # Binary mask (Square wave)
            mask_x = (np.cos(2 * np.pi * self.X / d) > 0).astype(float)
            trans = mask_x
            if "Crossed" in pattern:
                mask_y = (np.cos(2 * np.pi * self.Y / d) > 0).astype(float)
                trans *= mask_y
        else:
            # Cosine mask (Sinusoidal)
            trans = 0.5 * (1 + np.cos(2 * np.pi * self.X / d))
            if "Crossed" in pattern: 
                trans *= 0.5 * (1 + np.cos(2 * np.pi * self.Y / d))
        return field * trans

    def apply_aperture(self, field, r):
        mask = (np.sqrt(self.X**2 + self.Y**2) <= r).astype(float)
        return field * mask

    def apply_high_pass(self, field, r):
        mask = (np.sqrt(self.X**2 + self.Y**2) >= r).astype(float)
        return field * mask

    def calculate_on_axis(self, components, axis_path):
        if not axis_path or not components: return None
        
        segments = []
        for i in range(len(axis_path)-1):
            p1, p2 = axis_path[i], axis_path[i+1]
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            segments.append({'start': sum(s['dist'] for s in segments), 'dist': dist, 'p1': p1, 'p2': p2})
            
        src = next((c for c in components if "Source" in c.__class__.__name__), None)
        if not src: return None
        
        self.wavelength = src.params.get('wavelength', 532.0) * 1e-9
        src_type = "planar" if "Beam" in src.__class__.__name__ else "point"
        src_r = src.params.get('width', 0.1) / 2.0
        
        # For Point Source, we simulate a small divergence initially
        # A smaller z_div means a steeper spherical wave
        field = self.generate_source(type=src_type, r=src_r, z_div=1e-3)
        
        visit_list = []
        for comp in components:
            if "Source" in comp.__class__.__name__: continue
            c_pt = np.array([comp.x, comp.y])
            for seg in segments:
                p1 = np.array(seg['p1'])
                v = np.array(seg['p2']) - p1
                w = c_pt - p1
                t = np.dot(w, v) / np.dot(v, v)
                # Check if component is on this segment
                if -0.01 <= t <= 1.01:
                    proj = p1 + t * v
                    dist = np.linalg.norm(c_pt - proj)
                    if dist < 0.05:
                        visit_list.append({'z': seg['start'] + t * seg['dist'], 'comp': comp})
                        break
        
        visit_list.sort(key=lambda x: x['z'])
        logger.info(f"Starting propagation for {len(visit_list)} components")
        
        curr_z = 0
        for visit in visit_list:
            z_step = max(0, visit['z'] - curr_z)
            if z_step > 0:
                logger.info(f"Propagating {z_step:.3f}m to {visit['comp'].name}")
                field = self.propagate(field, z_step)
            curr_z = visit['z']
            
            p = visit['comp'].params
            c_name = visit['comp'].__class__.__name__
            if c_name == "Lens": 
                field = self.apply_lens(field, p.get('f', 0.5), p.get('r', 0.1))
            elif c_name == "Grating": 
                field = self.apply_grating(field, p.get('line_density', 300), pattern=p.get('pattern', 'Linear'))
            elif c_name == "Aperture": 
                field = self.apply_aperture(field, p.get('r', 0.05))
            elif c_name == "HighPassFilter": 
                field = self.apply_high_pass(field, p.get('r', 0.01))
            elif c_name == "Detector": 
                return field
                
        return field
