import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from components import PointSource, BeamSource, Grating, Mirror, Detector

def run_axial_analysis(system):
    from optics_engine import Ray

    src = next((c for c in system.components if isinstance(c, (PointSource, BeamSource))), None)
    if not src: 
        return "ANALYSIS ERROR: No Light Source found in the system.", None
    
    # Text report setup
    nx, ny = np.cos(np.radians(src.angle)), np.sin(np.radians(src.angle))
    axis_ray = Ray([src.x, src.y, 0.0], [nx, ny, 0.0])
    
    beam_width = src.params.get("width", 0.01) if isinstance(src, BeamSource) else 0.001
    curr_pos = np.array([src.x, src.y, 0.0])
    curr_dir = np.array([nx, ny, 0.0])
    
    wl_nm = src.params.get('wavelength', 532.0)
    wl_m = wl_nm * 1e-9
    k = 2 * np.pi / wl_m
    
    lines = []
    lines.append("="*60)
    lines.append("        OPTICAL SYSTEM ANALYSIS (AXIAL REPORT)")
    lines.append("="*60)
    lines.append(f"START: {src.name}")
    lines.append(f"  - Initial Beam Diameter: {beam_width*1000:.2f} mm")
    lines.append(f"  - Wavelength: {wl_nm} nm")
    
    # Source Direction and Divergence info
    fan_deg = np.degrees(src.params.get('angle_range', 0.1)) if isinstance(src, PointSource) else 0.0
    lines.append(f"  - Emission Direction: {src.angle:.2f} deg (Vector: [{nx:.3f}, {ny:.3f}])")
    if isinstance(src, PointSource):
        lines.append(f"  - Divergence (Fan Angle): {fan_deg:.2f} deg")
    
    # Initial offsets for Wave Engine
    det_offset_x = 0.0
    det_offset_y = 0.0
    
    lines.append("-" * 40)

    # Find first component after source to optimize initial propagation
    first_comp = None
    first_hit_dist = 0.0
    for comp in system.components:
        if comp == src: continue
        res = axis_ray.propagate_to_plane(np.array([comp.x, comp.y, 0.0]), np.array([np.cos(np.radians(comp.angle)), np.sin(np.radians(comp.angle)), 0.0]))
        if res:
            first_hit_dist = np.linalg.norm(res[0] - np.array([src.x, src.y, 0.0]))
            first_comp = comp
            break

    # Wave Engine Setup: Adaptive Window (Back to 2048 for performance)
    total_z = 0.0
    det = next((c for c in reversed(system.components) if isinstance(c, Detector)), None)
    det_r = det.params.get("r", 0.05) if det else 0.05
    window_limit = max(det_r * 2.5, beam_width * 1.5, 0.05)
    window_size = window_limit
    
    N = 2048
    dx = window_size / N
    x = np.linspace(-window_size/2, window_size/2, N, endpoint=False)
    y = np.linspace(-window_size/2, window_size/2, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    fx = fftfreq(N, dx)
    fy = fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fy)
    
    # Track if detector was actually hit
    detector_was_hit = False
    det_offset_x, det_offset_y = 0.0, 0.0

    # Initialize Field U directly at the FIRST component to avoid aliasing
    if first_comp and first_hit_dist > 0.01:
        # Analytical propagation of point source to first plane
        r_sq = X**2 + Y**2
        if isinstance(src, BeamSource):
            U = np.zeros((N, N), dtype=complex)
            U[r_sq <= (beam_width/2)**2] = 1.0 + 0j
        else:
            fan_angle = src.params.get("angle_range", 0.1)
            w_z = first_hit_dist * np.tan(fan_angle/2)
            U = np.exp(-r_sq / (w_z**2)) * np.exp(1j * k * r_sq / (2 * first_hit_dist))
        
        curr_pos = np.array([first_comp.x, first_comp.y, 0.0])
        processed = [src]
    else:
        U = np.zeros((N, N), dtype=complex)
        if isinstance(src, BeamSource):
            U[X**2 + Y**2 <= (beam_width/2)**2] = 1.0 + 0j
        else:
            sigma = max(0.0005, 3 * dx)
            U = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        curr_pos = np.array([src.x, src.y, 0.0])
        processed = [src]

    for _ in range(15):
        closest_hit, closest_t, hitted_comp = None, float('inf'), None
        for comp in system.components:
            if comp in processed: continue
            angle_rad = np.radians(comp.angle)
            normal = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            res = axis_ray.propagate_to_plane(np.array([comp.x, comp.y, 0.0]), normal)
            if res:
                hit_pt, t = res
                if 1e-5 < t < closest_t:
                    closest_t, closest_hit, hitted_comp = t, hit_pt, comp
        
        if not hitted_comp: break
        
        # Track cumulative distance for point source expansion
        dist = np.linalg.norm(closest_hit - curr_pos)
        total_z += dist
        
        # Physical check: Distance of axial ray from component center
        comp_pos_3d = np.array([hitted_comp.x, hitted_comp.y, 0.0])
        hit_dist_from_center = np.linalg.norm(closest_hit - comp_pos_3d)
        
        # Update beam width for point source based on total distance
        if isinstance(src, PointSource):
            fan_angle = src.params.get("angle_range", 0.1)
            beam_width = 2 * total_z * np.tan(fan_angle/2)
        
        width_in = beam_width
        
        r_comp = hitted_comp.params.get("r", 0.1)
        if hit_dist_from_center > r_comp:
            processed.append(hitted_comp)
            continue
        
        # Propagation (ASM)
        if dist > 1e-6:
            term = 1.0 - (wl_m * FX)**2 - (wl_m * FY)**2
            term = np.clip(term, 0.0, None)
            H = np.exp(1j * k * dist * np.sqrt(term))
            U = ifft2(fft2(U) * H)
        
        if isinstance(src, PointSource) and len(processed) == 1:
            beam_width = 2 * dist * np.tan(src.params.get("angle_range", 0.1)/2)
        
        width_in = beam_width
        comp_type = hitted_comp.__class__.__name__
        
        if comp_type == "HighPassFilter":
            width_out = width_in
            F_U = fftshift(fft2(U))
            cutoff = hitted_comp.params.get("cutoff_freq", 5000)
            F_U[(fftshift(FX)**2 + fftshift(FY)**2) < cutoff**2] = 0.0
            U = ifft2(ifftshift(F_U))
        elif comp_type == "Lens":
            f = hitted_comp.params.get("f", 0.5)
            # Anti-aliased phase mask
            r_sq = X**2 + Y**2
            max_df = 0.5 / dx
            local_df = (k * np.sqrt(r_sq) / f) / (2 * np.pi)
            aa_mask = np.clip(1.5 - (local_df / max_df), 0, 1)
            U = U * np.exp(-1j * (k / (2*f)) * r_sq * aa_mask)
            U = U * (r_sq <= r_comp**2)
            width_out = min(width_in, 2*r_comp)
        else:
            U = U * (X**2 + Y**2 <= r_comp**2)
            width_out = min(width_in, 2*r_comp)
            if comp_type == "Grating":
                d = 1e-3 / hitted_comp.params.get("line_density", 300)
                U = U * 0.5 * (1 + np.cos(2 * np.pi * X / d))

        if isinstance(hitted_comp, Grating):
            lines_mm = hitted_comp.params.get("line_density", 300)
            d = 1e-3 / lines_mm
            n_vec = np.array([np.cos(np.radians(hitted_comp.angle)), np.sin(np.radians(hitted_comp.angle)), 0.0])
            v_y = np.array([-n_vec[1], n_vec[0], 0.0]) 
            sin_i = np.dot(curr_dir, v_y)
            n_orders = hitted_comp.params.get("n_orders", 2)
            lines.append(f"  - Multi-beam branching (Order Angles):")
            for m in range(-n_orders, n_orders + 1):
                sin_m = sin_i + m * (wl_m / d)
                theta_m = np.degrees(np.arcsin(sin_m)) if abs(sin_m) <= 1.0 else None
                lines.append(f"    * m={m:2d}: Angle = {f'{theta_m:6.2f} deg' if theta_m is not None else 'EVANESCENT'}")

        lines.append(f"NEXT: {hitted_comp.name}")
        lines.append(f"  - Distance from prev: {dist*1000:.1f} mm")
        # Ensure we show a sensible diameter even if it's the start
        disp_width_in = width_in if width_in > 0 else (src.params.get('width', 0.01) if isinstance(src, BeamSource) else 0.0)
        lines.append(f"  - Incident Diameter: {disp_width_in*1000:.2f} mm")
        lines.append(f"  - Output Diameter:   {width_out*1000:.2f} mm")
        
        for p_key, p_val in hitted_comp.params.items():
            if p_key not in ['r', 'wavelength', 'n_rays', 'angle_range', 'width', 'pattern']:
                unit = " m" if p_key == 'f' else (" lines/mm" if p_key == 'line_density' else "")
                lines.append(f"  - Parameter '{p_key}': {p_val}{unit}")

        lines.append("-" * 40)
        curr_pos, beam_width = closest_hit, width_out
        processed.append(hitted_comp)
        axis_ray.origin = closest_hit
        
        if isinstance(hitted_comp, Mirror):
            angle_rad = np.radians(hitted_comp.angle)
            n_3d = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            curr_dir = curr_dir - 2 * np.dot(curr_dir, n_3d) * n_3d
            axis_ray.direction = curr_dir
        elif isinstance(hitted_comp, Detector):
            det_offset_x = hitted_comp.x - closest_hit[0]
            det_offset_y = hitted_comp.y - closest_hit[1]
            detector_was_hit = True
            break
            
    lines.append("ANALYSIS COMPLETE.")
    lines.append(f"  - Wave matrix physical size: {window_size*1000:.1f}x{window_size*1000:.1f} mm")
    lines.append("="*60)
    
    image_data_full = np.abs(U)**2
    
    # Final Cropping to exactly 2048x2048
    if detector_was_hit:
        r_target = det.params.get("r", 0.05)
        pix_per_mm = N / window_size
        crop_px = int(r_target * pix_per_mm)
        shift_px_x, shift_px_y = int(det_offset_x * pix_per_mm), int(det_offset_y * pix_per_mm)
        cx, cy = (N // 2) - shift_px_x, (N // 2) - shift_px_y
        y_s, y_e = max(0, cy - crop_px), min(N, cy + crop_px)
        x_s, x_e = max(0, cx - crop_px), min(N, cx + crop_px)
        
        cropped = image_data_full[y_s:y_e, x_s:x_e]
        
        if cropped.size > 1:
            h, w = cropped.shape
            y_new = np.linspace(0, h-1, N)
            x_new = np.linspace(0, w-1, N)
            y_f, x_f = np.floor(y_new).astype(int), np.floor(x_new).astype(int)
            y_c, x_c = np.minimum(y_f + 1, h - 1), np.minimum(x_f + 1, w - 1)
            y_diff, x_diff = (y_new - y_f)[:, None], (x_new - x_f)[None, :]
            
            I00, I10 = cropped[y_f[:, None], x_f[None, :]], cropped[y_c[:, None], x_f[None, :]]
            I01, I11 = cropped[y_f[:, None], x_c[None, :]], cropped[y_c[:, None], x_c[None, :]]
            
            image_data = I00 * (1 - y_diff) * (1 - x_diff) + \
                         I10 * y_diff * (1 - x_diff) + \
                         I01 * (1 - y_diff) * x_diff + \
                         I11 * y_diff * x_diff
        else:
            image_data = np.zeros((N, N))
    else:
        # Detector not hit -> Show dark field
        image_data = np.zeros((N, N))

    v_max = np.max(image_data)
    if v_max > 0: image_data = image_data / v_max
    return "\n".join(lines), image_data