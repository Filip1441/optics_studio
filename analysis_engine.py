import numpy as np
from components import PointSource, BeamSource, Grating, Mirror, Detector

def run_axial_analysis(system):
    """
    Separated analysis engine. 
    Traces the optical axis and returns a structured text report.
    """
    # Local import to avoid circular dependency with Ray
    from optics_engine import Ray

    src = next((c for c in system.components if isinstance(c, (PointSource, BeamSource))), None)
    if not src: 
        return "ANALYSIS ERROR: No Light Source found in the system."
    
    nx, ny = np.cos(np.radians(src.angle)), np.sin(np.radians(src.angle))
    axis_ray = Ray([src.x, src.y, 0.0], [nx, ny, 0.0])
    
    # Incident beam width
    beam_width = src.params.get("width", 0.01) if isinstance(src, BeamSource) else 0.0
    curr_pos = np.array([src.x, src.y, 0.0])
    curr_dir = np.array([nx, ny, 0.0])
    
    lines = []
    lines.append("="*60)
    lines.append("        OPTICAL SYSTEM ANALYSIS (AXIAL REPORT)")
    lines.append("="*60)
    lines.append(f"START: {src.name}")
    lines.append(f"  - Initial Beam Diameter: {beam_width*1000:.2f} mm")
    lines.append(f"  - Wavelength: {src.params.get('wavelength', 532.0)} nm")
    lines.append("-" * 40)

    processed = [src]
    for _ in range(15):
        closest_hit = None
        closest_t = float('inf')
        hitted_comp = None
        
        for comp in system.components:
            if comp in processed: continue
            angle_rad = np.radians(comp.angle)
            normal = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            res = axis_ray.propagate_to_plane(np.array([comp.x, comp.y]), normal)
            if res:
                hit_pt, t = res
                if t < closest_t and t > 1e-5:
                    closest_t = t
                    closest_hit = hit_pt
                    hitted_comp = comp
        
        if not hitted_comp: break
        
        dist = np.linalg.norm(closest_hit - curr_pos)
        # Geometric expansion for point source in first segment
        if isinstance(src, PointSource) and len(processed) == 1:
            beam_width = 2 * dist * np.tan(src.params.get("angle_range", 0.1)/2)
        
        width_in = beam_width
        r_comp = hitted_comp.params.get("r", 0.05)
        
        is_hp = hitted_comp.__class__.__name__ == "HighPassFilter"
        if is_hp:
            width_out = width_in # Diameter stays same
            clipping_info = f"Blocks central rays within R={r_comp*1000:.1f}mm"
        else:
            width_out = min(width_in, 2*r_comp)
            clipping_info = f"Clipped by R={r_comp*1000:.1f}mm"
        
        lines.append(f"NEXT: {hitted_comp.name}")
        lines.append(f"  - Distance from prev: {dist*1000:.1f} mm")
        lines.append(f"  - Incident Diameter: {width_in*1000:.2f} mm")
        lines.append(f"  - Output Diameter:   {width_out*1000:.2f} mm ({clipping_info})")
        
        # Physics specifics (Clarified Units)
        for p_key, p_val in hitted_comp.params.items():
            if p_key not in ['r', 'wavelength', 'n_rays', 'angle_range', 'width', 'pattern']:
                unit = ""
                if p_key == 'f': unit = " m"
                if p_key == 'line_density': unit = " lines/mm"
                lines.append(f"  - Parameter '{p_key}': {p_val}{unit}")
        
        # Special Grating multi-beam analysis
        if isinstance(hitted_comp, Grating):
            lines_mm = hitted_comp.params.get("line_density", 300)
            d = 1e-3 / lines_mm
            wl_m = src.params.get('wavelength', 532.0) * 1e-9
            
            n_vec = np.array([np.cos(np.radians(hitted_comp.angle)), np.sin(np.radians(hitted_comp.angle)), 0.0])
            v_y = np.array([-n_vec[1], n_vec[0], 0.0]) 
            sin_i = np.dot(curr_dir, v_y)
            
            n_orders = hitted_comp.params.get("n_orders", 2)
            lines.append(f"  - Multi-beam branching (Order Angles):")
            for m in range(-n_orders, n_orders + 1):
                sin_m = sin_i + m * (wl_m / d)
                if abs(sin_m) <= 1.0:
                    theta_m = np.degrees(np.arcsin(sin_m))
                    lines.append(f"    * m={m:2d}: Angle = {theta_m:6.2f}°")
                else:
                    lines.append(f"    * m={m:2d}: EVANESCENT (Blocked)")

        lines.append("-" * 40)
        
        # Update
        curr_pos = closest_hit
        beam_width = width_out
        processed.append(hitted_comp)
        axis_ray.origin = closest_hit
        
        # Change direction if mirror
        if isinstance(hitted_comp, Mirror):
            angle_rad = np.radians(hitted_comp.angle)
            n_3d = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
            curr_dir = curr_dir - 2 * np.dot(curr_dir, n_3d) * n_3d
            axis_ray.direction = curr_dir
        elif isinstance(hitted_comp, Detector):
            break
            
    lines.append("ANALYSIS COMPLETE.")
    lines.append("="*60)
    
    # Generate temporary random noise image (placeholder for wave engine)
    # High-resolution 2048x2048 computation grid
    image_data = np.random.rand(2048, 2048)
    
    return "\n".join(lines), image_data
