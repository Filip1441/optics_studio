import numpy as np
import json
import LightPipes as lp
from components import PointSource, Lens, Mirror, Detector, Aperture, Grating, HighPassFilter, TestTarget
from optics_engine import Ray


def calculate_analysis(system, detect_comp, cancel_check=None):
	"""
	Full wave-optics analysis using LightPipes SmartGrid engine.
	All GUI coordinates are in mm; LightPipes works in meters.
	Returns (image_uint8, report_text) or (None, None) if cancelled.
	"""
	MM = 1e-3  # mm -> m conversion factor

	report = []
	report.append("\n" + "="*95)
	report.append(f" LIGHTPIPES WAVE ANALYSIS: {detect_comp.name}")
	report.append("="*95)

	# ── 1. Find source ──────────────────────────────────────────────
	src = next((c for c in system.components if isinstance(c, PointSource)), None)
	if not src:
		msg = "Error: No light source found in system."
		print(msg)
		return np.zeros((512, 512), dtype=np.uint8), msg

	wavelength_nm = src.params.get('wavelength', 532.0)
	wvl = wavelength_nm * 1e-9  # nm -> m

	# ── 2. Trace axis ray to get ordered component list ─────────────
	nx, ny = np.cos(np.radians(src.angle)), np.sin(np.radians(src.angle))
	axis_ray = Ray([src.x, src.y, 0.0], [nx, ny, 0.0])
	axis_ray.wavelength = wavelength_nm
	axis_ray._is_axis = True
	system.trace_ray(axis_ray)

	# Build ordered visit list along the axis
	visit_list = []
	prev_pos = np.array([src.x, src.y, 0.0])
	for i, comp in enumerate(axis_ray.hitted_components):
		hit_pt = axis_ray.points[i + 1]
		p2 = np.array([hit_pt[0], hit_pt[1], 0.0])
		dist_mm = np.linalg.norm(p2 - prev_pos)
		visit_list.append({'comp': comp, 'dist_mm': dist_mm})
		prev_pos = p2

	if cancel_check and cancel_check():
		return None, None

	# ── 3. Source parameters ────────────────────────────────────────
	N = 2048  # High resolution as requested

	# Detector size determines the final interpolation window
	det_size_mm = detect_comp.params.get('size', 10.0)
	det_size_m = det_size_mm * MM

	# Point source — extract divergence angle
	angle_range_rad = src.params.get('angle_range', 0.4)
	angle_deg = np.degrees(angle_range_rad)
	# Pinhole size from divergence: w0 = λ / (π * θ_half) 
	theta_half = angle_range_rad / 2.0
	if theta_half > 0:
		pinhole_m = wvl / (np.pi * theta_half)
	else:
		pinhole_m = 0.001
	current_div = angle_deg

	# SmartGrid: start with Nyquist-safe grid
	nyq_l = (wvl * N) / (2 * np.sin(max(theta_half, 1e-6)))
	curr_l = min(0.005, nyq_l * 0.5)
	curr_l = max(curr_l, 0.0005)  # at least 0.5mm

	F = lp.Begin(curr_l, wvl, N)
	F = lp.GaussAperture(pinhole_m, 0, 0, 1, F)
	report.append(f"\n[SOURCE: Point / Diverging]")
	report.append(f"  > Divergence: {angle_deg:.3f}°")
	report.append(f"  > Pinhole:    {pinhole_m*1e6:.2f} µm")

	report.append(f"  > Wavelength: {wavelength_nm:.1f} nm")
	report.append(f"  > Grid:       {N}x{N}, L_start={curr_l*1000:.3f} mm")

	# ── 4. Report table header ──────────────────────────────────────
	report.append("\n" + "-"*95)
	report.append(f"{'Z [mm]':<8} | {'ΔZ':<7} | {'Component':<15} | {'Grid L':<10} | {'Action'}")
	report.append("-" * 95)

	current_z = 0.0  # accumulated propagation distance in meters

	# ── 5. Walk the axis, applying LightPipes actions ───────────────
	for vi, visit in enumerate(visit_list):
		if cancel_check and cancel_check():
			return None, None

		comp = visit['comp']
		dist_m = visit['dist_mm'] * MM  # segment distance in meters
		cumulative_mm = sum(v['dist_mm'] for v in visit_list[:vi+1])

		# ─── Propagation ───
		if dist_m > 1e-7:
			# Apodization: soft edges to kill boundary artifacts
			F = lp.GaussAperture(curr_l * 0.48, 0, 0, 10, F)

			# SmartGrid: resize if beam outgrows the grid
			beam_width = 2 * (current_z + dist_m) * np.tan(np.radians(current_div / 2.0))
			new_l = max(curr_l, beam_width * 2.5)
			nyq_l = (wvl * N) / (2 * np.sin(np.radians(max(current_div, 0.001) / 2.0)))
			new_l = min(new_l, nyq_l * 0.9)
			new_l = max(new_l, 0.0003)  # floor

			if abs(new_l - curr_l) / max(curr_l, 1e-9) > 0.05:
				F = lp.Interpol(new_l, N, 0, 0, 0, F)
				curr_l = new_l

			F = lp.Forvard(dist_m, F)
			current_z += dist_m

		# ─── Component action ───
		action_str = "propagate"

		if isinstance(comp, Lens):
			f_mm = comp.params.get('f', 50.0)
			r_mm = comp.params.get('r', 12.5)
			f_m = f_mm * MM
			r_m = r_mm * MM
			F = lp.Lens(f_m, 0, 0, F)
			if r_m > 0:
				F = lp.CircAperture(r_m, 0, 0, F)
			# Update divergence estimate
			if f_m != 0:
				current_div = abs(current_div - (curr_l / f_m) * (180 / np.pi))
			action_str = f"Lens f={f_mm:.1f}mm, R={r_mm:.1f}mm"

		elif isinstance(comp, Aperture):
			r_mm = comp.params.get('r', 5.0)
			r_m = r_mm * MM
			shape = comp.params.get('shape', 'Circular')
			if shape == 'Square':
				F = lp.RectAperture(r_m*2, r_m*2, 0, 0, 0, F)
			elif shape == 'Gaussian':
				x = np.linspace(-curr_l/2, curr_l/2, N)
				X, Y = np.meshgrid(x, x[::-1])
				F.field = F.field * np.exp(-(X**2 + Y**2) / (2 * r_m**2))
			else: # Default Circular
				F = lp.CircAperture(r_m, 0, 0, F)
			action_str = f"Aperture [{shape}] R={r_mm:.2f}mm"

		elif isinstance(comp, HighPassFilter):
			r_mm = comp.params.get('r', 1.0)
			r_m = r_mm * MM
			F = lp.CircScreen(r_m, 0, 0, F)
			action_str = f"CircScreen R={r_mm:.1f}mm"

		elif isinstance(comp, Grating):
			ld = comp.params.get('line_density', 300)
			pattern = comp.params.get('pattern', 'Linear Cosine')
			pitch_m = 1.0 / (ld * 1000)  # lines/mm -> period in meters
			dc = 0.5

			x = np.linspace(-curr_l / 2, curr_l / 2, N)
			if 'Zebra' in pattern:
				mask_v = ((x % pitch_m) < (pitch_m * dc)).astype(float)
				if 'Crossed' in pattern:
					F.field = F.field * mask_v[None, :] * mask_v[:, None]
				else:
					F.field = F.field * mask_v[None, :]
			else:  # Cosine
				mask = 0.5 * (1 + np.cos(2 * np.pi * x / pitch_m))
				if 'Crossed' in pattern:
					F.field = F.field * mask[None, :] * mask[:, None]
				else:
					F.field = F.field * mask[None, :]
			action_str = f"Grating {ld} l/mm [{pattern}]"

		elif isinstance(comp, Detector):
			action_str = f"Detector {det_size_mm:.1f}mm"

		elif isinstance(comp, Mirror):
			# Mirror reflects the coordinate system (Horizontal flip in our 2D-plane analysis)
			F.field = np.flip(F.field, axis=1)
			action_str = "Mirror Reflection (Field Flip)"

		elif isinstance(comp, TestTarget):
			s_m = comp.params.get('size', 5.0) * MM
			x = np.linspace(-curr_l / 2, curr_l / 2, N)
			# Align grid with image coordinates (Row 0 at Top)
			X, Y = np.meshgrid(x, x[::-1])
			# Letter 'F' Mask logic from optical_cli.py
			mask = np.zeros_like(X, dtype=bool)
			# Vertical stem
			mask |= (X > -s_m/4) & (X < -s_m/8) & (np.abs(Y) < s_m/2)
			# Top bar
			mask |= (Y > s_m/4) & (Y < s_m/2) & (X > -s_m/8) & (X < s_m/4)
			# Middle bar
			mask |= (np.abs(Y) < s_m/10) & (X > -s_m/8) & (X < s_m/8)
			
			F.field = F.field * mask.astype(float)
			action_str = f"Test Target 'F' (size={comp.params.get('size'):.1f}mm)"

		report.append(f"{cumulative_mm:<8.3f} | {visit['dist_mm']:<7.3f} | {comp.name:<15} | {curr_l*1000:<10.3f} | {action_str}")

		if comp == detect_comp:
			break

	# ── 6. Final interpolation to detector size ─────────────────────
	if cancel_check and cancel_check():
		return None, None

	# If the beam misses the detector completely, return empty
	if detect_comp not in axis_ray.hitted_components:
		return np.zeros((N, N), dtype=np.uint8), report

	# Calculate offset between axial ray and detector center
	idx = axis_ray.hitted_components.index(detect_comp)
	hit_pt = axis_ray.points_3d[idx+1]
	det_pos = np.array([detect_comp.x, detect_comp.y, 0.0])
	
	# Detector local orientation
	angle_rad = np.radians(detect_comp.angle)
	n = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
	v_y = np.array([-n[1], n[0], 0.0]) # Tangent in XY plane
	v_z = np.array([0.0, 0.0, 1.0])    # Binormal (Z)

	# Transverse offsets in meters
	dx_m = np.dot(hit_pt - det_pos, v_y) * MM
	dy_m = np.dot(hit_pt - det_pos, v_z) * MM

	# Interpolate with shift (-dx, -dy) to center the detector window on its physical location
	F = lp.Interpol(det_size_m, N, -dx_m, -dy_m, 0, F)
	intensity = lp.Intensity(0, F)

	# Normalize to uint8 with optional log scale
	if detect_comp.params.get('log_scale', False):
		# Log scale normalization
		intensity = np.log10(intensity + 1e-9)
		i_min, i_max = intensity.min(), intensity.max()
		if i_max > i_min:
			img_norm = ((intensity - i_min) / (i_max - i_min) * 255).astype(np.uint8)
		else:
			img_norm = np.zeros((N, N), dtype=np.uint8)
	else:
		# Linear scale normalization
		i_max = intensity.max()
		if i_max > 0:
			img_norm = (intensity / i_max * 255).astype(np.uint8)
		else:
			img_norm = np.zeros((N, N), dtype=np.uint8)

	report.append("-" * 95)
	report.append(f"  Final grid interpolated to {det_size_mm:.1f}mm detector")
	report.append(f"  Peak intensity: {intensity.max():.4e}")
	report.append(f"✓ LightPipes analysis complete.")
	report.append("=" * 95 + "\n")

	final_report = "\n".join(report)
	
	# Only print if not cancelled (final safety check)
	if not (cancel_check and cancel_check()):
		print(final_report)

	return img_norm, final_report
