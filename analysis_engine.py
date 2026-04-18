import numpy as np
import random
import json
from components import PointSource, BeamSource, Lens, Mirror, Detector, Aperture, Grating, HighPassFilter
from optics_engine import Ray

def calculate_analysis(system, detect_comp):
	"""
	Analyzes the optical path from the main source to the target detector.
	Prints detailed beam transformation data for each element encountered.
	Returns a 2048x2048 placeholder image (noise).
	"""
	print("\n" + "="*80)
	print(f" FULL OPTICAL SYSTEM DIAGNOSTICS: {detect_comp.name}")
	print("="*80)

	# 1. Identify main source and axis path
	src = next((c for c in system.components if isinstance(c, (PointSource, BeamSource))), None)
	if not src:
		print("Error: No light source found in system.")
		return np.zeros((2048, 2048), dtype=np.uint8)

	# Start tracking beam: [width, convergence_angle (half-angle)]
	# Positive angle = diverging, Negative = converging
	if isinstance(src, BeamSource):
		beam_w = src.params.get('width', 0.1)
		beam_angle = 0.0
	else: # PointSource
		beam_w = 0.0
		beam_angle = src.params.get('angle_range', 0.4) / 2.0 # half angle
	
	# Trace axis ray to get component sequence
	nx, ny = np.cos(np.radians(src.angle)), np.sin(np.radians(src.angle))
	axis_ray = Ray([src.x, src.y, 0.0], [nx, ny, 0.0])
	axis_ray._is_axis = True
	system.trace_ray(axis_ray)

	print(f"\n[PRIMARY SOURCE: {src.name}]")
	print(f"  > Position:  X={src.x:.3f}m, Y={src.y:.3f}m")
	print(f"  > Rotation:  {src.angle:.1f}°")
	print(f"  > Parameters: {json.dumps(src.params, indent=2).replace('{', '').replace('}', '').strip()}")

	print("\n" + "-"*80)
	print(f"{'Path Pos':<10} | {'Component':<15} | {'Global X, Y':<15} | {'Angle':<8} | {'Ø Input':<8} | {'Ø Apert'}")
	print("-" * 80)

	curr_pos = np.array([src.x, src.y, 0.0])
	total_dist = 0.0
	
	# Iterate through all hit components up to (and including) the detector
	for i, comp in enumerate(axis_ray.hitted_components):
		hit_pt = axis_ray.points[i+1]
		p2 = np.array([hit_pt[0], hit_pt[1], 0.0])
		segment_dist = np.linalg.norm(p2 - curr_pos)
		total_dist += segment_dist
		
		w_in = beam_w + 2 * segment_dist * np.tan(beam_angle)
		r_comp = comp.params.get('r', 0.05)
		aperture_diam = 2 * r_comp
		w_effective = min(w_in, aperture_diam)
		
		# Table Row with consistent alignment
		pos_str = f"({comp.x:+.3f}, {comp.y:+.3f})"
		print(f"z={total_dist:<7.3f} | {comp.name:<15} | {pos_str:<18} | {comp.angle:>7.1f}° | Ø:{w_in:>7.3f} | Ø_ap:{aperture_diam:>7.3f}")
		
		# Cleaner Parameter Print
		p_list = [f"{k}={v}" for k, v in comp.params.items() if k not in ['r', 'pattern', 'size']]
		p_str = ", ".join(p_list)
		if p_str:
			print(f"        ↳ Params: {p_str}")

		# Specific detailed prints
		if isinstance(comp, Grating):
			pat = comp.params.get('pattern', 'Linear Cosine')
			print(f"        ↳ Grating Type: {pat}")
		elif isinstance(comp, Detector):
			s = comp.params.get('size', 0.05)
			print(f"        ↳ Sensor Size: {s*1000:.1f} x {s*1000:.1f} mm")
			print(f"        ↳ Pixel Clock: 2048 x 2048 (px size: {s*1000/2048:.4f} mm)")
		
		if isinstance(comp, Lens):
			f = comp.params.get('f', 0.5)
			if f != 0: beam_angle -= (w_effective / 2.0) / f

		beam_w = w_effective
		curr_pos = p2
		if comp == detect_comp:
			break

	print("-" * 85)
	print(f"✓ Diagnostic report for {detect_comp.name} generated successfully.")
	print("="*85 + "\n")
	
	# Return a random noise image (placeholder for future wave-propagation results)
	return np.random.randint(0, 255, (512, 512), dtype=np.uint8) 
