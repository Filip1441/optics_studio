import numpy as np
import logging

# Local imports (Moving to top to avoid isinstance issues with dynamic loading)
from components import PointSource, BeamSource, Lens, Mirror, Detector, Aperture, Grating, HighPassFilter

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("OpticsEngine")

class Ray:
	"""Simple 3D-aware Ray representation for 2D-view-3D-math optics."""
	def __init__(self, origin, direction, wavelength=532.0):
		# origin: [x, y, z], direction: [vx, vy, vz]
		self.origin = np.array(origin, dtype=float) 
		self.direction = np.array(direction, dtype=float)
		self.direction /= np.linalg.norm(self.direction)
		self.wavelength = wavelength # nm
		self.points = [self.origin[:2]] # XY projections for 2D UI
		self.z_hits = [self.origin[2]] # Z coordinates track
		self.hitted_components = [] # For analysis
		self.alive = True

	def propagate_to_plane(self, p_on_plane, n_plane):
		"""Propagates ray (in 3D) to intersection with a 2D component plane (YZ-aligned in local)."""
		# (p_3d - p0_3d) . n_3d = 0
		# We assume components are symmetric cylinders, so the plane is defined by x,y in 2D
		# n_plane here is [nx, ny, 0]
		n_3d = np.array([n_plane[0], n_plane[1], 0.0])
		p_on_plane_3d = np.array([p_on_plane[0], p_on_plane[1], 0.0])
		
		denom = np.dot(self.direction, n_3d)
		if abs(denom) < 1e-9: return None
		
		t = np.dot(p_on_plane_3d - self.origin, n_3d) / denom
		if t < 1e-6: return None
		
		hit_pt = self.origin + self.direction * t
		return hit_pt, t

class OpticalSystem:
	"""Geometric optics engine (V3 - Ray Only)."""
	def __init__(self):
		self.components = []
		self.rays = [] # Rays for GUI display
		self.table_bounds = 10.0 # 10x10 meter table

	def update(self):
		self.update_rays()

	def update_rays(self):
		from collections import deque
		self.rays = []
		all_trace_queue = deque() # Queue for BFS
		
		# --- LAYER 1: GUI RAYS (DETERMINISTIC) ---
		for src in [c for c in self.components if isinstance(c, (PointSource, BeamSource))]:
			n_gui = src.params.get("n_rays", 21)
			wl = src.params.get("wavelength", 532.0)
			rad_base = np.radians(src.angle)
			
			if isinstance(src, PointSource):
				div_ang = src.params.get("angle_range", 0.1)
				for a in np.linspace(-div_ang/2, div_ang/2, n_gui):
					ray_dir = [np.cos(rad_base + a), np.sin(rad_base + a), 0.0]
					ray = Ray([src.x, src.y, 0.0], ray_dir, wavelength=wl)
					self.rays.append(ray)
			else: # BeamSource
				width = src.params.get("width", 0.1)
				nx, ny = np.cos(rad_base), np.sin(rad_base)
				tx, ty = -ny, nx
				for offset in np.linspace(-width/2, width/2, n_gui):
					origin = np.array([src.x, src.y, 0.0]) + np.array([tx*offset, ty*offset, 0.0])
					ray = Ray(origin, [nx, ny, 0.0], wavelength=wl)
					self.rays.append(ray)

		# Process all rays
		for r in self.rays:
			all_trace_queue.append(r)
			
		for comp in self.components:
			if isinstance(comp, Grating):
				comp._gui_hits = []

		# Main Tracing Loop
		while all_trace_queue:
			ray = all_trace_queue.popleft()
			self.trace_ray(ray, all_trace_queue)

		# --- SECOND PASS: Automatic Grating Beam Regeneration ---
		# For gratings that were hit by GUI rays, we now spawn the dense fans
		grating_spawn_queue = deque()
		for comp in [c for c in self.components if isinstance(c, Grating)]:
			if hasattr(comp, '_gui_hits') and comp._gui_hits:
				# Calculate footprint in LOCAL coordinates
				p_center = np.array([comp.x, comp.y, 0.0])
				angle_rad = np.radians(comp.angle)
				n = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
				v_y = np.array([-n[1], n[0], 0.0]) 
				v_z = np.array([0.0, 0.0, 1.0])
				
				hits_data = []
				for h in comp._gui_hits:
					loc_y = np.dot(h['pt'] - p_center, v_y)
					sin_iy = np.dot(h['ray'].direction, v_y)
					sin_iz = np.dot(h['ray'].direction, v_z)
					hits_data.append({'y': loc_y, 'sy': sin_iy, 'sz': sin_iz, 'wl': h['ray'].wavelength, 'z': h['pt'][2], 'points': h['ray'].points})
				
				hits_data.sort(key=lambda x: x['y'])
				
				# Simplified logic: The output beam center should coincide with the input beam center
				y_center = np.mean([h['y'] for h in hits_data])
				width = max(0.01, hits_data[-1]['y'] - hits_data[0]['y'])
				
				n_orders = comp.params.get("n_orders", 2)
				rpo = comp.params.get("rays_per_order", 9)
				pattern = comp.params.get("pattern", "Linear")
				
				orders_y = range(-n_orders, n_orders + 1)
				orders_z = orders_y if pattern in ["Crossed", "Chessboard"] else range(1)
				
				lines_mm = comp.params.get("line_density", 300)
				d = (1e-3) / lines_mm 
				
				# For 'Simplified' look, we use the average incident ray as a template for ALL rays
				# This makes all rays in the diffracted fans appear to branch from the center of the hit
				idx_mid = len(hits_data) // 2
				template_hit = hits_data[idx_mid]
				
				for my in orders_y:
					for mz in orders_z:
						# For Chessboard, we might want to block half the orders, 
						# but usually it's just a 2D grating.
						if pattern == "Chessboard" and (my + mz) % 2 != 0:
							continue
							
						for i, offset in enumerate(np.linspace(y_center - width/2, y_center + width/2, rpo)):
							# Interpolate physics if possible, otherwise use template
							if len(hits_data) > 1:
								y_vals = [hd['y'] for hd in hits_data]
								interp_sy = np.interp(offset, y_vals, [hd['sy'] for hd in hits_data])
								interp_sz = np.interp(offset, y_vals, [hd['sz'] for hd in hits_data])
								interp_z = np.interp(offset, y_vals, [hd['z'] for hd in hits_data])
							else:
								interp_sy, interp_sz, interp_z = template_hit['sy'], template_hit['sz'], template_hit['z']
							
							wl_m = template_hit['wl'] * 1e-9
							sy, sz = interp_sy + my*(wl_m/d), interp_sz + mz*(wl_m/d)
							
							if (sy**2 + sz**2) <= 1.0:
								sc = np.sqrt(1.0 - sy**2 - sz**2)
								if np.dot(template_hit['sy']*v_y + template_hit['sz']*v_z + n, n) < 0: sc = -sc
								
								new_dir = sy*v_y + sz*v_z + sc*n
								new_dir /= np.linalg.norm(new_dir)
								
								origin = p_center + offset * v_y + interp_z * v_z + 1e-4 * n
								child = Ray(origin, new_dir, template_hit['wl'])
								# IMPROVED: Map each ray in the diffracted fan to the hit closest to its offset.
								# This ensures the output beams maintain the correct width and alignment with input rays.
								y_vals = [hd['y'] for hd in hits_data]
								closest_idx = np.abs(np.array(y_vals) - offset).argmin()
								parent_hit = hits_data[closest_idx]
								
								# START FIX: To avoid 'thick' rays (over-drawing the incident path), 
								# we start the child history exactly where the parent hit.
								child.points = [origin[:2]]
								child._is_generated = True
								self.rays.append(child)
								grating_spawn_queue.append(child)
					
		while grating_spawn_queue:
			ray = grating_spawn_queue.popleft()
			self.trace_ray(ray, grating_spawn_queue)

	def trace_ray(self, ray, queue=None):
		max_intersections = 10
		for _ in range(max_intersections):
			if not ray.alive: break
			
			closest_hit = None
			closest_t = float('inf')
			hitted_comp = None
			
			for comp in self.components:
				if isinstance(comp, (PointSource, BeamSource)): continue
				
				# Simplified collision: Treat everything as infinite line for math
				# but bound by component radius/size visually.
				# Lens/Mirror plane normal is its angle
				angle_rad = np.radians(comp.angle)
				normal = np.array([np.cos(angle_rad), np.sin(angle_rad)])
				
				res = ray.propagate_to_plane(np.array([comp.x, comp.y]), normal)
				if res:
					hit_pt, t = res
					# Check if hit is within component boundary (radius)
					dist_from_center = np.linalg.norm(hit_pt - np.array([comp.x, comp.y, 0.0]))
					# For Mirror/Lens, let's assume 10cm radius for now or use params
					# For Aperture, we need a large detection radius to catch rays hitting the walls
					r_comp = comp.params.get("r", 0.05) if hasattr(comp, 'params') else 0.05
					if isinstance(comp, Aperture):
						r_comp = 1.0 # 1m detection wall
					
					if t < closest_t and dist_from_center < r_comp:
						closest_t = t
						closest_hit = hit_pt
						hitted_comp = comp

			if hitted_comp:
				# Add XY projection for 2D UI
				ray.points.append(closest_hit[:2])
				ray.hitted_components.append(hitted_comp)
				ray.origin = closest_hit # Update 3D ray start for next segment
				
				if isinstance(hitted_comp, Mirror):
					angle_rad = np.radians(hitted_comp.angle)
					n_3d = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
					ray.direction = ray.direction - 2 * np.dot(ray.direction, n_3d) * n_3d
					ray.direction /= np.linalg.norm(ray.direction)
				elif isinstance(hitted_comp, Lens):
					f = hitted_comp.params.get("f", 0.5)
					if f == 0: f = 1e-9 
					
					angle_rad = np.radians(hitted_comp.angle)
					nx, ny = np.cos(angle_rad), np.sin(angle_rad)
					vx, vy = -ny, nx
					v_tangent = np.array([vx, vy, 0.0])
					w_z = np.array([0.0, 0.0, 1.0])
					
					# Transverse offsets in 3D
					h_u = np.dot(closest_hit[:2] - np.array([hitted_comp.x, hitted_comp.y]), v_tangent[:2])
					h_z = closest_hit[2]
					
					# Spherical lens formula (paraxial approximation in 3D)
					# deflection depends on axial ray speed to keep focus perfect
					v_axial = np.dot(ray.direction, np.array([nx, ny, 0.0]))
					ray.direction -= (v_axial * h_u / f) * v_tangent + (v_axial * h_z / f) * w_z
					ray.direction /= np.linalg.norm(ray.direction)
				elif isinstance(hitted_comp, Grating):
					is_anal = getattr(ray, '_is_analysis', False)
					is_axis = getattr(ray, '_is_axis', False)
					is_generated = getattr(ray, '_is_generated', False)
					
					# Axis rays continue as Order 0 to explore the full scene
					if not is_anal and not is_axis and not is_generated:
						if not hasattr(hitted_comp, '_gui_hits'): hitted_comp._gui_hits = []
						hitted_comp._gui_hits.append({'pt': closest_hit, 'ray': ray})
						ray.alive = False
						break
					
					lines_mm = hitted_comp.params.get("line_density", 300)
					d = (1e-3) / lines_mm 
					wl_m = ray.wavelength * 1e-9
					angle_rad = np.radians(hitted_comp.angle)
					n = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
					v_y = np.array([-n[1], n[0], 0.0]) 
					v_z = np.array([0.0, 0.0, 1.0])
					
					n_orders = hitted_comp.params.get("n_orders", 2)
					pattern = hitted_comp.params.get("pattern", "Linear Cosine")
					orders_y = range(-n_orders, n_orders + 1)
					orders_z = orders_y if "Crossed" in pattern else range(1)

					if is_anal:
						sin_iy, sin_iz = np.dot(ray.direction, v_y), np.dot(ray.direction, v_z)
						for my in orders_y:
							for mz in orders_z:
								if pattern == "Chessboard" and (my + mz) % 2 != 0: continue
								sy, sz = sin_iy + my*(wl_m/d), sin_iz + mz*(wl_m/d)
								if (sy**2 + sz**2) <= 1.0:
									sc = np.sqrt(1.0 - sy**2 - sz**2)
									if np.dot(ray.direction, n) < 0: sc = -sc
									new_dir = sy*v_y + sz*v_z + sc*n
									child = Ray(closest_hit, new_dir/np.linalg.norm(new_dir), ray.wavelength)
									child._is_analysis = True
									child.points = list(ray.points)
									if queue is not None: queue.append(child)
					elif is_axis:
						# Axis ray continues as order 0
						sin_iy, sin_iz = np.dot(ray.direction, v_y), np.dot(ray.direction, v_z)
						sc = np.sqrt(max(0, 1.0 - sin_iy**2 - sin_iz**2))
						if np.dot(ray.direction, n) < 0: sc = -sc
						ray.direction = sin_iy*v_y + sin_iz*v_z + sc*n
						ray.origin = closest_hit
						continue

				elif isinstance(hitted_comp, Aperture):
					r_stop = hitted_comp.params.get("r", 0.05)
					dist_3d = np.linalg.norm(closest_hit - np.array([hitted_comp.x, hitted_comp.y, 0.0]))
					if dist_3d > r_stop:
						ray.alive = False
						break
				elif hitted_comp.__class__.__name__ == "HighPassFilter": # Dynamic check for new type
					r_stop = hitted_comp.params.get("r", 0.01)
					dist_3d = np.linalg.norm(closest_hit - np.array([hitted_comp.x, hitted_comp.y, 0.0]))
					if dist_3d < r_stop: # Block central rays
						ray.alive = False
						break
				elif isinstance(hitted_comp, Detector):
					ray.origin = closest_hit
					continue
			else:
				end_pt = ray.origin + ray.direction * 10.0
				ray.points.append(end_pt[:2])
				ray.alive = False
				# logger.debug("Ray left the scene.")
				break

	def get_axis_path(self):
		"""Calculates the main optical axis path (starts from the first source)."""
		src = next((c for c in self.components if isinstance(c, (PointSource, BeamSource))), None)
		
		# Axis starts from source position and follows its main emission direction
		if src:
			nx, ny = np.cos(np.radians(src.angle)), np.sin(np.radians(src.angle))
			axis_ray = Ray([src.x, src.y, 0.0], [nx, ny, 0.0])
			axis_ray._is_axis = True
		else:
			axis_ray = Ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0])
			axis_ray._is_axis = True
			
		# Clear previous axis cache (optional but good for debugging)
		self.trace_ray(axis_ray)
		return axis_ray.points
