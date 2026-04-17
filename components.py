import numpy as np
import uuid
import json

class OpticalComponent:
	"""Base class for all optical components (V3 - Geometric)."""
	def __init__(self, x=0, y=0, angle=0, name="Component"):
		self.x = x # meters
		self.y = y # meters
		self.angle = angle # degrees (0 = normal facing RIGHT)
		self.name = name
		self.uid = str(uuid.uuid4())
		self.params = {}

	def to_dict(self):
		return {
			"type": self.__class__.__name__,
			"x": self.x,
			"y": self.y,
			"angle": self.angle,
			"name": self.name,
			"uid": self.uid,
			"params": self.params
		}

	@classmethod
	def from_dict(cls, data):
		# Create with defaults, then override explicitly
		obj = cls()
		obj.x = data.get("x", 0)
		obj.y = data.get("y", 0)
		obj.angle = data.get("angle", 0)
		obj.name = data.get("name", obj.name)
		obj.uid = data.get("uid", obj.uid)
		obj.params = data.get("params", {})
		
		# Specialized logic for certain components
		if isinstance(obj, Lens):
			f = obj.params.get("f", 0.5)
			obj.name = f"Lens (f={f})"
		return obj

class PointSource(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0):
		super().__init__(x, y, angle, "Source")
		# Default params for ray generation
		self.params = {"n_rays": 21, "angle_range": 0.1, "r": 0.02, "wavelength": 532.0}

class Lens(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0, f=0.5, r=0.2):
		# Default to larger scale (f=50cm, r=20cm) as user wants 10m table
		super().__init__(x, y, angle, f"Lens (f={f})")
		self.params = {"f": f, "r": r}

class Mirror(OpticalComponent):
	def __init__(self, x=0, y=0, angle=45, r=0.2):
		super().__init__(x, y, angle, "Mirror")
		self.params = {"r": r}

class Grating(OpticalComponent):
	"""Diffraction grating that splits rays into multiple orders."""
	def __init__(self, x=0, y=0, angle=0, r=0.2, line_density=300):
		super().__init__(x, y, angle, "Grating")
		# Physics: line_density
		# Visualization: n_orders to show, rays per each order to make it look 'solid'
		self.params = {
			"r": r,
			"line_density": line_density, 
			"n_orders": 2, 
			"rays_per_order": 9,
			"beam_width_preview": 0.1
		}

class Detector(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0, r=0.2):
		super().__init__(x, y, angle, "Detector")
		self.params = {"r": r}
		self.hits = []

class Aperture(OpticalComponent):
	"""An opening that clips rays outside its radius."""
	def __init__(self, x=0, y=0, angle=0, r=0.05):
		super().__init__(x, y, angle, "Aperture")
		self.params = {"r": r}

class BeamSource(OpticalComponent):
	"""Source that emits parallel rays (collimated beam)."""
	def __init__(self, x=0, y=0, angle=0, width=0.1):
		super().__init__(x, y, angle, "Beam Source")
		self.params = {"n_rays": 11, "width": width, "r": 0.05, "wavelength": 532.0}

class ArrowObject(OpticalComponent):
	"""Placeholder for V3. Acts as a collection of point sources or decorative."""
	def __init__(self, x=0, y=0, angle=0):
		super().__init__(x, y, angle, "Arrow")
		self.params = {"r": 0.1}
class HighPassFilter(OpticalComponent):
	"""Blocks central rays (DC component) providing edge enhancement."""
	def __init__(self, x=0, y=0, angle=0, r=0.01):
		super().__init__(x, y, angle, "HighPassFilter")
		self.params = {"r": r}

class CrossGrating(OpticalComponent):
	"""2D diffraction grating (grid)."""
	def __init__(self, x=0, y=0, angle=0, r=0.2, line_density=300):
		super().__init__(x, y, angle, "CrossGrating")
		self.params = {
			"r": r,
			"line_density": line_density, 
			"n_orders": 2, 
			"rays_per_order": 5,
			"beam_width_preview": 0.1
		}
