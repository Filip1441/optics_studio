import numpy as np
import uuid
import json

class OpticalComponent:
	"""Base class for all optical components (V3 - Geometric)."""
	def __init__(self, x=0, y=0, angle=0, name="Component"):
		self.x = x # mm
		self.y = y # mm
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
			f = obj.params.get("f", 50.0)
			obj.name = f"Lens (f={f})"
		return obj

class PointSource(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0):
		super().__init__(x, y, angle, "Source")
		# Default params for ray generation
		self.params = {"n_rays": 21, "angle_range": 0.1, "r": 2.0, "wavelength": 532.0}

class Lens(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0, f=50.0, r=12.5):
		super().__init__(x, y, angle, "Lens")
		self.params = {"f": f, "r": r}

class Mirror(OpticalComponent):
	def __init__(self, x=0, y=0, angle=45, r=12.5):
		super().__init__(x, y, angle, "Mirror")
		self.params = {"r": r}

class Grating(OpticalComponent):
	"""Diffraction grating that splits rays into multiple orders."""
	def __init__(self, x=0, y=0, angle=0, r=12.5, line_density=300):
		super().__init__(x, y, angle, "Grating")
		# Physics: line_density
		# Visualization: n_orders to show, rays per each order to make it look 'solid'
		self.params = {
			"r": r,
			"line_density": line_density, 
			"n_orders": 2, 
			"rays_per_order": 9,
			"beam_width_preview": 10.0,
			"pattern": "Linear Cosine" # Linear Zebra/Cosine, Crossed Zebra/Cosine
		}

class Detector(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0, r=12.5):
		super().__init__(x, y, angle, "Detector")
		# size: physical side length of the 2048x2048 sensor [mm]
		self.params = {"r": r, "size": 10.0} 
		self.hits = []

class Aperture(OpticalComponent):
	"""An opening that clips rays outside its radius."""
	def __init__(self, x=0, y=0, angle=0, r=5.0):
		super().__init__(x, y, angle, "Aperture")
		self.params = {"r": r, "shape": "Circular"}

class ArrowObject(OpticalComponent):
	"""Placeholder for V3. Acts as a collection of point sources or decorative."""
	def __init__(self, x=0, y=0, angle=0):
		super().__init__(x, y, angle, "Arrow")
		self.params = {"r": 10.0}

class TestTarget(OpticalComponent):
	def __init__(self, x=0, y=0, angle=0):
		super().__init__(x, y, angle, "Target")
		self.params = {"size": 5.0}

class HighPassFilter(OpticalComponent):
	"""Blocks central rays (DC component) providing edge enhancement."""
	def __init__(self, x=0, y=0, angle=0, r=1.0):
		super().__init__(x, y, angle, "HighPassFilter")
		self.params = {"r": r}

