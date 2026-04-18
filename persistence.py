import json
import os
from optics_engine import OpticalSystem
from components import *

class SceneManager:
	"""Handles persistence and orchestration of the optical setup."""
	def __init__(self, engine=None):
		self.system = OpticalSystem()
		
		# Component factory for deserialization (classes must be registered)
		self.factory = {
			"PointSource": PointSource,
			"Lens": Lens,
			"Mirror": Mirror,
			"Detector": Detector,
			"Aperture": Aperture,
			"Grating": Grating,
			"HighPassFilter": HighPassFilter,
			"ArrowObject": ArrowObject,
			"TestTarget": TestTarget
		}

	@property
	def components(self):
		return self.system.components

	@property
	def rays(self):
		return self.system.rays

	def update(self):
		return self.system.update()

	def clear(self):
		self.system.components = []

	def add_component(self, component):
		self.system.components.append(component)

	def save(self, filename):
		data = [comp.to_dict() for comp in self.components]
		with open(filename, 'w') as f:
			json.dump(data, f, indent=4)
		print(f"Scene saved to {filename}")

	def load(self, filename):
		if not os.path.exists(filename):
			print(f"File {filename} not found.")
			return
		
		with open(filename, 'r') as f:
			data = json.load(f)
			
		self.clear()
		for item_data in data:
			comp_type = item_data.get("type")
			if comp_type in self.factory:
				cls = self.factory[comp_type]
				comp = cls.from_dict(item_data)
				self.add_component(comp)
		print(f"Loaded {len(self.components)} items from {filename}")

if __name__ == "__main__":
	sm = SceneManager()
	sm.add_component(Lens(0.1, 0, 0, f=0.1))
	sm.save("test_scene.json")
	sm.load("test_scene.json")
	print(f"Component: {sm.components[0].name}")
