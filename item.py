class Item:
	def __init__(self,
				 location: str,
				 id: str,
				 thickness: int,
				 dim: tuple[int, int, int],
				 area: float,
				 n_pallets: int,
				 n_boxes: int) -> None:

		self.location = location
		self.id = id
		self.thickness = thickness
		self.dim = dim
		self.area = area
		self.n_pallets = n_pallets
		self.n_boxes = n_boxes

	def get_volume(self) -> None:
		return dim[0] * dim[1] * dim[2]
