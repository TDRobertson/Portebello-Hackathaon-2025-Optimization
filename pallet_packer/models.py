from __future__ import annotations

from typing import Literal, Optional, Tuple, List
from pydantic import BaseModel, Field, field_validator


LabelSide = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Axis = Literal["x", "y", "z"]


class Pallet(BaseModel):
    length: float
    width: float
    height_limit: float


class BoxType(BaseModel):
    id: str
    length: float
    width: float
    height: float
    quantity: int
    label_side: Optional[LabelSide] = None

    @field_validator("length", "width", "height")
    @classmethod
    def positive_dimensions(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

    @field_validator("quantity")
    @classmethod
    def positive_quantity(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Quantity must be non-negative")
        return v

    def orientations(self) -> List[Tuple[float, float, float, str]]:
        # return all axis-aligned orientations (L,W,H) with a tag string
        l, w, h = self.length, self.width, self.height
        return [
            (l, w, h, "LWH"),
            (w, l, h, "WLH"),
            (l, h, w, "LHW"),
            (w, h, l, "WHL"),
            (h, l, w, "HLW"),
            (h, w, l, "HWL"),
        ]


class InputSpec(BaseModel):
    pallet: Pallet
    boxes: List[BoxType]


class Placement(BaseModel):
    box_id: str
    index: int
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    orientation: str
    label_side: Optional[LabelSide] = None

    def bbox(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.x + self.length, self.y + self.width, self.z + self.height)


class Result(BaseModel):
    placements: List[Placement]
    unplaced: List[Tuple[str, int]] = Field(default_factory=list)
    utilization: float
    used_height: float
    pick_sequence: List[Tuple[str, int]]
    stack_sequence: List[Tuple[str, int]]


