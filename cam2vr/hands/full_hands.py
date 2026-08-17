from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class HandPoint:
    """Represents 1 point of the hand

    Args:
        x (float): x position
        y (float): y position
        z (float): z position
    """
    x: float
    y: float
    z: float


@dataclass(slots=True)
class HandObservation:
    """Represents the position and orientation of 1 hand

    Args:
        sequence (int): frame number
        handedness (str): indicates whether the hand is left or right
        handedness_score (float): confidence of the detected handedness
        palm (HandPoint): palm position in image coordinates
        wrist (HandPoint): wrist position in image coordinates
        palm_world (HandPoint): palm position in world coordinates
        wrist_world (HandPoint): wrist position in world coordinates
        yaw (float): rotation around the vertical axis
        pitch (float): rotation around the lateral axis
        roll (float): rotation around the longitudinal axis
        orientation_valid (bool): if the calculated orientation is valid
        points_2d (list[HandPoint]): hand points in image coordinates
        points_3d (list[HandPoint]): hand points in world coordinates
        capture_timestamp_100ns (int): Media Foundation capture timestamp
        valid (bool): if the hand is valid
    """
    sequence: int
    handedness: str
    handedness_score: float
    palm: HandPoint
    wrist: HandPoint
    palm_world: HandPoint
    wrist_world: HandPoint
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    orientation_valid: bool = False
    points_2d: list[HandPoint] = field(default_factory=list)
    points_3d: list[HandPoint] = field(default_factory=list)
    capture_timestamp_100ns: int = 0
    valid: bool = False

    @property
    def x(self):
        return self.palm.x

    @property
    def y(self):
        return self.palm.y

    @property
    def z(self):
        return self.palm.z


@dataclass(slots=True)
class HandsObservation:
    """Represents the hands detected in 1 frame

    Args:
        sequence (int): frame number
        capture_timestamp_100ns (int): Media Foundation capture timestamp
        inference_start_timestamp (float): inference start timestamp
        inference_end_timestamp (float): inference end timestamp
        left (Optional[HandObservation]): detected left hand
        right (Optional[HandObservation]): detected right hand
    """
    sequence: int
    capture_timestamp_100ns: int
    inference_start_timestamp: float = 0.0
    inference_end_timestamp: float = 0.0
    left: Optional[HandObservation] = None
    right: Optional[HandObservation] = None

    @property
    def has_left(self):
        return self.left is not None and self.left.valid

    @property
    def has_right(self):
        return self.right is not None and self.right.valid

    @property
    def has_hands(self):
        return self.has_left or self.has_right

    @property
    def inference_ms(self):
        return (self.inference_end_timestamp - self.inference_start_timestamp) * 1000.0