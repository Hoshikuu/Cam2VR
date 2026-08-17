from dataclasses import dataclass, field


@dataclass(slots=True)
class PosePoint:
    """Represents 1 point of the body

    Args:
        x (float): x position
        y (float): y position
        z (float): z position
        visibility (float): visibility confidence
        presence (float): presence confidence
    """
    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(slots=True)
class FullPose:
    """Represents the body detected in 1 frame

    Args:
        sequence (int): frame number
        capture_timestamp_100ns (int): Media Foundation capture timestamp
        inference_start_timestamp (float): inference start timestamp
        inference_end_timestamp (float): inference end timestamp
        points_2d (list[PosePoint]): body points in image coordinates
        points_3d (list[PosePoint]): body points in world coordinates
        valid (bool): if the pose is valid
    """
    sequence: int
    capture_timestamp_100ns: int
    inference_start_timestamp: float
    inference_end_timestamp: float
    points_2d: list[PosePoint] = field(default_factory=list)
    points_3d: list[PosePoint] = field(default_factory=list)
    valid: bool = False

    @property
    def inference_ms(self):
        return (self.inference_end_timestamp - self.inference_start_timestamp) * 1000.0