from dataclasses import dataclass, field

@dataclass(slots=True)
class PosePoint:
    """Represents 1 point of the body

    Args:
        x (float): represents x in a 2D image
        y (float): represents y in a 2D image
        z (float): represents z in a 3D space
        visibility (float): is for knowing if is visible
        presence (float): is for knowing if is in the image
    """
    x: float
    y: float
    z: float
    visibility: float
    presence: float

@dataclass(slots=True)
class FullPose:
    """Creates 2 lists for the full body position 2D and 3D, having extras for more information

    Args:
        sequence (int): the frame number
        capture_timestamp_ns (float): timestamp of the frame
        inference_start_timestamp (float): inference start timestamp
        inference_end_timestamp (float): inference end timestamp
        points_2d (list[PosePoint]): list of pose point in 2D
        points_3d (list[PosePoint]): list of pose point in 3D
        valid (bool): if the pose is valid
    """
    sequence: int
    capture_timestamp_ns: float
    inference_start_timestamp: float
    inference_end_timestamp: float
    points_2d: list[PosePoint] = field(default_factory=list)
    points_3d: list[PosePoint] = field(default_factory=list)
    valid: bool = False

    @property
    def inference_ms(self):
        """returns the time taken for the inference to finish

        Returns:
            float: time in ms
        """
        return (self.inference_end_timestamp - self.inference_start_timestamp) * 1000.0