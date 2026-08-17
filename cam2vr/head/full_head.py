from dataclasses import dataclass


@dataclass(slots=True)
class FullHead:
    """Represents the detected position and orientation of the head

    Args:
        sequence (int): frame number
        capture_timestamp_100ns (int): Media Foundation capture timestamp
        inference_start_timestamp (float): inference start timestamp
        inference_end_timestamp (float): inference end timestamp
        x (float): head position on the x axis
        y (float): head position on the y axis
        z (float): head position on the z axis
        yaw (float): rotation around the vertical axis
        pitch (float): rotation around the lateral axis
        roll (float): rotation around the longitudinal axis
        confidence (float): tracking confidence
        valid (bool): if the head is valid
    """
    sequence: int
    capture_timestamp_100ns: int
    inference_start_timestamp: float
    inference_end_timestamp: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    confidence: float = 0.0
    valid: bool = False

    @property
    def inference_ms(self):
        return (self.inference_end_timestamp - self.inference_start_timestamp) * 1000.0