from dataclasses import dataclass

@dataclass(slots=True)
class FullHead:
    """Represents the detected position and orientation of the head.

    Args:
        sequence: camera frame sequence number.
        x: head position on X axis.
        y: head position on Y axis.
        z: head position on Z axis.
        yaw: rotation left/right in degrees.
        pitch: rotation up/down in degrees.
        roll: side tilt in degrees.
        confidence: tracking confidence approximation.
        capture_timestamp_ns: media Foundation timestamp.
        valid: whether a head was detected.
    """
    sequence: int
    capture_timestamp_ns: int
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