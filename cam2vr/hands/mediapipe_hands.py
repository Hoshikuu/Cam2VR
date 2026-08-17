from math import atan2, degrees, sqrt
from os.path import isfile
from time import perf_counter

from mediapipe import Image, ImageFormat, tasks
from numpy import (
    array,
    ascontiguousarray,
    column_stack,
    cross,
    float64,
    identity,
    ndarray,
    uint8
)
from numpy.linalg import norm

from .full_hands import HandObservation, HandPoint, HandsObservation
from .hand_map import HAND_POINT_COUNT, PALM_POINTS, INDEX_MCP, MIDDLE_MCP, PINKY_MCP, WRIST


class MediaPipeHands:
    """MediaPipe model to detect hand position and orientation
    """
    def __init__(self, model_path: str):
        """Constructor

        Args:
            model_path (str): path of the vision model

        Raises:
            FileNotFoundError: if the vision model does not exist
        """
        self.model_path = model_path

        if not isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        options = tasks.vision.HandLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = tasks.vision.HandLandmarker.create_from_options(options)
        self.previous_timestamp_ms = -1
        self.last_rotation_matrices = {}

    def process(self, frame_rgb: ndarray, sequence: int, capture_timestamp_100ns: int):
        """Processes the given frame

        Args:
            frame_rgb (ndarray): captured RGB frame
            sequence (int): frame number
            capture_timestamp_100ns (int): Media Foundation capture timestamp

        Returns:
            HandsObservation: detected hands
        """
        self.validate(frame_rgb)

        sequence = int(sequence)
        capture_timestamp_100ns = int(capture_timestamp_100ns)

        timestamp_ms = max(
            capture_timestamp_100ns // 10_000,
            self.previous_timestamp_ms + 1
        )
        self.previous_timestamp_ms = timestamp_ms

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=ascontiguousarray(frame_rgb)
        )

        inference_start_timestamp = perf_counter()
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        inference_end_timestamp = perf_counter()

        hands = HandsObservation(
            sequence=sequence,
            capture_timestamp_100ns=capture_timestamp_100ns,
            inference_start_timestamp=inference_start_timestamp,
            inference_end_timestamp=inference_end_timestamp
        )

        self.last_rotation_matrices = {}

        if not result.hand_landmarks:
            return hands

        detected_count = min(
            len(result.hand_landmarks),
            len(result.hand_world_landmarks),
            len(result.handedness)
        )

        for index in range(detected_count):
            handedness_result = result.handedness[index]

            if not handedness_result:
                continue

            category = handedness_result[0]
            handedness = category.category_name or category.display_name or "Unknown"
            handedness = handedness.strip().capitalize()
            handedness_score = float(category.score) if category.score is not None else 0.0

            points_2d = [self.convert_point(point) for point in result.hand_landmarks[index]]
            points_3d = [self.convert_point(point) for point in result.hand_world_landmarks[index]]

            if len(points_2d) != HAND_POINT_COUNT or len(points_3d) != HAND_POINT_COUNT:
                continue

            rotation_matrix, yaw, pitch, roll, orientation_valid = self.calculate_orientation(points_3d)

            hand = HandObservation(
                sequence=sequence,
                handedness=handedness,
                handedness_score=handedness_score,
                palm=self.calculate_palm_center(points_2d),
                wrist=points_2d[WRIST],
                palm_world=self.calculate_palm_center(points_3d),
                wrist_world=points_3d[WRIST],
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                orientation_valid=orientation_valid,
                points_2d=points_2d,
                points_3d=points_3d,
                capture_timestamp_100ns=capture_timestamp_100ns,
                valid=True
            )

            if handedness not in ("Left", "Right"):
                continue

            current_hand = hands.left if handedness == "Left" else hands.right

            if current_hand is not None and hand.handedness_score <= current_hand.handedness_score:
                continue

            if handedness == "Left":
                hands.left = hand
            else:
                hands.right = hand

            if orientation_valid:
                self.last_rotation_matrices[handedness] = rotation_matrix

        return hands

    @staticmethod
    def convert_point(point):
        """Converts a MediaPipe point to a Cam2VR point

        Args:
            point: MediaPipe hand point

        Returns:
            HandPoint: converted point
        """
        return HandPoint(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z)
        )

    @staticmethod
    def calculate_palm_center(points: list[HandPoint]):
        """Calculates the center of the palm

        Args:
            points (list[HandPoint]): hand points

        Returns:
            HandPoint: palm center
        """
        count = len(PALM_POINTS)

        return HandPoint(
            x=sum(points[index].x for index in PALM_POINTS) / count,
            y=sum(points[index].y for index in PALM_POINTS) / count,
            z=sum(points[index].z for index in PALM_POINTS) / count
        )

    @staticmethod
    def point_to_vector(point: HandPoint):
        """Converts a hand point to a NumPy vector
        """
        return array([point.x, point.y, point.z], dtype=float64)

    @staticmethod
    def normalize_vector(vector: ndarray):
        """Normalizes a vector
        """
        length = norm(vector)

        if length < 1e-8:
            return None

        return vector / length

    @classmethod
    def calculate_orientation(cls, points_3d: list[HandPoint]):
        """Calculates the orientation of the hand

        Returns:
            ndarray, float, float, float, bool: matrix, yaw, pitch, roll and validity
        """
        wrist = cls.point_to_vector(points_3d[WRIST])
        index_mcp = cls.point_to_vector(points_3d[INDEX_MCP])
        middle_mcp = cls.point_to_vector(points_3d[MIDDLE_MCP])
        pinky_mcp = cls.point_to_vector(points_3d[PINKY_MCP])

        y_axis = cls.normalize_vector(middle_mcp - wrist)
        x_hint = cls.normalize_vector(index_mcp - pinky_mcp)

        if y_axis is None or x_hint is None:
            return identity(3), 0.0, 0.0, 0.0, False

        z_axis = cls.normalize_vector(cross(x_hint, y_axis))

        if z_axis is None:
            return identity(3), 0.0, 0.0, 0.0, False

        x_axis = cls.normalize_vector(cross(y_axis, z_axis))

        if x_axis is None:
            return identity(3), 0.0, 0.0, 0.0, False

        rotation_matrix = column_stack((x_axis, y_axis, z_axis))
        yaw, pitch, roll = cls.rotation_to_euler(rotation_matrix)

        return rotation_matrix, yaw, pitch, roll, True

    @staticmethod
    def rotation_to_euler(rotation: ndarray):
        """Extracts yaw, pitch and roll from a rotation matrix
        """
        r00 = float(rotation[0, 0])
        r10 = float(rotation[1, 0])
        r11 = float(rotation[1, 1])
        r12 = float(rotation[1, 2])
        r20 = float(rotation[2, 0])
        r21 = float(rotation[2, 1])
        r22 = float(rotation[2, 2])

        sy = sqrt(r00 * r00 + r10 * r10)

        if sy >= 1e-6:
            pitch_rad = atan2(r21, r22)
            yaw_rad = atan2(-r20, sy)
            roll_rad = atan2(r10, r00)
        else:
            pitch_rad = atan2(-r12, r11)
            yaw_rad = atan2(-r20, sy)
            roll_rad = 0.0

        return degrees(yaw_rad), degrees(pitch_rad), degrees(roll_rad)

    @staticmethod
    def validate(frame_rgb: ndarray):
        """Validates the given frame
        """
        if not isinstance(frame_rgb, ndarray):
            raise TypeError("frame_rgb must be np.ndarray")

        if frame_rgb.dtype != uint8:
            raise TypeError(f"frame_rgb must be np.uint8 not {frame_rgb.dtype}")

        if frame_rgb.ndim != 3:
            raise ValueError(f"frame_rgb must be 3 dimensions not {frame_rgb.ndim} dimensions")

        if frame_rgb.shape[2] != 3:
            raise ValueError(f"frame_rgb must be (x, y, 3) not (x, y, {frame_rgb.shape[2]})")

    def close(self):
        """Closes the MediaPipe model
        """
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        self.last_rotation_matrices = {}