from math import atan2, degrees, sqrt
from os.path import isfile
from time import perf_counter

from mediapipe import Image, ImageFormat, tasks
from numpy import asarray, ascontiguousarray, float64, ndarray, uint8

from .full_head import FullHead


class MediaPipeHead:
    """MediaPipe model to detect head position and orientation
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

        options = tasks.vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True
        )

        self.landmarker = tasks.vision.FaceLandmarker.create_from_options(options)
        self.previous_timestamp_ms = -1
        self.last_face_landmarks = None
        self.last_transformation_matrix = None

    def process(self, frame_rgb: ndarray, sequence: int, capture_timestamp_100ns: int):
        """Processes the given frame

        Args:
            frame_rgb (ndarray): captured RGB frame
            sequence (int): frame number
            capture_timestamp_100ns (int): Media Foundation capture timestamp

        Returns:
            FullHead: detected head
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

        self.last_face_landmarks = None
        self.last_transformation_matrix = None

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return FullHead(
                sequence=sequence,
                capture_timestamp_100ns=capture_timestamp_100ns,
                inference_start_timestamp=inference_start_timestamp,
                inference_end_timestamp=inference_end_timestamp,
                valid=False
            )

        matrix = asarray(result.facial_transformation_matrixes[0], dtype=float64)

        if matrix.size != 16:
            return FullHead(
                sequence=sequence,
                capture_timestamp_100ns=capture_timestamp_100ns,
                inference_start_timestamp=inference_start_timestamp,
                inference_end_timestamp=inference_end_timestamp,
                valid=False
            )

        matrix = matrix.reshape(4, 4)
        self.last_face_landmarks = result.face_landmarks[0]
        self.last_transformation_matrix = matrix.copy()

        x, y, z = self.extract_position(matrix)
        yaw, pitch, roll = self.extract_rotation(matrix)

        return FullHead(
            sequence=sequence,
            capture_timestamp_100ns=capture_timestamp_100ns,
            inference_start_timestamp=inference_start_timestamp,
            inference_end_timestamp=inference_end_timestamp,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            confidence=1.0,
            valid=True
        )

    @staticmethod
    def extract_position(matrix: ndarray):
        """Extracts x, y and z from a transformation matrix
        """
        return (
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3])
        )

    @staticmethod
    def extract_rotation(matrix: ndarray):
        """Extracts yaw, pitch and roll from a transformation matrix
        """
        rotation = matrix[:3, :3]

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

        self.last_face_landmarks = None
        self.last_transformation_matrix = None