from time import perf_counter
from math import atan2, degrees, sqrt
from os.path import isfile
from numpy import ndarray, uint8, float64, ascontiguousarray, asarray
from mediapipe import Image, ImageFormat, tasks
from .full_head import FullHead

class MediaPipeHead:
    """MediaPipe model to detect head position and orientation
    """
    def __init__(self, model_path: str):
        """Constructor

        Args:
            model_path (str): the path of the vision model

        Raises:
            FileNotFoundError: if the vision model doesnt exists
        """
        self.model_path = model_path

        if not isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.options = tasks.vision.FaceLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )

        self.landmarker = (tasks.vision.FaceLandmarker.create_from_options(self.options))
        self.previous_timestamp_ms = -1
    
        self.last_face_landmarks = None
        self.last_transformation_matrix = None

    def process(self, frame_rgb: ndarray, sequence: int, capture_timestamp_ns: int):
        """Processes the given frame

        Args:
            frame_rgb (ndarray): the captured frame
            sequence (int): the frame number
            capture_timestamp_ns (int): timestamp of the captured frame

        Returns:
            FullHead: full head information, position and orientation
        """
        self.validate(frame_rgb)

        self.timestamp_ms = max(int(capture_timestamp_ns // 10_000), self.previous_timestamp_ms + 1)

        self.previous_timestamp_ms = self.timestamp_ms

        self.mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=ascontiguousarray(frame_rgb),
        )

        self.inference_start = perf_counter()

        self.result = self.landmarker.detect_for_video(
            self.mp_image,
            self.timestamp_ms,
        )

        self.inference_end = perf_counter()

        self.last_face_landmarks = None
        self.last_transformation_matrix = None

        if not self.result.face_landmarks:
            return FullHead(
                sequence=sequence,
                capture_timestamp_ns=capture_timestamp_ns,
                inference_start_timestamp=self.inference_start,
                inference_end_timestamp=self.inference_end,
                valid=False,
            )

        if not self.result.facial_transformation_matrixes:
            return FullHead(
                sequence=sequence,
                capture_timestamp_ns=capture_timestamp_ns,
                inference_start_timestamp=self.inference_start,
                inference_end_timestamp=self.inference_end,
                valid=False,
            )

        face_landmarks = self.result.face_landmarks[0]

        self.matrix = asarray(
            self.result.facial_transformation_matrixes[0],
            dtype=float64,
        )

        if self.matrix.shape != (4, 4):
            if self.matrix.size == 16:
                self.matrix = self.matrix.reshape(4, 4)
            else:
                return FullHead(
                    sequence=sequence,
                    capture_timestamp_ns=capture_timestamp_ns,
                    inference_start_timestamp=self.inference_start,
                    inference_end_timestamp=self.inference_end,
                    valid=False,
                )

        self.last_face_landmarks = face_landmarks
        self.last_transformation_matrix = self.matrix.copy()

        x, y, z = self.extract_position(self.matrix)
        yaw, pitch, roll = self.extract_rotation(self.matrix)

        return FullHead(
            sequence=sequence,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            confidence=1.0,
            capture_timestamp_ns=capture_timestamp_ns,
            inference_start_timestamp=self.inference_start,
            inference_end_timestamp=self.inference_end,
            valid=True
        )

    @staticmethod
    def validate(frame_rgb: ndarray):
        """Validates the frame to the standards

        Args:
            frame_rgb (np.ndarray): the frame motherfucker

        Raises:
            TypeError: type of array not right
            TypeError: type of data not right
            ValueError: dimensions of the array not right
            ValueError: colors of the array not right
        """
        if not isinstance(frame_rgb, ndarray):
            raise TypeError("frame_rgb must be np.ndarray")

        if frame_rgb.dtype != uint8:
            raise TypeError(f"frame_rgb must be np.uint8 not {frame_rgb.dtype}")

        if frame_rgb.ndim != 3:
            raise ValueError(f"frame_rgb must be 3 dimensions not {frame_rgb.ndim} dimensions")

        if frame_rgb.shape[2] != 3:
            raise ValueError(f"frame_rgb must be (x, y, 3) not (x, y, {frame_rgb.shape[2]})")

    @staticmethod
    def extract_position(matrix: ndarray):
        """Extracts the position of the matrix

        Args:
            matrix (ndarray): the matrix with the position

        Returns:
            flaot, float, float: x, y, z
        """

        x = float(matrix[0, 3])
        y = float(matrix[1, 3])
        z = float(matrix[2, 3])

        return x, y, z

    @staticmethod
    def extract_rotation(matrix: ndarray):
        """Extracts the rotation of the matrix

        Args:
            matrix (ndarray): the matrix with the rotation

        Returns:
            degrees, degrees, degrees: yaw, pitch, roll
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

        singular = sy < 1e-6

        if not singular:
            pitch_rad = atan2(r21, r22)
            yaw_rad = atan2(-r20, sy)
            roll_rad = atan2(r10, r00)
        else:
            pitch_rad = atan2(-r12, r11)
            yaw_rad = atan2(-r20, sy)
            roll_rad = 0.0

        return degrees(yaw_rad), degrees(pitch_rad), degrees(roll_rad)

    def close(self):
        """Closing cleaning
        """
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

        self.last_face_landmarks = None
        self.last_transformation_matrix = None