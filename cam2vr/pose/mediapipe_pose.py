from time import perf_counter
from os.path import isfile
from numpy import ndarray, uint8, ascontiguousarray
from mediapipe import Image, ImageFormat, tasks
from .full_pose import FullPose, Point

class MediaPipePose:
    """MediaPipe model to detect body points
    """
    def __init__(self, model_path: str):
        """Constructor

        Args:
            model_path (str): the path of the vision model

        Raises:
            FileNotFoundError: if the vision model doesnt exists
        """
        self.model_path: str = model_path

        if not isfile(self.model_path):
            raise FileNotFoundError(f"Model not found {self.model_path}")

        self.options = tasks.vision.PoseLandmarkerOptions(
            base_options=tasks.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )

        self.landmarker = tasks.vision.PoseLandmarker.create_from_options(self.options)
        self.previous_timestamp = -1

    def process(self, frame_rgb: ndarray, sequence: int, capture_timestamp: float):
        """Processes the given frame

        Args:
            frame_rgb (ndarray): the captured frame
            sequence (int): the frame number
            capture_timestamp (float): timestamp of the captured frame

        Returns:
            FullPose: full pose of the poins of the body
        """
        self.validate(frame_rgb)

        self.timestamp = max(int(capture_timestamp // 10_000), self.previous_timestamp + 1)

        self.previous_timestamp = self.timestamp

        self.mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=ascontiguousarray(frame_rgb)
        )

        self.inference_start = perf_counter()

        self.result = self.landmarker.detect_for_video(
            self.mp_image,
            self.timestamp
        )

        self.inference_end = perf_counter()

        if not self.result.pose_landmarks:
            return FullPose(
                sequence=sequence,
                capture_timestamp=capture_timestamp,
                inference_start_timestamp=self.inference_start,
                inference_end_timestamp=self.inference_end,
                valid=False
            )

        self.points_2d = []
        for point in self.result.pose_landmarks[0]:
            self.points_2d.append(self.convert_point(point))

        self.points_3d = []
        if self.result.pose_world_landmarks:
            for point in self.result.pose_world_landmarks[0]:
                self.points_3d.append(self.convert_point(point))

        return FullPose(
            sequence=sequence,
            capture_timestamp=capture_timestamp,
            inference_start_timestamp=self.inference_start,
            inference_end_timestamp=self.inference_end,
            points_2d=self.points_2d,
            points_3d=self.points_3d,
            valid=True
        )

    @staticmethod
    def convert_point(point):
        """Converts the MediaPipe point to Cam2VR point

        Args:
            point (?): point of mediapipe

        Returns:
            PositionPoint: the point converted to Cam2VR point
        """
        visibility = 0.0
        presence = 0.0
        if point.visibility is not None:
            visibility = point.visibility
        if point.presence is not None:
            presence = point.presence

        return Point(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=float(visibility),
            presence=float(presence)
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

    def close(self):
        """Closing cleaning
        """
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None