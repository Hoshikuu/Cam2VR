from os.path import isfile
from time import perf_counter

from mediapipe import Image, ImageFormat, tasks
from numpy import ascontiguousarray, ndarray, uint8

from .full_pose import FullPose, PosePoint
from .pose_map import POSE_POINT_COUNT


class MediaPipePose:
    """MediaPipe model to detect body points
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

        options = tasks.vision.PoseLandmarkerOptions(
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

        self.landmarker = tasks.vision.PoseLandmarker.create_from_options(options)
        self.previous_timestamp_ms = -1

    def process(self, frame_rgb: ndarray, sequence: int, capture_timestamp_100ns: int):
        """Processes the given frame

        Args:
            frame_rgb (ndarray): captured RGB frame
            sequence (int): frame number
            capture_timestamp_100ns (int): Media Foundation capture timestamp

        Returns:
            FullPose: detected body pose
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

        if not result.pose_landmarks or not result.pose_world_landmarks:
            return FullPose(
                sequence=sequence,
                capture_timestamp_100ns=capture_timestamp_100ns,
                inference_start_timestamp=inference_start_timestamp,
                inference_end_timestamp=inference_end_timestamp,
                valid=False
            )

        points_2d = [self.convert_point(point) for point in result.pose_landmarks[0]]
        points_3d = [self.convert_point(point) for point in result.pose_world_landmarks[0]]

        if len(points_2d) != POSE_POINT_COUNT or len(points_3d) != POSE_POINT_COUNT:
            return FullPose(
                sequence=sequence,
                capture_timestamp_100ns=capture_timestamp_100ns,
                inference_start_timestamp=inference_start_timestamp,
                inference_end_timestamp=inference_end_timestamp,
                valid=False
            )

        return FullPose(
            sequence=sequence,
            capture_timestamp_100ns=capture_timestamp_100ns,
            inference_start_timestamp=inference_start_timestamp,
            inference_end_timestamp=inference_end_timestamp,
            points_2d=points_2d,
            points_3d=points_3d,
            valid=True
        )

    @staticmethod
    def convert_point(point):
        """Converts a MediaPipe point to a Cam2VR point

        Args:
            point: MediaPipe pose point

        Returns:
            PosePoint: converted point
        """
        visibility = point.visibility if point.visibility is not None else 0.0
        presence = point.presence if point.presence is not None else 0.0

        return PosePoint(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=float(visibility),
            presence=float(presence)
        )

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