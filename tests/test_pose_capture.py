from collections import deque
from pathlib import Path
from sys import path
from time import perf_counter

import cv2
from numpy import mean, ndarray, percentile


FILE_DIRECTORY = Path(__file__).resolve().parent
ROOT = FILE_DIRECTORY.parent if FILE_DIRECTORY.name == "tests" else FILE_DIRECTORY

if str(ROOT) not in path:
    path.insert(0, str(ROOT))


from cam2vr.hskcamera import hskcamera
from cam2vr.pose.mediapipe_pose import MediaPipePose
from cam2vr.visualization.pose_overlay import draw_pose


MODEL_PATH = ROOT / "cam2vr" / "models" / "pose_landmarker_full.task"

CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -5
FRAME_TIMEOUT_MS = 1000
MAX_TIMEOUTS = 5

WINDOW_NAME = "Cam2VR - Pose Tracking"


class RealtimeMetrics:
    """Calculates recent camera and pose metrics
    """
    def __init__(self, maximum_samples: int = 120):
        self.camera_timestamps = deque(maxlen=maximum_samples)
        self.pose_timestamps = deque(maxlen=maximum_samples)
        self.inference_times_ms = deque(maxlen=maximum_samples)
        self.last_sequence = None
        self.dropped_frames = 0
        self.processed_frames = 0
        self.invalid_detections = 0

    def add_camera_frame(self, sequence: int, capture_timestamp: float):
        self.camera_timestamps.append(capture_timestamp)

        if self.last_sequence is not None:
            sequence_difference = sequence - self.last_sequence

            if sequence_difference > 1:
                self.dropped_frames += sequence_difference - 1

        self.last_sequence = sequence

    def add_pose_result(self, inference_ms: float, valid: bool):
        self.pose_timestamps.append(perf_counter())
        self.inference_times_ms.append(inference_ms)
        self.processed_frames += 1

        if not valid:
            self.invalid_detections += 1

    @staticmethod
    def calculate_fps(timestamps: deque):
        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]

        if elapsed <= 0.0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    @property
    def camera_fps(self):
        return self.calculate_fps(self.camera_timestamps)

    @property
    def pose_fps(self):
        return self.calculate_fps(self.pose_timestamps)

    @property
    def average_inference_ms(self):
        if not self.inference_times_ms:
            return 0.0

        return float(mean(self.inference_times_ms))

    @property
    def p95_inference_ms(self):
        if not self.inference_times_ms:
            return 0.0

        return float(percentile(self.inference_times_ms, 95))

    @property
    def invalid_percentage(self):
        if self.processed_frames == 0:
            return 0.0

        return self.invalid_detections / self.processed_frames * 100.0


def draw_metrics(
    frame_rgb: ndarray,
    metrics: RealtimeMetrics,
    sequence: int,
    frame_age_ms: float,
    inference_ms: float,
    pose_valid: bool
):
    """Draws camera and pose metrics
    """
    lines = [
        f"Camera FPS: {metrics.camera_fps:.1f}",
        f"Pose FPS: {metrics.pose_fps:.1f}",
        f"Inference: {inference_ms:.1f} ms",
        f"Inference avg: {metrics.average_inference_ms:.1f} ms",
        f"Inference P95: {metrics.p95_inference_ms:.1f} ms",
        f"Frame age: {frame_age_ms:.1f} ms",
        f"Sequence: {sequence}",
        f"Frames skipped: {metrics.dropped_frames}",
        f"Invalid pose: {metrics.invalid_percentage:.1f}%",
        f"Pose valid: {pose_valid}"
    ]

    output = frame_rgb.copy()
    panel = output.copy()
    line_height = 23

    cv2.rectangle(
        panel,
        (5, 5),
        (330, 13 + line_height * len(lines)),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(panel, 0.55, output, 0.45, 0, output)

    for index, line in enumerate(lines):
        color = (80, 255, 100)

        if line.startswith("Pose valid") and not pose_valid:
            color = (255, 80, 80)

        cv2.putText(
            output,
            line,
            (15, 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA
        )

    return output


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

    camera = hskcamera.Camera()
    pose_backend = None
    camera_opened = False
    camera_started = False
    metrics = RealtimeMetrics()

    try:
        camera_opened = camera.open(
            camera_index=CAMERA_INDEX,
            format_index=FORMAT_INDEX,
            exposure=EXPOSURE,
            list_formats=False
        )

        if not camera_opened:
            raise RuntimeError("No se pudo abrir la camara")

        camera_started = camera.start()

        if not camera_started:
            raise RuntimeError("No se pudo iniciar la captura")

        pose_backend = MediaPipePose(str(MODEL_PATH))

        print(f"Camara abierta: {camera.width}x{camera.height}")
        print("Pose Tracking iniciado")
        print("Pulsa Q o ESC para cerrar")

        last_sequence = 0
        consecutive_timeouts = 0
        timestamp_offset = None

        while True:
            result = camera.wait_for_next_frame(
                last_sequence=last_sequence,
                timeout_ms=FRAME_TIMEOUT_MS
            )

            if result is None:
                consecutive_timeouts += 1

                if consecutive_timeouts >= MAX_TIMEOUTS:
                    raise RuntimeError("La camara ha dejado de entregar frames")

                continue

            consecutive_timeouts = 0
            frame_rgb, sequence, capture_timestamp_100ns = result
            sequence = int(sequence)
            capture_timestamp_100ns = int(capture_timestamp_100ns)
            last_sequence = sequence

            capture_timestamp_s = capture_timestamp_100ns / 10_000_000.0

            if timestamp_offset is None:
                timestamp_offset = perf_counter() - capture_timestamp_s

            local_capture_timestamp = capture_timestamp_s + timestamp_offset
            metrics.add_camera_frame(sequence, local_capture_timestamp)

            pose = pose_backend.process(
                frame_rgb=frame_rgb,
                sequence=sequence,
                capture_timestamp_100ns=capture_timestamp_100ns
            )

            metrics.add_pose_result(
                inference_ms=pose.inference_ms,
                valid=pose.valid
            )

            frame_age_ms = (
                perf_counter() - local_capture_timestamp
            ) * 1000.0

            debug_rgb = draw_pose(frame_rgb, pose)
            debug_rgb = draw_metrics(
                frame_rgb=debug_rgb,
                metrics=metrics,
                sequence=sequence,
                frame_age_ms=frame_age_ms,
                inference_ms=pose.inference_ms,
                pose_valid=pose.valid
            )

            debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(WINDOW_NAME, debug_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nCaptura interrumpida")

    finally:
        if pose_backend is not None:
            pose_backend.close()

        if camera_started:
            camera.stop()

        if camera_opened:
            camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()