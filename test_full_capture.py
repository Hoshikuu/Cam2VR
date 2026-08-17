from pathlib import Path
from sys import path
from threading import Condition, Lock, Thread
from time import perf_counter

import cv2
from numpy import ndarray


FILE_DIRECTORY = Path(__file__).resolve().parent
ROOT = FILE_DIRECTORY.parent if FILE_DIRECTORY.name == "tests" else FILE_DIRECTORY

if str(ROOT) not in path:
    path.insert(0, str(ROOT))


from cam2vr.hands.mediapipe_hands import MediaPipeHands
from cam2vr.head.mediapipe_head import MediaPipeHead
from cam2vr.hskcamera import hskcamera
from cam2vr.pose.mediapipe_pose import MediaPipePose
from cam2vr.visualization.hand_overlay import draw_hand_axes, draw_single_hand
from cam2vr.visualization.head_overlay import draw_face_landmarks, draw_head_axes
from cam2vr.visualization.pose_overlay import draw_pose


POSE_MODEL_PATH = ROOT / "cam2vr" / "models" / "pose_landmarker_full.task"
HAND_MODEL_PATH = ROOT / "cam2vr" / "models" / "hand_landmarker.task"
HEAD_MODEL_PATH = ROOT / "cam2vr" / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -5
FRAME_TIMEOUT_MS = 1000
MAX_TIMEOUTS = 5

WINDOW_NAME = "Cam2VR - Full Tracking"


class LatestFrame:
    """Stores only the latest captured frame
    """
    def __init__(self):
        self.condition = Condition()
        self.frame_rgb = None
        self.sequence = -1
        self.capture_timestamp_100ns = 0
        self.stopped = False

    def publish(
        self,
        frame_rgb: ndarray,
        sequence: int,
        capture_timestamp_100ns: int
    ):
        with self.condition:
            if self.stopped:
                return

            self.frame_rgb = frame_rgb.copy()
            self.sequence = sequence
            self.capture_timestamp_100ns = capture_timestamp_100ns
            self.condition.notify_all()

    def wait(self, previous_sequence: int):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stopped or self.sequence > previous_sequence
            )

            if self.stopped:
                return None

            return (
                self.frame_rgb,
                self.sequence,
                self.capture_timestamp_100ns
            )

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()


class ModelWorker:
    """Processes the latest frame in a separate thread
    """
    def __init__(
        self,
        name: str,
        backend,
        latest_frame: LatestFrame,
        extra_getter=None
    ):
        self.name = name
        self.backend = backend
        self.latest_frame = latest_frame
        self.extra_getter = extra_getter
        self.lock = Lock()
        self.result = None
        self.extra = None
        self.error = None
        self.started = False
        self.thread = Thread(
            name=f"{name}Worker",
            target=self.run
        )

    def start(self):
        self.thread.start()
        self.started = True

    def run(self):
        last_sequence = -1

        while True:
            frame = self.latest_frame.wait(last_sequence)

            if frame is None:
                return

            frame_rgb, sequence, capture_timestamp_100ns = frame
            last_sequence = sequence

            try:
                result = self.backend.process(
                    frame_rgb=frame_rgb,
                    sequence=sequence,
                    capture_timestamp_100ns=capture_timestamp_100ns
                )
                extra = (
                    self.extra_getter(self.backend)
                    if self.extra_getter is not None
                    else None
                )

                with self.lock:
                    self.result = result
                    self.extra = extra

            except Exception as error:
                with self.lock:
                    self.error = f"{self.name}: {error}"

                self.latest_frame.stop()
                return

    def snapshot(self):
        with self.lock:
            return self.result, self.extra, self.error

    def join(self):
        if self.started:
            self.thread.join()


def get_hand_data(backend: MediaPipeHands):
    return dict(backend.last_rotation_matrices)


def get_head_data(backend: MediaPipeHead):
    face_landmarks = backend.last_face_landmarks
    transformation_matrix = backend.last_transformation_matrix

    return (
        list(face_landmarks) if face_landmarks is not None else None,
        (
            transformation_matrix.copy()
            if transformation_matrix is not None
            else None
        )
    )


def result_status(result, valid: bool, current_sequence: int):
    if result is None:
        return "WAITING"

    state = "OK" if valid else "NOT FOUND"
    frame_delay = max(0, current_sequence - result.sequence)

    return f"{state} | {result.inference_ms:.1f} ms | delay {frame_delay}"


def draw_tracking(
    frame_rgb: ndarray,
    sequence: int,
    fps: float,
    pose,
    hands,
    head,
    hand_rotation_matrices,
    face_landmarks,
    head_transformation_matrix
):
    output = frame_rgb.copy()

    if pose is not None:
        output = draw_pose(output, pose)

    if hands is not None:
        if hands.has_left:
            output = draw_single_hand(output, hands.left)
            output = draw_hand_axes(
                output,
                hands.left,
                hand_rotation_matrices.get("Left")
            )

        if hands.has_right:
            output = draw_single_hand(output, hands.right)
            output = draw_hand_axes(
                output,
                hands.right,
                hand_rotation_matrices.get("Right")
            )

    if head is not None and head.valid:
        output = draw_face_landmarks(output, face_landmarks)
        output = draw_head_axes(
            output,
            face_landmarks,
            head_transformation_matrix
        )

    pose_valid = pose is not None and pose.valid
    hands_valid = hands is not None and hands.has_hands
    head_valid = head is not None and head.valid

    lines = [
        "Cam2VR Full Tracking",
        f"Camera: {fps:.1f} FPS | Sequence: {sequence}",
        f"Pose: {result_status(pose, pose_valid, sequence)}",
        f"Hands: {result_status(hands, hands_valid, sequence)}",
        f"Head: {result_status(head, head_valid, sequence)}"
    ]

    panel = output.copy()
    line_height = 24

    cv2.rectangle(
        panel,
        (5, 5),
        (440, 15 + line_height * len(lines)),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(panel, 0.60, output, 0.40, 0, output)

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (15, 28 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


def validate_models():
    for model_path in (
        POSE_MODEL_PATH,
        HAND_MODEL_PATH,
        HEAD_MODEL_PATH
    ):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No se encontro el modelo: {model_path}"
            )


def main():
    validate_models()

    camera = hskcamera.Camera()
    pose_backend = None
    hands_backend = None
    head_backend = None
    camera_opened = False
    camera_started = False
    workers = []

    latest_frame = LatestFrame()

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

        pose_backend = MediaPipePose(str(POSE_MODEL_PATH))
        hands_backend = MediaPipeHands(str(HAND_MODEL_PATH))
        head_backend = MediaPipeHead(str(HEAD_MODEL_PATH))

        pose_worker = ModelWorker(
            "Pose",
            pose_backend,
            latest_frame
        )
        hands_worker = ModelWorker(
            "Hands",
            hands_backend,
            latest_frame,
            get_hand_data
        )
        head_worker = ModelWorker(
            "Head",
            head_backend,
            latest_frame,
            get_head_data
        )
        workers = [
            pose_worker,
            hands_worker,
            head_worker
        ]

        for worker in workers:
            worker.start()

        print(f"Camara abierta: {camera.width}x{camera.height}")
        print("Pose, manos y cabeza iniciados")
        print("Pulsa Q o ESC para cerrar")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_sequence = 0
        consecutive_timeouts = 0
        fps = 0.0
        frame_count = 0
        fps_start = perf_counter()

        while True:
            camera_result = camera.wait_for_next_frame(
                last_sequence=last_sequence,
                timeout_ms=FRAME_TIMEOUT_MS
            )

            if camera_result is None:
                consecutive_timeouts += 1

                if consecutive_timeouts >= MAX_TIMEOUTS:
                    raise RuntimeError("La camara ha dejado de entregar frames")

                continue

            consecutive_timeouts = 0
            frame_rgb, sequence, capture_timestamp_100ns = camera_result
            sequence = int(sequence)
            capture_timestamp_100ns = int(capture_timestamp_100ns)
            last_sequence = sequence

            latest_frame.publish(
                frame_rgb,
                sequence,
                capture_timestamp_100ns
            )

            pose, _, pose_error = pose_worker.snapshot()
            hands, hand_rotation_matrices, hands_error = hands_worker.snapshot()
            head, head_data, head_error = head_worker.snapshot()

            tracking_error = pose_error or hands_error or head_error

            if tracking_error is not None:
                raise RuntimeError(tracking_error)

            hand_rotation_matrices = hand_rotation_matrices or {}
            face_landmarks, head_transformation_matrix = (
                head_data if head_data is not None else (None, None)
            )

            frame_count += 1
            elapsed = perf_counter() - fps_start

            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = perf_counter()

            debug_rgb = draw_tracking(
                frame_rgb=frame_rgb,
                sequence=sequence,
                fps=fps,
                pose=pose,
                hands=hands,
                head=head,
                hand_rotation_matrices=hand_rotation_matrices,
                face_landmarks=face_landmarks,
                head_transformation_matrix=head_transformation_matrix
            )

            debug_bgr = cv2.cvtColor(
                debug_rgb,
                cv2.COLOR_RGB2BGR
            )
            cv2.imshow(WINDOW_NAME, debug_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nCaptura interrumpida")

    finally:
        latest_frame.stop()

        if camera_started:
            camera.stop()

        for worker in workers:
            worker.join()

        if pose_backend is not None:
            pose_backend.close()

        if hands_backend is not None:
            hands_backend.close()

        if head_backend is not None:
            head_backend.close()

        if camera_opened:
            camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()