from collections import deque
from pathlib import Path
import time
from os.path import dirname, isfile

import cv2
import numpy as np

from cam2vr.hskcamera import hskcamera
from cam2vr.pose.mediapipe_pose import MediaPipePose
from cam2vr.visualization.pose_overlay import draw_pose


ROOT = dirname(__file__).replace("\\", "/")
MODEL_PATH = f"{ROOT}/cam2vr/models/pose_landmarker_full.task"

CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -6

WINDOW_NAME = "Cam2VR - MediaPipe Pose"


class RealtimeMetrics:
    """
    Guarda pequeñas ventanas de muestras para calcular
    métricas recientes sin acumular memoria indefinidamente.
    """

    def __init__(self, maximum_samples: int = 120):
        self.camera_timestamps = deque(maxlen=maximum_samples)
        self.pose_timestamps = deque(maxlen=maximum_samples)
        self.inference_times_ms = deque(maxlen=maximum_samples)

        self.last_sequence = None
        self.dropped_frames = 0
        self.processed_frames = 0
        self.invalid_detections = 0

    def add_camera_frame(
        self,
        sequence: int,
        local_capture_timestamp: float,
    ) -> None:
        self.camera_timestamps.append(local_capture_timestamp)

        if self.last_sequence is not None:
            sequence_difference = sequence - self.last_sequence

            if sequence_difference > 1:
                self.dropped_frames += sequence_difference - 1

        self.last_sequence = sequence

    def add_pose_result(
        self,
        inference_ms: float,
        valid: bool,
    ) -> None:
        self.pose_timestamps.append(time.perf_counter())
        self.inference_times_ms.append(inference_ms)

        self.processed_frames += 1

        if not valid:
            self.invalid_detections += 1

    @staticmethod
    def _calculate_fps(timestamps: deque) -> float:
        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]

        if elapsed <= 0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    @property
    def camera_fps(self) -> float:
        return self._calculate_fps(self.camera_timestamps)

    @property
    def pose_fps(self) -> float:
        return self._calculate_fps(self.pose_timestamps)

    @property
    def average_inference_ms(self) -> float:
        if not self.inference_times_ms:
            return 0.0

        return float(np.mean(self.inference_times_ms))

    @property
    def p95_inference_ms(self) -> float:
        if not self.inference_times_ms:
            return 0.0

        return float(np.percentile(self.inference_times_ms, 95))

    @property
    def invalid_percentage(self) -> float:
        if self.processed_frames == 0:
            return 0.0

        return (
            self.invalid_detections
            / self.processed_frames
            * 100.0
        )


def draw_metrics(
    frame_rgb: np.ndarray,
    metrics: RealtimeMetrics,
    sequence: int,
    frame_age_ms: float,
    inference_ms: float,
    pose_valid: bool,
) -> np.ndarray:
    """
    Añade información de rendimiento al frame RGB.
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
        f"Pose valid: {pose_valid}",
    ]

    x = 15
    y = 25
    line_height = 23

    # Fondo oscuro para poder leer el texto.
    overlay = frame_rgb.copy()

    cv2.rectangle(
        overlay,
        (5, 5),
        (330, 5 + line_height * len(lines) + 8),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame_rgb,
        0.45,
        0,
        frame_rgb,
    )

    for index, line in enumerate(lines):
        color = (80, 255, 100)

        if line.startswith("Pose valid") and not pose_valid:
            color = (255, 80, 80)

        cv2.putText(
            frame_rgb,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame_rgb


def main() -> None:
    if not isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo: {MODEL_PATH}"
        )

    camera = hskcamera.Camera()
    pose_backend = None

    camera_started = False
    camera_opened = False

    metrics = RealtimeMetrics()

    # Se utilizará para convertir el timestamp de C++
    # al reloj monotónico empleado por time.perf_counter().
    timestamp_offset = None

    try:
        print("Abriendo cámara...")

        camera_opened = camera.open(
            camera_index=CAMERA_INDEX,
            format_index=FORMAT_INDEX,
            exposure=EXPOSURE,
            list_formats=False,
        )

        if not camera_opened:
            raise RuntimeError("No se pudo abrir la cámara")

        print(
            f"Cámara abierta: {camera.width} x {camera.height}"
        )

        camera_started = camera.start()

        if not camera_started:
            raise RuntimeError(
                "No se pudo iniciar la captura"
            )

        print("Cargando MediaPipe Pose Landmarker...")

        pose_backend = MediaPipePose(MODEL_PATH)

        print("Pipeline iniciado")
        print("Pulsa Q o ESC para cerrar")

        last_sequence = 0

        while True:
            result = camera.wait_for_next_frame(
                last_sequence,
                timeout_ms=1000,
            )

            if result is None:
                print("Esperando un frame nuevo...")
                continue

            frame_rgb, sequence, capture_timestamp = result

            # Guardamos la secuencia inmediatamente.
            last_sequence = sequence

            # Conversión a tipos Python normales por si pybind11
            # entrega enteros o floats propios de NumPy.
            sequence = int(sequence)
            capture_timestamp = float(capture_timestamp)

            # Calculamos una vez la diferencia entre ambos relojes.
            #
            # Si hskcamera ya usa el mismo origen que perf_counter(),
            # el desplazamiento será aproximadamente cero.
            if timestamp_offset is None:
                timestamp_offset = (
                    time.perf_counter()
                    - capture_timestamp
                )

            local_capture_timestamp = (
                capture_timestamp + timestamp_offset
            )

            metrics.add_camera_frame(
                sequence,
                local_capture_timestamp,
            )

            observation = pose_backend.process(
                frame_rgb=frame_rgb,
                sequence=sequence,
                capture_timestamp=capture_timestamp,
            )

            metrics.add_pose_result(
                inference_ms=observation.inference_ms,
                valid=observation.valid,
            )

            frame_age_ms = (
                time.perf_counter()
                - local_capture_timestamp
            ) * 1000.0

            debug_rgb = draw_pose(
                frame_rgb,
                observation,
            )

            debug_rgb = draw_metrics(
                frame_rgb=debug_rgb,
                metrics=metrics,
                sequence=sequence,
                frame_age_ms=frame_age_ms,
                inference_ms=observation.inference_ms,
                pose_valid=observation.valid,
            )

            # hskcamera entrega RGB, pero cv2.imshow espera BGR.
            #
            # Esta conversión es solamente para mostrar la imagen.
            # El frame enviado a MediaPipe permanece en RGB.
            debug_bgr = cv2.cvtColor(
                debug_rgb,
                cv2.COLOR_RGB2BGR,
            )

            cv2.imshow(WINDOW_NAME, debug_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        print("Cerrando Cam2VR...")

        if pose_backend is not None:
            pose_backend.close()

        if camera_started:
            camera.stop()

        if camera_opened:
            camera.close()

        cv2.destroyAllWindows()

        print("Recursos liberados")


if __name__ == "__main__":
    main()