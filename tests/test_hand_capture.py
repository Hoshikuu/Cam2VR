from pathlib import Path
from sys import path

import cv2


FILE_DIRECTORY = Path(__file__).resolve().parent
ROOT = FILE_DIRECTORY.parent if FILE_DIRECTORY.name == "tests" else FILE_DIRECTORY

if str(ROOT) not in path:
    path.insert(0, str(ROOT))


from cam2vr.hands.mediapipe_hands import MediaPipeHands
from cam2vr.hskcamera import hskcamera
from cam2vr.visualization.hand_overlay import draw_hands


MODEL_PATH = ROOT / "cam2vr" / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -5
FRAME_TIMEOUT_MS = 1000
MAX_TIMEOUTS = 5

WINDOW_NAME = "Cam2VR - Hand Tracking"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

    camera = hskcamera.Camera()
    hands_backend = None
    camera_opened = False
    camera_started = False

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

        hands_backend = MediaPipeHands(str(MODEL_PATH))

        print(f"Camara abierta: {camera.width}x{camera.height}")
        print("Hand Tracking iniciado")
        print("Pulsa Q o ESC para cerrar")

        last_sequence = 0
        consecutive_timeouts = 0

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

            hands = hands_backend.process(
                frame_rgb=frame_rgb,
                sequence=sequence,
                capture_timestamp_100ns=capture_timestamp_100ns
            )

            debug_rgb = draw_hands(
                frame_rgb=frame_rgb,
                hands=hands,
                rotation_matrices=hands_backend.last_rotation_matrices
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
        if hands_backend is not None:
            hands_backend.close()

        if camera_started:
            camera.stop()

        if camera_opened:
            camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()