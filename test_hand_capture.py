from pathlib import Path
import sys

import cv2


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from cam2vr.hskcamera import (
    hskcamera,
)

from cam2vr.hands.mediapipe_hands import (
    MediaPipeHands,
)

from cam2vr.visualization.hand_overlay import (
    draw_hands,
)


MODEL_PATH = (
    ROOT
    / "Cam2VR"
    / "cam2vr"
    / "models"
    / "hand_landmarker.task"
)


CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -5


WINDOW_NAME = (
    "Cam2VR - Hand Tracking"
)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "No se encontro el modelo: "
            f"{MODEL_PATH}"
        )

    camera = (
        hskcamera.Camera()
    )

    hands_backend = None

    camera_opened = False
    camera_started = False

    try:
        print(
            "Abriendo camara..."
        )

        camera_opened = camera.open(
            camera_index=CAMERA_INDEX,
            format_index=FORMAT_INDEX,
            exposure=EXPOSURE,
            list_formats=False,
        )

        if not camera_opened:
            raise RuntimeError(
                "No se pudo abrir "
                "la camara"
            )

        print(
            "Camara abierta: "
            f"{camera.width}x"
            f"{camera.height}"
        )

        camera_started = (
            camera.start()
        )

        if not camera_started:
            raise RuntimeError(
                "No se pudo iniciar "
                "la captura"
            )

        print(
            "Cargando MediaPipe "
            "Hand Landmarker..."
        )

        hands_backend = (
            MediaPipeHands(
                str(MODEL_PATH)
            )
        )

        print(
            "Hand Tracking iniciado"
        )

        print(
            "Muestra una o ambas manos"
        )

        print(
            "Pulsa Q o ESC para cerrar"
        )

        last_sequence = 0

        while True:
            result = (
                camera
                .wait_for_next_frame(
                    last_sequence=(
                        last_sequence
                    ),
                    timeout_ms=1000,
                )
            )

            if result is None:
                print(
                    "Timeout esperando frame"
                )

                continue

            (
                frame_rgb,
                sequence,
                capture_timestamp_100ns,
            ) = result

            sequence = int(
                sequence
            )

            capture_timestamp_100ns = int(
                capture_timestamp_100ns
            )

            last_sequence = (
                sequence
            )

            hands = (
                hands_backend.process(
                    frame_rgb=frame_rgb,

                    sequence=sequence,

                    capture_timestamp_100ns=(
                        capture_timestamp_100ns
                    ),
                )
            )

            debug_rgb = draw_hands(
                frame_rgb=frame_rgb,

                hands=hands,

                rotation_matrices=(
                    hands_backend
                    .last_rotation_matrices
                ),
            )

            # hskcamera -> RGB
            # OpenCV     -> BGR
            debug_bgr = cv2.cvtColor(
                debug_rgb,
                cv2.COLOR_RGB2BGR,
            )

            cv2.imshow(
                WINDOW_NAME,
                debug_bgr,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if (
                key == ord("q")
                or key == 27
            ):
                break

    finally:
        print(
            "Cerrando Cam2VR..."
        )

        if hands_backend is not None:
            hands_backend.close()

        if camera_started:
            camera.stop()

        if camera_opened:
            camera.close()

        cv2.destroyAllWindows()

        print(
            "Recursos liberados"
        )


if __name__ == "__main__":
    main()