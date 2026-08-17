from os.path import dirname, isfile

import cv2

from cam2vr.hskcamera import hskcamera

from cam2vr.head.mediapipe_head import (
    MediaPipeHead,
)

from cam2vr.visualization.head_overlay import (
    draw_head,
)


ROOT = dirname(
    dirname(__file__)
).replace(
    "\\",
    "/",
)

MODEL_PATH = (
    f"{ROOT}/Cam2VR/cam2vr/models/"
    "face_landmarker.task"
)


CAMERA_INDEX = 0

FORMAT_INDEX = 312

EXPOSURE = -5


WINDOW_NAME = (
    "Cam2VR - Head Tracking"
)


def main():
    if not isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontro el modelo: "
            f"{MODEL_PATH}"
        )

    camera = hskcamera.Camera()

    head_backend = None

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
                "No se pudo abrir la camara"
            )

        print(
            f"Camara abierta: "
            f"{camera.width}x"
            f"{camera.height}"
        )

        camera_started = camera.start()

        if not camera_started:
            raise RuntimeError(
                "No se pudo iniciar "
                "la captura"
            )

        print(
            "Cargando Face Landmarker..."
        )

        head_backend = MediaPipeHead(
            MODEL_PATH
        )

        print(
            "Head Tracking iniciado"
        )

        print(
            "Q / ESC para cerrar"
        )

        last_sequence = 0

        while True:
            result = (
                camera.wait_for_next_frame(
                    last_sequence=last_sequence,
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
                capture_timestamp_ns,
            ) = result

            sequence = int(
                sequence
            )

            capture_timestamp_ns = int(
                capture_timestamp_ns
            )

            last_sequence = sequence

            head = head_backend.process(
                frame_rgb=frame_rgb,
                sequence=sequence,
                capture_timestamp_ns=(
                    capture_timestamp_ns
                ),
            )

            debug_rgb = draw_head(
                frame_rgb=frame_rgb,
                head=head,

                face_landmarks=(
                    head_backend
                    .last_face_landmarks
                ),

                transformation_matrix=(
                    head_backend
                    .last_transformation_matrix
                ),
            )

            # Camera = RGB
            # OpenCV = BGR
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

        if head_backend is not None:
            head_backend.close()

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