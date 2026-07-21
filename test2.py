import time

import cv2
from HSKCamera import *


CAMERA_INDEX = 1
FORMAT_INDEX = 312
EXPOSURE = -6
LIST_FORMATS = False
FRAME_TIMEOUT_MS = 1000

WINDOW_NAME = "Camera - Captura en tiempo real"


def main() -> None:
    camera = hskcamera.Camera()
    last_sequence = 0

    try:
        opened = camera.open(
            camera_index=CAMERA_INDEX,
            format_index=FORMAT_INDEX,
            exposure=EXPOSURE,
            list_formats=LIST_FORMATS,
        )

        if not opened:
            raise RuntimeError("No se pudo abrir la cámara.")

        if not camera.start():
            raise RuntimeError("No se pudo iniciar la captura.")

        print("Cámara abierta correctamente.")
        print(f"Resolución activa: {camera.width}x{camera.height}")
        print("Pulsa Q o Esc para cerrar.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        fps = 0.0
        frames_since_update = 0
        fps_update_start = time.perf_counter()

        while True:
            result = camera.wait_for_next_frame(
                last_sequence,
                timeout_ms=FRAME_TIMEOUT_MS,
            )

            if result is None:
                print("Timeout: no se recibió un frame nuevo.")
                continue

            rgb_frame, sequence, timestamp = result
            last_sequence = sequence

            # hskcamera entrega RGB, mientras que OpenCV muestra BGR.
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            frames_since_update += 1
            now = time.perf_counter()
            elapsed = now - fps_update_start

            if elapsed >= 0.5:
                fps = frames_since_update / elapsed
                frames_since_update = 0
                fps_update_start = now

            cv2.putText(
                bgr_frame,
                f"FPS: {fps:.1f} | Secuencia: {sequence}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, bgr_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            # Permite cerrar desde la X de la ventana.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nCaptura interrumpida por el usuario.")

    finally:
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()
        print("Cámara cerrada correctamente.")


if __name__ == "__main__":
    main()