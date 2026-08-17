from pathlib import Path
from sys import path
from time import perf_counter

import cv2


FILE_DIRECTORY = Path(__file__).resolve().parent
ROOT = FILE_DIRECTORY.parent if FILE_DIRECTORY.name == "tests" else FILE_DIRECTORY

if str(ROOT) not in path:
    path.insert(0, str(ROOT))


from cam2vr.hskcamera import hskcamera


CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -5
FRAME_TIMEOUT_MS = 1000
MAX_TIMEOUTS = 5

WINDOW_NAME = "Cam2VR - Camera"


def main():
    camera = hskcamera.Camera()
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

        print(f"Camara abierta: {camera.width}x{camera.height}")
        print("Pulsa Q o ESC para cerrar")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        last_sequence = 0
        consecutive_timeouts = 0
        fps = 0.0
        frame_count = 0
        fps_start = perf_counter()

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
            frame_rgb, sequence, _ = result
            sequence = int(sequence)
            last_sequence = sequence

            frame_count += 1
            elapsed = perf_counter() - fps_start

            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = perf_counter()

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(
                frame_bgr,
                f"FPS: {fps:.1f} | Sequence: {sequence}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow(WINDOW_NAME, frame_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("\nCaptura interrumpida")

    finally:
        if camera_started:
            camera.stop()

        if camera_opened:
            camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()