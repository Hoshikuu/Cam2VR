from pathlib import Path
from sys import path


FILE_DIRECTORY = Path(__file__).resolve().parent
ROOT = FILE_DIRECTORY.parent if FILE_DIRECTORY.name == "tests" else FILE_DIRECTORY

if str(ROOT) not in path:
    path.insert(0, str(ROOT))


from cam2vr.hskcamera import hskcamera


CAMERA_INDEX = 0
FORMAT_INDEX = 312
EXPOSURE = -5
FRAME_TIMEOUT_MS = 1000
FRAME_COUNT = 20


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
        print(f"Capturando: {camera.is_capturing}")

        last_sequence = 0

        for index in range(FRAME_COUNT):
            result = camera.wait_for_next_frame(
                last_sequence=last_sequence,
                timeout_ms=FRAME_TIMEOUT_MS
            )

            if result is None:
                raise RuntimeError("Timeout esperando frame")

            frame_rgb, sequence, capture_timestamp_100ns = result
            sequence = int(sequence)
            capture_timestamp_100ns = int(capture_timestamp_100ns)
            last_sequence = sequence

            print(
                f"Frame {index + 1}: "
                f"shape={frame_rgb.shape}, "
                f"dtype={frame_rgb.dtype}, "
                f"sequence={sequence}, "
                f"timestamp_100ns={capture_timestamp_100ns}"
            )

    finally:
        if camera_started:
            camera.stop()

        if camera_opened:
            camera.close()


if __name__ == "__main__":
    main()