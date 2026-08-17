from pathlib import Path
from sys import path

import cv2


FILE_DIRECTORY = Path(__file__).resolve().parent
ROOT = FILE_DIRECTORY.parent if FILE_DIRECTORY.name == "tests" else FILE_DIRECTORY

if str(ROOT) not in path:
    path.insert(0, str(ROOT))


from cam2vr.pose.mediapipe_pose import MediaPipePose
from cam2vr.pose.pose_map import LEFT_SHOULDER, NOSE, RIGHT_SHOULDER


MODEL_PATH = ROOT / "cam2vr" / "models" / "pose_landmarker_full.task"
IMAGE_PATH = FILE_DIRECTORY / "test_person.jpg"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {MODEL_PATH}")

    image_bgr = cv2.imread(str(IMAGE_PATH))

    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pose_backend = MediaPipePose(str(MODEL_PATH))

    try:
        pose = pose_backend.process(
            frame_rgb=image_rgb,
            sequence=0,
            capture_timestamp_100ns=0
        )

        print(f"Deteccion valida: {pose.valid}")
        print(f"Inferencia: {pose.inference_ms:.2f} ms")
        print(f"Puntos 2D: {len(pose.points_2d)}")
        print(f"Puntos 3D: {len(pose.points_3d)}")

        if pose.valid:
            print(f"Nariz: {pose.points_2d[NOSE]}")
            print(f"Hombro izquierdo: {pose.points_2d[LEFT_SHOULDER]}")
            print(f"Hombro derecho: {pose.points_2d[RIGHT_SHOULDER]}")

    finally:
        pose_backend.close()


if __name__ == "__main__":
    main()