from pathlib import Path
import time

import cv2

from cam2vr.pose.mediapipe_pose import MediaPipePoseBackend


ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
IMAGE_PATH = ROOT / "test_person.jpg"


image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(f"No se pudo abrir: {IMAGE_PATH}")

# OpenCV abre imágenes como BGR.
# MediaPipe espera RGB.
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

backend = MediaPipePoseBackend(MODEL_PATH)

try:
    observation = backend.process(
        frame_rgb=image_rgb,
        sequence=0,
        capture_timestamp=time.perf_counter(),
    )

    print(f"Detección válida: {observation.valid}")
    print(f"Inferencia: {observation.inference_ms:.2f} ms")
    print(f"Landmarks 2D: {len(observation.landmarks_2d)}")
    print(f"Landmarks 3D: {len(observation.landmarks_3d)}")

    if observation.valid:
        nose = observation.landmarks_2d[0]
        left_shoulder = observation.landmarks_2d[11]
        right_shoulder = observation.landmarks_2d[12]

        print()
        print("Nariz:")
        print(nose)

        print()
        print("Hombro izquierdo:")
        print(left_shoulder)

        print()
        print("Hombro derecho:")
        print(right_shoulder)

finally:
    backend.close()