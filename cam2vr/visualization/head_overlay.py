import cv2

from mediapipe import tasks
from numpy import asarray, float64, ndarray

from cam2vr.head.full_head import FullHead
from cam2vr.head.head_map import NOSE_TIP


def point_to_pixel(point, width: int, height: int):
    """Converts normalized coordinates to image coordinates
    """
    return int(point.x * width), int(point.y * height)


def draw_face_landmarks(frame_rgb: ndarray, face_landmarks):
    """Draws the facial points and contour connections
    """
    output = frame_rgb.copy()

    if not face_landmarks:
        return output

    height, width = output.shape[:2]
    connections = tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

    for connection in connections:
        if connection.start >= len(face_landmarks) or connection.end >= len(face_landmarks):
            continue

        start = point_to_pixel(face_landmarks[connection.start], width, height)
        end = point_to_pixel(face_landmarks[connection.end], width, height)
        cv2.line(output, start, end, (40, 220, 220), 1, cv2.LINE_AA)

    for point in face_landmarks:
        x, y = point_to_pixel(point, width, height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(output, (x, y), 1, (80, 255, 100), -1, cv2.LINE_AA)

    return output


def draw_head_axes(
    frame_rgb: ndarray,
    face_landmarks,
    transformation_matrix,
    axis_length: float = 80.0
):
    """Draws the local head axes from the nose
    """
    output = frame_rgb.copy()

    if not face_landmarks or transformation_matrix is None:
        return output

    if len(face_landmarks) <= NOSE_TIP:
        return output

    transformation_matrix = asarray(transformation_matrix, dtype=float64)

    if transformation_matrix.shape != (4, 4):
        return output

    height, width = output.shape[:2]
    origin = point_to_pixel(face_landmarks[NOSE_TIP], width, height)
    rotation = transformation_matrix[:3, :3]

    def project_axis(axis):
        return (
            int(origin[0] + axis[0] * axis_length),
            int(origin[1] - axis[1] * axis_length)
        )

    x_endpoint = project_axis(rotation[:, 0])
    y_endpoint = project_axis(rotation[:, 1])
    z_endpoint = project_axis(rotation[:, 2])

    cv2.line(output, origin, x_endpoint, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(output, origin, y_endpoint, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(output, origin, z_endpoint, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.circle(output, origin, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(output, x_endpoint, 4, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(output, y_endpoint, 4, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(output, z_endpoint, 4, (0, 0, 255), -1, cv2.LINE_AA)

    return output


def draw_head_info(frame_rgb: ndarray, head: FullHead, landmark_count: int = 0):
    """Draws head tracking information
    """
    output = frame_rgb.copy()

    if not head.valid:
        cv2.putText(
            output,
            "HEAD NOT FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 60, 60),
            2,
            cv2.LINE_AA
        )
        return output

    lines = [
        "Cam2VR Head Tracking",
        "",
        f"Inference: {head.inference_ms:.2f} ms",
        "",
        f"Position X: {head.x:8.2f}",
        f"Position Y: {head.y:8.2f}",
        f"Position Z: {head.z:8.2f}",
        "",
        f"Yaw: {head.yaw:8.2f} deg",
        f"Pitch: {head.pitch:8.2f} deg",
        f"Roll: {head.roll:8.2f} deg",
        "",
        f"Landmarks: {landmark_count}",
        f"Sequence: {head.sequence}"
    ]

    line_height = 22
    panel = output.copy()
    cv2.rectangle(panel, (5, 5), (315, 15 + len(lines) * line_height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.60, output, 0.40, 0, output)

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (15, 28 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


def draw_head(frame_rgb: ndarray, head: FullHead, face_landmarks=None, transformation_matrix=None):
    """Draws the complete head tracking overlay
    """
    output = frame_rgb.copy()

    if not head.valid:
        return draw_head_info(output, head)

    output = draw_face_landmarks(output, face_landmarks)
    output = draw_head_axes(output, face_landmarks, transformation_matrix)
    landmark_count = len(face_landmarks) if face_landmarks else 0

    return draw_head_info(output, head, landmark_count)