import cv2
import numpy as np

from mediapipe import tasks

from cam2vr.head.full_head import FullHead


NOSE_LANDMARK_INDEX = 1


def landmark_to_pixel(
    landmark,
    width: int,
    height: int,
) -> tuple[int, int]:
    """
    Converts a normalized MediaPipe landmark into pixels.
    """

    x = int(
        landmark.x * width
    )

    y = int(
        landmark.y * height
    )

    return x, y


def draw_face_landmarks(
    frame_rgb: np.ndarray,
    face_landmarks,
) -> np.ndarray:
    """
    Draws facial contours and all detected landmarks.
    """

    output = frame_rgb.copy()

    if not face_landmarks:
        return output

    height, width = output.shape[:2]

    # Official MediaPipe facial contour connections.
    connections = (
        tasks.vision
        .FaceLandmarksConnections
        .FACE_LANDMARKS_CONTOURS
    )

    # Draw connections first.
    for connection in connections:
        if (
            connection.start >= len(face_landmarks)
            or connection.end >= len(face_landmarks)
        ):
            continue

        start = face_landmarks[
            connection.start
        ]

        end = face_landmarks[
            connection.end
        ]

        start_pixel = landmark_to_pixel(
            start,
            width,
            height,
        )

        end_pixel = landmark_to_pixel(
            end,
            width,
            height,
        )

        cv2.line(
            output,
            start_pixel,
            end_pixel,

            # RGB cyan-ish.
            (40, 220, 220),

            1,
            cv2.LINE_AA,
        )

    # Draw every detected point.
    for landmark in face_landmarks:
        x, y = landmark_to_pixel(
            landmark,
            width,
            height,
        )

        if (
            x < 0
            or y < 0
            or x >= width
            or y >= height
        ):
            continue

        cv2.circle(
            output,
            (x, y),
            1,

            # RGB green.
            (80, 255, 100),

            -1,
            cv2.LINE_AA,
        )

    return output


def draw_head_axes(
    frame_rgb: np.ndarray,
    face_landmarks,
    transformation_matrix: np.ndarray,
    axis_length: float = 80.0,
) -> np.ndarray:
    """
    Draws XYZ orientation axes starting at the nose.

    Red   = X
    Green = Y
    Blue  = Z
    """

    output = frame_rgb.copy()

    if face_landmarks is None:
        return output

    if transformation_matrix is None:
        return output

    if len(face_landmarks) <= NOSE_LANDMARK_INDEX:
        return output

    height, width = output.shape[:2]

    nose = face_landmarks[
        NOSE_LANDMARK_INDEX
    ]

    origin_x, origin_y = landmark_to_pixel(
        nose,
        width,
        height,
    )

    origin = (
        origin_x,
        origin_y,
    )

    rotation = np.asarray(
        transformation_matrix[:3, :3],
        dtype=np.float64,
    )

    # Unit vectors for local head coordinates.
    x_axis_3d = np.array(
        [1.0, 0.0, 0.0]
    )

    y_axis_3d = np.array(
        [0.0, 1.0, 0.0]
    )

    z_axis_3d = np.array(
        [0.0, 0.0, 1.0]
    )

    # Rotate them according to the detected head.
    x_rotated = (
        rotation @ x_axis_3d
    )

    y_rotated = (
        rotation @ y_axis_3d
    )

    z_rotated = (
        rotation @ z_axis_3d
    )

    def project_axis(
        axis: np.ndarray,
    ) -> tuple[int, int]:
        """
        Simple orthographic projection.

        Image Y grows downwards, hence the minus sign.
        """

        x = int(
            origin_x
            + axis[0] * axis_length
        )

        y = int(
            origin_y
            - axis[1] * axis_length
        )

        return x, y

    x_endpoint = project_axis(
        x_rotated
    )

    y_endpoint = project_axis(
        y_rotated
    )

    z_endpoint = project_axis(
        z_rotated
    )

    # X axis - RED
    cv2.line(
        output,
        origin,
        x_endpoint,
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # Y axis - GREEN
    cv2.line(
        output,
        origin,
        y_endpoint,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    # Z axis - BLUE
    cv2.line(
        output,
        origin,
        z_endpoint,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        origin,
        5,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        x_endpoint,
        4,
        (255, 0, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        y_endpoint,
        4,
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        z_endpoint,
        4,
        (0, 0, 255),
        -1,
        cv2.LINE_AA,
    )

    return output


def draw_head_info(
    frame_rgb: np.ndarray,
    head: FullHead,
    landmark_count: int = 0,
) -> np.ndarray:
    """
    Draws head tracking information.
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
            cv2.LINE_AA,
        )

        return output

    lines = [
        "Cam2VR Head Tracking",
        "",
        f"Position X: {head.x:8.2f}",
        f"Position Y: {head.y:8.2f}",
        f"Position Z: {head.z:8.2f}",
        "",
        f"Yaw:   {head.yaw:8.2f} deg",
        f"Pitch: {head.pitch:8.2f} deg",
        f"Roll:  {head.roll:8.2f} deg",
        "",
        f"Landmarks: {landmark_count}",
        f"Sequence: {head.sequence}",
    ]

    x = 15
    y = 28

    line_height = 22

    overlay = output.copy()

    panel_height = (
        15
        + line_height * len(lines)
    )

    cv2.rectangle(
        overlay,
        (5, 5),
        (315, panel_height),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.60,
        output,
        0.40,
        0,
        output,
    )

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (
                x,
                y + index * line_height,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def draw_head(
    frame_rgb: np.ndarray,
    head: FullHead,
    face_landmarks=None,
    transformation_matrix=None,
) -> np.ndarray:
    """
    Complete head debugging visualization.
    """

    output = frame_rgb.copy()

    if not head.valid:
        return draw_head_info(
            output,
            head,
        )

    output = draw_face_landmarks(
        output,
        face_landmarks,
    )

    output = draw_head_axes(
        output,
        face_landmarks,
        transformation_matrix,
    )

    landmark_count = (
        len(face_landmarks)
        if face_landmarks
        else 0
    )

    output = draw_head_info(
        output,
        head,
        landmark_count,
    )

    return output