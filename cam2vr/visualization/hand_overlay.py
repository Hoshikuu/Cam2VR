import cv2

from mediapipe import tasks
from numpy import asarray, float64, ndarray

from cam2vr.hands.full_hands import HandObservation, HandPoint, HandsObservation
from cam2vr.hands.hand_map import HAND_POINT_COUNT, WRIST


def point_to_pixel(point: HandPoint, width: int, height: int):
    """Converts normalized coordinates to image coordinates
    """
    return int(point.x * width), int(point.y * height)


def draw_single_hand(frame_rgb: ndarray, hand: HandObservation):
    """Draws the points and connections of 1 hand
    """
    output = frame_rgb.copy()

    if not hand.valid or len(hand.points_2d) != HAND_POINT_COUNT:
        return output

    height, width = output.shape[:2]
    hand_color = (255, 120, 80) if hand.handedness == "Left" else (80, 180, 255)
    connections = tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        if connection.start >= len(hand.points_2d) or connection.end >= len(hand.points_2d):
            continue

        start = point_to_pixel(hand.points_2d[connection.start], width, height)
        end = point_to_pixel(hand.points_2d[connection.end], width, height)
        cv2.line(output, start, end, hand_color, 2, cv2.LINE_AA)

    for index, point in enumerate(hand.points_2d):
        pixel = point_to_pixel(point, width, height)
        radius = 6 if index == WRIST else 4
        cv2.circle(output, pixel, radius, hand_color, -1, cv2.LINE_AA)
        cv2.circle(output, pixel, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    palm_pixel = point_to_pixel(hand.palm, width, height)
    cv2.circle(output, palm_pixel, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(output, palm_pixel, 10, hand_color, 2, cv2.LINE_AA)
    cv2.putText(
        output,
        hand.handedness.upper(),
        (palm_pixel[0] + 12, palm_pixel[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        hand_color,
        2,
        cv2.LINE_AA
    )

    return output


def draw_hand_axes(
    frame_rgb: ndarray,
    hand: HandObservation,
    rotation_matrix,
    axis_length: float = 60.0
):
    """Draws the local palm axes
    """
    output = frame_rgb.copy()

    if not hand.valid or not hand.orientation_valid or rotation_matrix is None:
        return output

    rotation = asarray(rotation_matrix, dtype=float64)

    if rotation.shape != (3, 3):
        return output

    height, width = output.shape[:2]
    origin = point_to_pixel(hand.palm, width, height)

    def project_axis(axis):
        return (
            int(origin[0] + axis[0] * axis_length),
            int(origin[1] - axis[1] * axis_length)
        )

    cv2.line(output, origin, project_axis(rotation[:, 0]), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.line(output, origin, project_axis(rotation[:, 1]), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(output, origin, project_axis(rotation[:, 2]), (0, 0, 255), 3, cv2.LINE_AA)

    return output


def draw_hand_info(frame_rgb: ndarray, hands: HandsObservation):
    """Draws tracking information for both hands
    """
    output = frame_rgb.copy()
    lines = ["Cam2VR Hand Tracking", "", f"Inference: {hands.inference_ms:.2f} ms", ""]

    if hands.has_left:
        hand = hands.left
        lines.extend([
            "LEFT",
            f"Palm: {hand.x:.3f} {hand.y:.3f} {hand.z:.3f}",
            f"Yaw: {hand.yaw:7.2f}",
            f"Pitch: {hand.pitch:7.2f}",
            f"Roll: {hand.roll:7.2f}",
            f"Side score: {hand.handedness_score:.2f}",
            ""
        ])
    else:
        lines.extend(["LEFT: NOT FOUND", ""])

    if hands.has_right:
        hand = hands.right
        lines.extend([
            "RIGHT",
            f"Palm: {hand.x:.3f} {hand.y:.3f} {hand.z:.3f}",
            f"Yaw: {hand.yaw:7.2f}",
            f"Pitch: {hand.pitch:7.2f}",
            f"Roll: {hand.roll:7.2f}",
            f"Side score: {hand.handedness_score:.2f}"
        ])
    else:
        lines.append("RIGHT: NOT FOUND")

    line_height = 21
    panel = output.copy()
    cv2.rectangle(panel, (5, 5), (325, 15 + len(lines) * line_height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.60, output, 0.40, 0, output)

    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (15, 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


def draw_hands(frame_rgb: ndarray, hands: HandsObservation, rotation_matrices=None):
    """Draws the complete hand tracking overlay
    """
    output = frame_rgb.copy()
    rotation_matrices = rotation_matrices or {}

    if hands.has_left:
        output = draw_single_hand(output, hands.left)
        output = draw_hand_axes(output, hands.left, rotation_matrices.get("Left"))

    if hands.has_right:
        output = draw_single_hand(output, hands.right)
        output = draw_hand_axes(output, hands.right, rotation_matrices.get("Right"))

    return draw_hand_info(output, hands)