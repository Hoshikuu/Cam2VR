import cv2

from numpy import ndarray
from cam2vr.pose.full_pose import FullPose, PosePoint
from cam2vr.pose.pose_map import BODY_CONNECTIONS


def point_pixel(point: PosePoint, width: int, height: int):
    """Converts cam2vr points to image coordinates

    Args:
        point (Point): cam2vr Point
        width (int): image width
        height (int): image height

    Returns:
        tuple: return a tuple with the coordinates
    """
    x = int(point.x * width)
    y = int(point.y * height)
    return x, y

def point_drawable(point: PosePoint, min_visibility: float = 0.4):
    """Decides if 1 point needs to be drawn

    Args:
        point (Point): the point to inspect
        min_visibility (float, optional): min visibility to draw. Defaults to 0.4.

    Returns:
        bool: yes or no
    """
    if point.visibility < min_visibility:
        return False
    if not (-0.1 <= point.x <= 1.1):
        return False
    if not (-0.1 <= point.y <= 1.1):
        return False

    return True

def draw_line(fullpose: FullPose, start_index, end_index, width, height, output):
    start = fullpose.points_2d[start_index]
    end = fullpose.points_2d[end_index]

    if not point_drawable(start):
        return False

    if not point_drawable(end):
        return False

    start_pixel = point_pixel(start, width, height)
    end_pixel = point_pixel(end, width, height)

    cv2.line(
        output,
        start_pixel,
        end_pixel,
        (50, 220, 80),
        3,
        cv2.LINE_AA,
    )

    return True

def draw_point(point, width, height, output):
    if not point_drawable(point):
        return False

    pixel = point_pixel(point, width, height)

    cv2.circle(
        output,
        pixel,
        5,
        (255, 100, 40),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        output,
        pixel,
        7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return True

def draw_pose(frame_rgb: ndarray, fullpose: FullPose):
    output = frame_rgb.copy()

    if not fullpose.valid:
        cv2.putText(
            output,
            "Body not found",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 60, 60),
            2,
            cv2.LINE_AA,
        )

        return output

    if len(fullpose.points_2d) != 33:
        return output

    height, width = output.shape[:2]

    for start_index, end_index in BODY_CONNECTIONS:
        if not draw_line(fullpose, start_index, end_index, width, height, output):
            continue

    for point in fullpose.points_2d:
        if not draw_point(point, width, height, output):
            continue

    return output