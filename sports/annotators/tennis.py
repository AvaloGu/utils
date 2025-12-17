from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from sports.configs.tennis import TennisCourtConfiguration

import numpy as np
import cv2
import supervision as sv

def draw_tennis_court(
    config,
    outer_color: sv.Color = sv.Color(90, 130, 115),   # green surround 
    court_color: sv.Color = sv.Color(95, 110, 150),   # bluish court fill
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 200,
    line_thickness: int = 1,
    point_radius: int = 5,
    scale: float = 0.5,
    draw_points: bool = True,
) -> np.ndarray:
    """
    Draw a tennis court (vertical) using config.vertices (cm) + config.edges (1-based indices).
    Produces a hard-court style image: green surround + blue court + white lines.
    """

    scaled_w = int(config.width * scale)
    scaled_l = int(config.length * scale)

    # background (greenish surround)
    img = np.ones((scaled_l + 2 * padding, scaled_w + 2 * padding, 3), dtype=np.uint8)
    img[:] = np.array(outer_color.as_bgr(), dtype=np.uint8)

    def to_px(x_cm: float, y_cm: float) -> tuple[int, int]:
        # image coordinates are (col=x, row=y)
        # add padding since we want a 'image origin' shift for better display (move origin away from top-left)
        x = int(x_cm * scale) + padding
        y = int(y_cm * scale) + padding
        return (x, y)

    # fill the court surface in bluish us open color
    court_poly = np.array([
        to_px(0.0, 0.0),
        to_px(config.width, 0.0),
        to_px(config.width, config.length),
        to_px(0.0, config.length),
    ], dtype=np.int32)
    cv2.fillPoly(img, [court_poly], color=court_color.as_bgr())

    # draw the white court lines from edges
    verts = config.vertices  # list of (x_cm, y_cm) in the vertices ordering
    for start, end in config.edges: # edges are 1-based indices
        x1, y1 = verts[start - 1]
        x2, y2 = verts[end - 1]
        cv2.line(
            img,
            to_px(x1, y1),
            to_px(x2, y2),
            color=line_color.as_bgr(),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    # net line (across doubles width)
    yN = config.net_y
    cv2.line(
        img,
        to_px(0.0, yN),
        to_px(config.width, yN),
        color=line_color.as_bgr(),
        thickness=max(1, line_thickness - 1),
        lineType=cv2.LINE_AA,
    )

    # draw keypoints
    # if draw_points:
    #     for (x_cm, y_cm) in verts:
    #         cv2.circle(img, to_px(x_cm, y_cm), point_radius, (255, 105, 180), -1, lineType=cv2.LINE_AA)

    return img


def draw_points_on_court(
    config: TennisCourtConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 200,
    scale: float = 0.5,
    court: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a tennic court diagram.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if court is None:
        court = draw_tennis_court(
            config=config,
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1 # the circle is drawn as a solid, filled disk
        )
        cv2.circle(
            img=court,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness # draws only the circumference or edge of the circle
        )

    return court