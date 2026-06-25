import pathlib

from .slice_video import slice_video
from .maths import get_body_angle as gba, get_box_ratio


def get_app_root():
    return pathlib.Path(__file__).parent.parent.parent.resolve()


def get_body_angle(*key_points):
    return max([gba(*kp) for kp in key_points])
