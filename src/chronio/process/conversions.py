#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This submodule contains tools for converting frames to times and vice versa.

Author: Aaron Limoges
"""


def frames_to_times(fps: float, frame_numbers: list) -> list:
    """
    Evaluate time of a specific frame.
    :param fps: frame rate (frames per second)
    :param frame_numbers:
    :return: list of timestamps that each frame corresponds to.
    """

    return list(map(lambda x: x / fps, frame_numbers))


def times_to_frames(fps: float, timestamps: list) -> list:
    """
    Evaluate frame numbers at given times.
    :param fps: frame rate (frames per second)
    :param timestamps:
    :return: list of frames that each timestamp corresponds to.
    """

    return list(map(lambda x: int(x * fps), timestamps))
