#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Conversions

This submodule contains tools for converting frames to times and vice versa.

@author: Aaron Limoges
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


if __name__ == '__main__':
    frate = 5
    stamps = frames_to_times(fps=frate, frame_numbers=[2, 2, 5, 10, 13])
    print(stamps)

    frames = times_to_frames(fps=frate, timestamps=stamps)
    print(frames)
