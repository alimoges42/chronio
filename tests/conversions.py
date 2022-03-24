from chronio.process.conversions import *

if __name__ == '__main__':
    frate = 5
    stamps = frames_to_times(fps=frate, frame_numbers=[2, 2, 5, 10, 13])
    print(stamps)

    frames = times_to_frames(fps=frate, timestamps=stamps)
    print(frames)