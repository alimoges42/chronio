from typing import Any


class Observation:
    def __init__(self, **files):
        for key in files:
            print(key)
            setattr(self, key, files[key])


if __name__ == '__main__':
    spec = Observation(file1={'path': 'yo', 'obj': 'behavior'}, file2={'path': 'hey', 'obj': 'neuro'})
    print(spec.file2)