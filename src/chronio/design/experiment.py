import os
import pathlib
import json


class Stage:
    def __init__(self, stage: str, stage_duration: float, trial_onsets: dict, trial_durations: dict):
        self.stage = stage
        self.stage_duration = stage_duration
        self.trial_onsets = trial_onsets
        self.trials = trial_onsets.keys()
        self.trial_durations = trial_durations

        if trial_onsets.keys() != trial_durations.keys():
            raise ValueError('Keys of trial_onsets and trial_durations do not match')

    def __repr__(self):
        return f'{self.__class__.__name__} object of {len(self.trial_onsets.keys())} cues'

    def __str__(self):
        return f'{self.__class__.__name__}(num_cues={len(self.trial_onsets.keys())})'

    def add_trial_type(self, trial_name: str, trial_onsets: list, trial_duration: float):
        self.trial_onsets[trial_name] = trial_onsets
        self.trial_durations[trial_name] = trial_duration

    def remove_trial_type(self, trial_name: str):
        if trial_name in self.trial_onsets.keys():
            del self.trial_onsets[trial_name]
            del self.trial_durations[trial_name]
            print(f'Trial type {trial_name} deleted.')

        else:
            raise ValueError(f'Trial type {trial_name} not found.')

    def export_template(self, path: str = os.getcwd()):
        path = pathlib.Path(path)
        fname = f'{self.stage}.json'
        path = pathlib.PurePath.joinpath(path, fname)
        print(path)
        to_save = {'duration': self.stage_duration,
                   'stage': self.stage,
                   'trial_onsets': self.trial_onsets,
                   'trial_durations': self.trial_durations}

        with open(str(path), 'w') as write_file:
            json.dump(to_save, write_file)


def stage_from_template(template_path: str) -> Stage:
    json_data = json.load(open(template_path))
    stage = Stage(stage=json_data['stage'], stage_duration=json_data['stage_duration'],
                  trial_onsets=json_data['trial_onsets'], trial_durations=json_data['trial_durations'])
    return stage
