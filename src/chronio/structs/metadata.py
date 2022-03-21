from typing import List as _List

from chronio.design.convention import Convention


class Metadata:
    def __init__(self,
                 # System metadata
                 fpath: str = None,

                 # Session metadata
                 stage: str = None,
                 fps: float = None,

                 # Window metadata
                 trial_type: str = None,
                 indices: _List[int] = None,
                 n_windows: int = None,

                 # Dictionary that will overwrite established values
                 meta_dict: dict = None
                 ):
        if not meta_dict:
            meta_dict = {}

        self.system = {'fpath': fpath}
        self.subject = {}
        self.session = {'fps': fps,
                        'stage': stage}
        self.window_params = {'indices': indices,
                              'n_windows': n_windows,
                              'trial_type': trial_type}
        self.pane_params = {}

        self.update(meta_dict=meta_dict)

    def set_val(self, group, value_dict):
        d = getattr(self, group)
        for key, value in value_dict.items():
            d[key] = value

    def update(self, meta_dict: dict):
        for group in list(self.__dict__.keys()):
            self.set_val(group, meta_dict)

    def export(self, convention: Convention, function_kwargs: dict = None, **exporter_kwargs):
        # TODO: Think about how metadata should be exported
        pass
