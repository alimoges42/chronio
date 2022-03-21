import pathlib
import json
import inspect
from numpy import savetxt


class Convention:
    def __init__(self, directory: str, suffix: str, fields: list,
                 append_date: bool = False, overwrite: bool = False,
                 to_csv_args: dict = None, savetxt_args: dict = None):
        self.directory = directory
        self.fields = fields
        self.suffix = suffix
        self.append_date = append_date
        self.overwrite = overwrite
        self.to_csv_args = to_csv_args
        self.savetxt_args = savetxt_args

    def get_params(self):
        return self.__dict__

    def export_convention(self, path: str = None):
        if path:
            path = pathlib.Path(path)
        else:
            path = pathlib.Path(self.directory)
        fname = 'convention.json'

        path = pathlib.PurePath.joinpath(path, fname)

        print(path)

        to_save = self.__dict__

        with open(str(path), 'w') as write_file:
            json.dump(to_save, write_file)


def convention_from_template(template_path: str) -> Convention:
    json_data = json.load(open(template_path))
    convention = Convention(directory=json_data['directory'], suffix=json_data['suffix'], fields=json_data['fields'],
                            append_date=json_data['append_date'], overwrite=json_data['overwrite'],
                            to_csv_args=json_data['to_csv_args'], savetxt_args=json_data['save_text_args'])
    return convention

