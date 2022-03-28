import pandas as _pd

from chronio.structs.structs import _Structure
from chronio.design.convention import Convention as _Convention

if __name__ == '__main__':
    test = _Structure(data=_pd.DataFrame())
    test.export(convention=_Convention(directory='//', suffix='csv', append_date=True, overwrite=False, fields=['name']),
                directory='//aaron/')

    test.export(convention=_Convention(directory='//', suffix='csv', append_date=True, overwrite=False, fields=['name']),
                function_kwargs={'header': False}, fields=['name', 'date'], overwrite=False,
                directory='/home/aaron', suffix='csv')
