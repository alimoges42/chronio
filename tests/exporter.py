from pandas import DataFrame
from numpy import ndarray
from chronio.export.exporter import _ArrayExporter, _DataFrameExporter


if __name__ == '__main__':
    fields_ = ['Aaron', 'Limoges']
    directory_ = 'C://Users/limogesaw/Desktop'
    suffix_ = 'csv'
    data = {'Names': ['Aaron', 'Bob', 'Jackie'],
           'ages': [30, 24, 57]}

    d = _DataFrameExporter(obj=DataFrame(data=data),
                           directory=directory_,
                           suffix=suffix_,
                           fields=fields_, function_kwargs={})
    print(d.fpath)

    a = _ArrayExporter(obj=ndarray([3, 3]),
                       directory=directory_,
                       suffix=suffix_,
                       fields=fields_, function_kwargs={})
