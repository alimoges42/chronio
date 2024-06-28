from functools import wraps
import pandas as _pd


def handle_inplace(copy_func):
    @wraps(copy_func)
    def wrapper(self, *args, inplace=False, **kwargs):
        result = copy_func(self, *args, **kwargs)

        if inplace:
            if isinstance(result, _pd.DataFrame):
                self.data = result
            else:
                for key, value in result.items():
                    setattr(self, key, value)
            return None
        else:
            if isinstance(result, _pd.DataFrame):
                return self.__class__(data=result, metadata=self.metadata)
            else:
                new_obj = self.__class__(data=self.data.copy(), metadata=self.metadata)
                for key, value in result.items():
                    setattr(new_obj, key, value)
                return new_obj

    return wrapper
