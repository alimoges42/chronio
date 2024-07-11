from functools import wraps
import pandas as _pd


def handle_inplace(copy_func):
    @wraps(copy_func)
    def wrapper(self, *args, inplace=False, **kwargs):
        result = copy_func(self, *args, **kwargs)

        if inplace:
            if isinstance(result, _pd.DataFrame):
                self.data = result
                self._update_fps()
            else:
                for key, value in result.items():
                    setattr(self, key, value)
                self._update_fps()
            return None
        else:
            if isinstance(result, _pd.DataFrame):
                new_obj = self.__class__(data=result, metadata=self.metadata.copy())
                new_obj._update_fps()
                return new_obj
            else:
                new_obj = self.__class__(data=self.data.copy(), metadata=self.metadata.copy())
                for key, value in result.items():
                    setattr(new_obj, key, value)
                new_obj._update_fps()
                return new_obj

    return wrapper
