
from functools import wraps
from dask import get
import inspect
from dask import threaded
from dask.base import Base
from dask.base import collections_to_dsk
from dask import delayed

__all__ = ['DAG']


def prepare_args(func, args, kwargs, funcs):
    """
    prepare arguments of the given func.
    If some arguments correspond to keys in the dict of delayed funcs,
    they are replaced by it.
    """
    args_dict = inspect.getcallargs(func, *args, **kwargs)
    args_spec = inspect.getargspec(func)
    if args_spec.keywords:
        kwds = args_dict.pop(args_spec.keywords)
        args_dict.update(kwds)
    args_dict.update({arg_name: funcs[arg_value]
                      for arg_name, arg_value in args_dict.iteritems()
                      if not isinstance(arg_value, Base) and
                      arg_value in funcs.keys()})
    # set list of arguments
    args_tuple = tuple([args_dict.pop(argname) for argname in args_spec.args])
    if args_spec.varargs:
        args_tuple += args_dict.pop(args_spec.varargs)
    return args_tuple, args_dict


class DAG(Base):
    __slots__ = ('dask', '_keys')
    _finalize = staticmethod(list)
    _default_get = staticmethod(threaded.get)
    _optimize = staticmethod(lambda d, k, **kwds: d)

    def __init__(self, use_cluster=False):
        self.funcs = dict()
        self.dask = dict()

    def _keys(self):
        return self.funcs.keys()

    def delayed(self, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return self.add_task(func, *args, **kwargs)
        return wrapped_func

    def submit(self, func, *args, **kwargs):
        """
        add_task to the graph and persist
        """
        func = self.add_task(func, *args, **kwargs)
        return func.persist()

    def add_task(self, func, *args, **kwargs):
        """
        Special keyword arguments are:
        - dask_key_name
        - dask_serializer
        """
        key = kwargs.pop('dask_key_name', None)
        serializer = kwargs.pop('dask_serializer', None)

        # Prepare arguments
        args_tuple, args_dict = prepare_args(func, args, kwargs, self.funcs)

        # wrap func in order that it dump data as a side-effect
        if serializer is not None:
            func = serializer.dump_result(func, key)

        # use dask delayed collection to wrap functions
        delayed_func = delayed(func, pure=True)(
            dask_key_name=key, *args_tuple, **args_dict)

        # set key
        if key is None:
            # use tokenize key named setted by delayed
            # in this case dask manage correctly unicity of keys
            key = delayed_func._key
        else:
            assert key not in self.dask, "key is already used"

        # store func and serializer
        self.funcs[key] = delayed_func
        if serializer is not None:
            self.serializer[key] = serializer

        # update state
        self.dask = self._dask
        return delayed_func

    @property
    def _dask(self):
        dask = collections_to_dsk(self.funcs.values())
        return dask

    def get(self, key):
        """
        Wrapper around dask.get.
        Use cache or serialzed data if available.
        """
        dsk = self.dask
        # get result
        result = get(dsk, key)
        # store in cache
        return result

    def run(self, key=None):
        if key is None:
            key = self.dask.keys()
        result = self.get(key)
        if isinstance(key, list):
            return dict(zip(key, result))
        else:
            return result

    def async_run(self, key=None):
        if key is None:
            key = self.dask.keys()
        if not isinstance(key, list):
            return self.funcs[key].persist()
        else:
            futures = [self.funcs[k].persist() for k in key]
            return futures
