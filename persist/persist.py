from functools import wraps
from dask import get
from dask.optimize import cull
import inspect

from dask import threaded
from dask.base import Base
from dask.base import collections_to_dsk
from dask import delayed

__all__ = ['PersistentDAG']


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
                      if arg_value in funcs.keys()})
    # set list of arguments
    args_tuple = tuple([args_dict.pop(argname) for argname in args_spec.args])
    if args_spec.varargs:
        args_tuple += args_dict.pop(args_spec.varargs)
    return args_tuple, args_dict


def get_relevant_keys_from_on_disk_cache(dsk, serializers):
    # the fact to call the method "is_computed" may slow down the code.
    dsk_serialized = {key: (serializers[key].delayed_load(key),)
                      for key in dsk.keys()
                      if key in serializers and
                      serializers[key].is_computed(key)}
    return dsk_serialized


def get_relevant_keys_from_memory_cache(dsk, cache):
    dsk_cached = {key: cache[key]
                  for key in dsk.keys() if key in cache}
    return dsk_cached


def persistent_collections_to_dsk(collections,
                                  key=None, serializers=None, cache=None,
                                  *args, **kwargs):

    dsk = collections_to_dsk(collections, *args, **kwargs)

    if key is not None:
        dsk, _ = cull(dsk, key)

    if serializers is not None:
        # load instead of compute
        dsk_serialized = get_relevant_keys_from_on_disk_cache(dsk, serializers)
        dsk.update(dsk_serialized)

    if cache is not None:
        # use cache instead of loadind
        dsk_cached = get_relevant_keys_from_memory_cache(dsk, cache)
        dsk.update(dsk_cached)

    # filter again task after function have been replaced by load or values
    if key is not None:
        dsk, _ = cull(dsk, key)

    return dsk


class PersistentDAG(Base):
    __slots__ = ('dask', '_keys')
    _finalize = staticmethod(list)
    _default_get = staticmethod(threaded.get)
    _optimize = staticmethod(lambda d, k, **kwds: d)

    def __init__(self, use_cluster=False):
        self.cache = dict()
        self.serializer = dict()
        if use_cluster:
            from distributed import LocalCluster, Client
            self.cluster = LocalCluster()
            self.client = Client(self.cluster)
        else:
            self.cluser = None
            self.client = None
        self.funcs = dict()
        self.dask = dict()

    def _keys(self):
        return self.funcs.keys()

    def delayed(self, func, key=None, serializer=None):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            self.add_task(key, serializer, func, *args, **kwargs)
        return wrapped_func

    def submit(self, func, *args, **kwargs):
        """
        submit func to the graph.
        Special keyword arguments are:
        - dask_key_name
        - dask_serializer
        """
        key = kwargs.pop('dask_key_name')
        serializer = kwargs.pop('dask_serializer')
        return self.add_task(key, serializer, func, *args, **kwargs)

    def add_task(self, key, serializer, func, *args, **kwargs):
        # prepare arguments
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
            keys = delayed_func.dask.keys()
            assert len(keys) == 1
            key = keys[0]
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

    def get_persistent_dask(self, key=None, *args, **kwargs):
        collections = self.funcs.values()
        return persistent_collections_to_dsk(
            collections, key, self.serializer, self.cache, *args, **kwargs)

    @property
    def persistent_dask(self):
        return self.get_persistent_dask()

    def is_computed(self):
        return {key: self.serializer[key].is_computed(key)
                for key in self.dask.keys() if key in self.serializer}

    def get(self, key):
        """
        Wrapper around dask.get.
        Use cache or serialzed data if available.
        """
        dsk = self.get_persistent_dask(key)
        # get result
        if self.client:
            result = self.client.get(dsk, key)
        else:
            result = get(dsk, key)
        # store in cache
        try:
            self.cache.update({key: result})
        except TypeError:
            self.cache.update(dict(zip(key, result)))
            result = list(result)
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
