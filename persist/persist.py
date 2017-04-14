from functools import wraps
from dask.optimize import cull
from dask.base import collections_to_dsk
from dask.base import visualize
from dask.delayed import Delayed
from .dag import DAG
from .dag import dask_to_collections

__all__ = ['PersistentDAG']


DOT_STATUS = {
    'computed': dict(style='filled', color='green'),
    'not_computed': dict(style='filled', color='red'),
    'pending': dict(style='filled', color='grey'),
}


def get_relevant_keys_from_on_disk_cache(dsk, serializers):
    # the fact to call the method "is_computed" may slow down the code.
    dsk_serialized = {key: (delayed_load(serializers[key].load, key),)
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
    """
    wrapper arount dask.base.collections_to_dsk
    *args and **kwargs are passed to collections_to_dsk
    """

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


def delayed_using_cache(delayed, serializers=None, cache=None, *args, **kwargs):
    """
    *args and **kwargs are passed to collections_to_dsk
    """
    key = delayed._key
    dsk = delayed.dask
    collections = dask_to_collections(dsk)
    collections = collections.values()
    persistent_dsk = persistent_collections_to_dsk(
        collections, key, serializers, cache, *args, **kwargs)
    return Delayed(key, persistent_dsk)


def dump_result(dump, func, key):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        dump(key, result)
        return result
    return wrapped_func


def delayed_load(load, key):
    @wraps(load)
    def wrapped_load():
        return load(key)
    return wrapped_load


def decorate_delayed(delayed_func, dump, decorate_mode=None):
    key = delayed_func._key
    dsk = dict(delayed_func.dask)
    if decorate_mode is None:
        # replace the function by decorated ones (with standard mechanism)
        task = list(dsk[key])
        task[0] = dump_result(dump,  task[0], key)
        dsk[key] = tuple(task)
        return Delayed(key, dsk)


class PersistentDAG(DAG):
    def __init__(self, dsk=None, cache=None, serializers=None):
        super(PersistentDAG, self).__init__(dsk)
        if cache is None:
            cache = dict()
        if serializers is None:
            serializers = dict()
        self.cache = cache
        self.serializers = serializers

    def get_persistent_dask(self, key=None, *args, **kwargs):
        collections = dask_to_collections(self._dask)
        collections = collections.values()
        return persistent_collections_to_dsk(
            collections, key, self.serializers, self.cache, *args, **kwargs)

    def add_task(self, func, *args, **kwargs):
        """
        Special keyword arguments are:
        - dask_key_name
        - dask_serializer
        """
        serializer = kwargs.pop('dask_serializer', None)
        delayed_func = super(PersistentDAG, self).add_task(
            func, *args, **kwargs)
        key = delayed_func._key
        if serializer is not None:
            delayed_dump = decorate_delayed(
                delayed_func, serializer.dump, decorate_mode=None)
            self._dask.update(delayed_dump.dask)
            self.serializers[key] = serializer
            return Delayed(key, self._dask)
        else:
            return delayed_func

    @property
    def persistent_dask(self):
        return self.get_persistent_dask()

    @property
    def dask(self):
        return self.get_persistent_dask()

    @dask.setter
    def dask(self, dsk):
        self._dask = dsk

    def is_computed(self):
        return {key: self.serializers[key].is_computed(key)
                for key in self._dask.keys() if key in self.serializers}

    def get(self, keys=None, **kwargs):
        """
        Wrapper around dask.get.
        Use cache or serialzed data if available.
        """
        if keys is None:
            keys = self._keys()
        dsk = self.get_persistent_dask(keys)
        # get result
        result = self._get(dsk, keys, **kwargs)
        # store in cache
        try:
            self.cache.update({keys: result})
        except TypeError:
            self.cache.update(dict(zip(keys, result)))
        return result

    def status(self):
        status = dict()
        for key in self._dask.keys():
            if key in self.serializers:
                if self.serializers[key].is_computed(key):
                    status[key] = 'computed'
                else:
                    status[key] = 'not_computed'
        return status

    def dot_status(self):
        status = self.status()
        dot_status = dict()
        for key, value in status.items():
            dot_status[key] = DOT_STATUS[value]
        return dot_status

    def visualize(self, raw_dask=True, *args, **kwargs):
        dot_status = self.dot_status()
        if raw_dask:
            dsk = self._dask
        else:
            dsk = self.dask
        return visualize(dsk,
                         data_attributes=dot_status,
                         # function_attributes=dot_status,
                         *args, **kwargs)
