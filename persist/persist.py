from dask.optimize import cull
from dask.base import collections_to_dsk
from dask.base import visualize
from dask.delayed import delayed

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


class PersistentDAG(DAG):
    def __init__(self, dsk=None, cache=None, serializer=None):
        super(PersistentDAG, self).__init__(dsk)
        if cache is None:
            cache = dict()
        if serializer is None:
            serializer = dict()
        self.cache = cache
        self.serializer = serializer

    def get_persistent_dask(self, key=None, *args, **kwargs):
        collections = dask_to_collections(self._dask)
        collections = collections.values()
        return persistent_collections_to_dsk(
            collections, key, self.serializer, self.cache, *args, **kwargs)

    def add_task(self, func, *args, **kwargs):
        """
        Special keyword arguments are:
        - dask_key_name
        - dask_serializer
        """
        serializer = kwargs.pop('dask_serializer', None)
        key = kwargs.get('dask_key_name')
        if key:
            assert key not in self._dask, "specified key is already used"

        # get the key before decorated with the serializer
        tmp_delayed_func = delayed(func, pure=True)(*args, **kwargs)
        key = tmp_delayed_func._key
        # wrap func in order that it dump data as a side-effect
        if serializer is not None:
            func = serializer.dump_result(func, key)
            self.serializer[key] = serializer

        delayed_func = delayed(func, pure=True)
        collections = dask_to_collections(self._dask)
        # normalize args and kwargs replacing values that are in the graph by
        # Delayed objects
        args = [collections[arg] if arg in collections else arg for arg in args]
        kwargs.update({k: v for k, v in collections.items() if k in kwargs})

        if 'dask_key_name' not in kwargs:
            # set dask_key_name in order to avoid that a new tokenize
            kwargs['dask_key_name'] = key
        else:
            # coherence check. TODO: remove
            assert kwargs['dask_key_name'] == key

        delayed_func = delayed_func(*args, **kwargs)
        assert key == delayed_func._key
        # update state
        collections[key] = delayed_func
        self.dask = collections_to_dsk(collections.values())

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
        return {key: self.serializer[key].is_computed(key)
                for key in self._dask.keys() if key in self.serializer}

    def get(self, key, **kwargs):
        """
        Wrapper around dask.get.
        Use cache or serialzed data if available.
        """
        dsk = self.get_persistent_dask(key)
        # get result
        result = self._get(dsk, key, **kwargs)
        # store in cache
        try:
            self.cache.update({key: result})
        except TypeError:
            self.cache.update(dict(zip(key, result)))
            result = list(result)
        return result

    def status(self):
        status = dict()
        for key in self._dask.keys():
            if key in self.serializer:
                if self.serializer[key].is_computed(key):
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
