# from functools import partial
from functools import wraps
from pprint import pprint
from dask.optimize import cull
from dask.base import collections_to_dsk
from dask.base import visualize
from dask.delayed import delayed
from dask.delayed import Delayed
from toolz import curry
from toolz import first
from .dag import DAG
from .dag import dask_to_collections
from .dag import in_dict

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

    """

    dsk = collections_to_dsk(collections, *args, **kwargs)

    if key is not None:
        dsk, _ = cull(dsk, key)

    if serializers is not None:
        # load instead of compute
        dsk_serialized = get_relevant_keys_from_on_disk_cache(dsk, serializers)
        dsk.update(dsk_serialized)
        # for k in dsk_serialized.keys():
        #     dump_key = ('serialize', k)
        #     if dump_key in dsk:
        #         del dsk[dump_key]
        #     compute_key = ('compute', k)
        #     if compute_key in dsk:
        #         # it may depends of the mode used to dump data.
        #         del dsk[compute_key]

    if cache is not None:
        # use cache instead of loadind
        dsk_cached = get_relevant_keys_from_memory_cache(dsk, cache)
        dsk.update(dsk_cached)

    # filter again task after function have been replaced by load or values
    if key is not None:
        dsk, _ = cull(dsk, key)

    return dsk


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
    # elif decorate_mode == 'add_dump_tasks':
    #     dump_key = ('serialize', key)
    #     # we must use partial in order to avoid confusion with the key of the graph
    #     delayed_dump = delayed(partial(dump, key=key), pure=True)
    #     # simply add tasks which dump data.
    #     # this way to do is problematic because it becomes complicated to retriew tasks that generate data in the graph
    #     delayed_dump = delayed_dump(value=delayed_func, dask_key_name=dump_key)
    #     return delayed_dump
    # elif decorate_mode == 'dask_decorate_with_dump':
    #     # decorate the original function via explicit dask tasks
    #     dump_key = ('serialize', key)
    #     compute_key = ('compute', key)
    #     # we must use partial in order to avoid confusion with the key of the graph
    #     delayed_dump = delayed(partial(dump, key=key), pure=True)
    #     dsk[compute_key] = dsk.pop(key)
    #     delayed_compute = Delayed(compute_key, dsk)
    #     delayed_dump = delayed_dump(value=delayed_compute, dask_key_name=dump_key)
    #     delayed_decorated = delayed(first, pure=True)([delayed_compute, delayed_dump], dask_key_name=key)
    #     return delayed_decorated


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
        delayed_func = super(PersistentDAG, self).add_task(
            func, *args, **kwargs)
        key = delayed_func._key
        if serializer is not None:
            delayed_dump = decorate_delayed(
                delayed_func, serializer.dump, decorate_mode=None)
            self._dask.update(delayed_dump.dask)
            self.serializer[key] = serializer
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

    # @staticmethod
    # def results(futures):
    #     results = {key: fut.compute() for key, fut in futures.items()}
    #     results = {k:v for k, v in results.items() if not (isinstance(k, tuple) and k[0] in ['serialize', 'compute'])}
    #     return results
