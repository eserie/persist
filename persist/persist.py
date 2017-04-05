from dask import get
from dask.optimize import cull
from dask.base import collections_to_dsk

from .dag import DAG

__all__ = ['PersistentDAG']


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
    def __init__(self, use_cluster=False):
        super(PersistentDAG, self).__init__(self)
        self.cache = dict()
        self.serializer = dict()
        if use_cluster:
            from distributed import LocalCluster, Client
            self.cluster = LocalCluster()
            self.client = Client(self.cluster)
        else:
            self.cluser = None
            self.client = None

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
