from functools import partial
from dask import get
from dask.optimize import cull
import inspect


class Persist(object):
    def __init__(self, use_cluster=False):
        self.dsk = dict()
        self.cache = dict()
        self.serializer = dict()
        if use_cluster:
            from distributed import LocalCluster, Client
            self.cluster = LocalCluster()
            self.client = Client(self.cluster)
        else:
            self.cluser = None
            self.client = None

    def add_task(self, key, serializer, func, *args, **kwargs):
        self.serializer[key] = serializer
        # prepare arguments for the dask graph specification
        args_dict = inspect.getcallargs(func, *args, **kwargs)
        args_spec = inspect.getargspec(func)
        args_list = [args_dict[argname] for argname in args_spec.args]
        # dump data as side effect
        func = serializer.dump_result(func, key)
        # propagate keyword arguments
        if args_spec.keywords:
            func = partial(func, **args_dict[args_spec.keywords])
        # add task to dask graph
        self.dsk[key] = (func,) + tuple(args_list)
        return key

    @property
    def persistent_dsk(self):
        return self.get_persistent_dsk()

    def is_computed(self):
        return {key: self.serializer[key].is_computed(key)
                for key in self.dsk.keys() if key in self.serializer}

    def get_persistent_dsk(self, key=None):
        dsk = self.dsk.copy()
        if key is not None:
            dsk, _ = cull(dsk, key)
        # load instead of compute
        # the fact to call the method "is_computed" may slow down the code.
        dsk.update({key: (self.serializer[key].delayed_load(key),)
                    for key in dsk.keys()
                    if key in self.serializer
                    and self.serializer[key].is_computed(key)})
        # use cache instead of loadind
        dsk.update({key: self.cache[key] for key in dsk.keys() if key in self.cache})
        # filter again task after function have been replaced by load or values
        if key is not None:
            dsk, _ = cull(dsk, key)
        return dsk

    def get(self, key):
        """
        Wrapper around dask.get.
        Use cache or serialzed data if available.
        """
        dsk = self.get_persistent_dsk(key)
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
            key = self.dsk.keys()
        result = self.get(key)
        if isinstance(key, list):
            return dict(zip(key, result))
        else:
            return result

    def async_run(self, key=None):
        assert self.client is not None
        self.client.persist(self.dsk)
        # return dsk
