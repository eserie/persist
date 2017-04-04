from functools import wraps
from dask import get
from dask.optimize import cull
import inspect


class PersistentDAG(object):
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
        # prepare arguments for the dask graph specification
        args_dict = inspect.getcallargs(func, *args, **kwargs)
        args_spec = inspect.getargspec(func)
        if args_spec.keywords:
            kwds = args_dict.pop(args_spec.keywords)
            args_dict.update(kwds)
        args_dict.update({arg_name: self.funcs[arg_value]
                          for arg_name, arg_value in args_dict.iteritems()
                          if arg_value in self.funcs.keys()})
        # set list of arguments
        args_tuple = tuple([args_dict.pop(argname)
                            for argname in args_spec.args])
        if args_spec.varargs:
            args_tuple += args_dict.pop(args_spec.varargs)
        # wrap func in order that it dump data as a side-effect
        if serializer is not None:
            func = serializer.dump_result(func, key)
        # use dask delayed collection to wrap functions
        from dask import delayed
        delayed_func = delayed(func)(
            dask_key_name=key, *args_tuple, **args_dict)
        # stotre delayed funcs
        if key is None:
            keys = delayed_func.dask.keys()
            assert len(keys) == 1
            key = keys[0]
        assert key not in self.dsk, "key is already used"            
        self.funcs[key] = delayed_func
        if serializer is not None:
            self.serializer[key] = serializer
        return key

    @property
    def dsk(self):
        from dask.base import collections_to_dsk
        dask = collections_to_dsk(self.funcs.values())
        return dask

    # @property
    # def old_dsk(self):
    #     from dask.delayed import to_task_dask
    #     task, dask = to_task_dask(self.funcs)
    #     dask = dict(dask)
    #     return dask

    @property
    def persistent_dsk(self):
        return self.get_persistent_dsk()

    def is_computed(self):
        return {key: self.serializer[key].is_computed(key)
                for key in self.dsk.keys() if key in self.serializer}

    def get_persistent_dsk(self, key=None):
        dsk = self.dsk
        if key is not None:
            dsk, _ = cull(dsk, key)
        # load instead of compute
        # the fact to call the method "is_computed" may slow down the code.
        dsk_serialized = {key: (self.serializer[key].delayed_load(key),)
                          for key in dsk.keys()
                          if key in self.serializer
                          and self.serializer[key].is_computed(key)}
        dsk.update(dsk_serialized)
        # use cache instead of loadind
        dsk_cached = {key: self.cache[key]
                      for key in dsk.keys() if key in self.cache}
        dsk.update(dsk_cached)
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
        if key is None:
            key = self.dsk.keys()
        if not isinstance(key, list):
            return self.funcs[key].persist()
        else:
            futures = [self.funcs[k].persist() for k in key]
            return futures
