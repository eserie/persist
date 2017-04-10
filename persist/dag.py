from functools import wraps
from dask import threaded
from dask.base import Base
from dask.delayed import Delayed
# from dask.optimize import cull
from toolz import first
from dask.base import collections_to_dsk
from dask import delayed

__all__ = ['DAG']


def in_dict(arg, collections):
    try:
        return arg in collections
    except TypeError:
        return False


def dask_to_collections(dask):
    funcs = dict()
    for key in dask.keys():
        # dsk, _ = cull(dask, key)
        # funcs[key] = Delayed(key, dsk)
        funcs[key] = Delayed(key, dask)
    return funcs


def dask_to_digraph(dsk):
    from networkx import DiGraph
    from dask.core import get_dependencies
    g = DiGraph()
    for key, value in dsk.items():
        g.add_node(key, dict(func=value))
    for key, value in dsk.items():
        g.add_node(key, dict(func=value))
        for dep in get_dependencies(dsk, key):
            g.add_edge(dep, key)
    return g


def digraph_to_dask(graph):
    dsk = dict()
    for v in graph.nodes():
        if 'func' in graph.node[v]:
            dsk[v] = graph.node[v]['func']
    return dsk


class DAG(Base):
    __slots__ = ('_dask', '_keys')
    _default_get = staticmethod(threaded.get)
    _optimize = staticmethod(lambda d, k, **kwds: d)

    def __init__(self, dsk=None):
        if dsk is None:
            dsk = dict()
        self._dask = dsk

    @property
    def dask(self):
        return self._dask

    @dask.setter
    def dask(self, dsk):
        self._dask = dsk

    @classmethod
    def from_digraph(cls, graph):
        dsk = digraph_to_dask(graph)
        return cls(dsk)

    @property
    def terminal_nodes(self):
        graph = self.to_digraph()
        terminal_nodes = [k for k, v in graph.succ.items() if not v]
        return terminal_nodes

    def _finalize(self, args):
        if len(self._keys()) > 1:
            return args
        else:
            return first(args)

    def _keys(self):
        return self.terminal_nodes
        # return self.dask.keys()

    def add_task(self, func, *args, **kwargs):
        """
        Special keyword arguments are:
        - dask_key_name
        """
        key = kwargs.get('dask_key_name')
        if key:
            assert key not in self._dask, "specified key is already used"

        delayed_func = delayed(func, pure=True)
        collections = dask_to_collections(self._dask)
        # normalize args and kwargs replacing values that are in the graph by
        # Delayed objects

        args = [collections[arg] if in_dict(
            arg, collections) else arg for arg in args]
        kwargs.update({k: v for k, v in collections.items() if k in kwargs})

        delayed_func = delayed_func(*args, **kwargs)

        if key is None:
            key = delayed_func._key
        else:
            # coherence check. TODO: remove
            assert key == delayed_func._key

        # update state
        collections[key] = delayed_func
        self.dask = collections_to_dsk(collections.values())

        return delayed_func

    def get(self, key, **get_kwargs):
        result = self._get(self.dask, key, **get_kwargs)
        if isinstance(key, list):
            # TODO: should we convert to the same type than key? This should be
            # done by dask?
            result = type(key)(result)
        return result

    def run(self, key=None):
        collections = dask_to_collections(self.dask)
        try:
            collections = {key: collections[key]}
        except (TypeError, KeyError):
            if key is not None:
                collections = {k: v for k,
                               v in collections.items() if k in key}
        futures = dict()
        for key, func in collections.items():
            futures[key] = func.persist()
        return futures

    @staticmethod
    def results(futures):
        return {key: fut.compute() for key, fut in futures.items()}

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

    def to_digraph(self):
        return dask_to_digraph(self.dask)
