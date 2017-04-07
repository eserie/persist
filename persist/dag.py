
from functools import wraps
import inspect
from dask import threaded
from dask.base import Base
from dask.delayed import Delayed
# from dask.optimize import cull
from dask.base import collections_to_dsk
from dask import delayed

__all__ = ['DAG']


def dask_to_collections(dask):
    funcs = dict()
    for key in dask.keys():
        #dsk, _ = cull(dask, key)
        #funcs[key] = Delayed(key, dsk)
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
    __slots__ = ('dask', '_keys')
    _finalize = staticmethod(list)
    _default_get = staticmethod(threaded.get)
    _optimize = staticmethod(lambda d, k, **kwds: d)

    def __init__(self, dsk=None):
        if dsk is None:
            dsk = dict()
        self.dask = dsk

    @property
    def collections(self):
        return dask_to_collections(self.dask)

    @classmethod
    def from_digraph(cls, graph):
        dsk = digraph_to_dask(graph)
        return cls(dsk)

    @property
    def terminal_nodes(self):
        graph = self.to_digraph()
        terminal_nodes = [k for k, v in graph.succ.items() if not v]
        return terminal_nodes

    def _keys(self):
        return self.terminal_nodes
        # return self.dask.keys()

    def add_task(self, func, *args, **kwargs):
        """
        Special keyword arguments are:
        - dask_key_name
        """
        if kwargs.get('dask_key_name'):
                assert kwargs.get('dask_key_name') not in self.dask, "specified key is already used"
        delayed_func = delayed(func, pure=True)

        # normalize args and kwargs replacing values that are in the graph by Delayed objects
        collections = dask_to_collections(self.dask)
        args = [collections[arg] if arg in collections else arg for arg in args]
        kwargs.update({k:v for k, v in collections.items() if k in kwargs})
        delayed_func = delayed_func(*args, **kwargs)
        key = delayed_func._key
        # update state
        collections[key] = delayed_func
        self.dask = collections_to_dsk(collections.values())
        return delayed_func

    def get(self, key, **kwargs):
        result = self._get(self.dask, key, **kwargs)
        if isinstance(key, list):
            # TODO: should we convert to the same type than key? This should be
            # done by dask?
            result = list(result)
        return result

    def run(self, key=None):
        if key is None:
            # key = self._keys()#dask.keys()
            key = self.dask.keys()
        result = self.get(key)
        if isinstance(key, list):
            return dict(zip(key, result))
        else:
            return result

    def async_run(self, key=None):
        dsk = self.dask
        if key is None:
            # key = self._keys()#dsk.keys()
            key = dsk.keys()
        from dask.base import persist
        if isinstance(key, list):
            funcs = self.collections.values()
        else:
            funcs = self.collections[key]
        (futures, ) = persist(funcs)
        return futures

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
