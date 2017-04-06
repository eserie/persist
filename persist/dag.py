
from functools import wraps
import inspect
from dask import threaded
from dask.base import Base
from dask.delayed import Delayed
from dask.optimize import cull
from dask.base import collections_to_dsk
from dask import delayed

__all__ = ['DAG']


def dask_to_collections(dask):
    funcs = dict()
    for key in dask.keys():
        dsk, _ = cull(dask, key)
        funcs[key] = Delayed(key, dsk)
    return funcs


def delayed_from_dask_func(dask, key, func, *args, **kwargs):
    """
    Special keyword arguments are:
    - dask_key_name
    - dask_serializer
    """
    collections = dask_to_collections(dask)
    # Prepare arguments
    args_tuple, args_dict = prepare_args(func, args, kwargs, collections)
    # use dask delayed collection to wrap functions
    delayed_func = delayed(func, pure=True)(
        dask_key_name=key, *args_tuple, **args_dict)
    return delayed_func


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


def is_key_in_collections(arg_value, collections):
    return not isinstance(arg_value, Base) and arg_value in collections.keys()


def prepare_args(func, args, kwargs, collections):
    """
    prepare arguments of the given func.
    If some arguments correspond to keys in the collections (dict of delayed funcs),
    they are replaced by it.
    """
    args_dict = inspect.getcallargs(func, *args, **kwargs)
    args_spec = inspect.getargspec(func)
    if args_spec.keywords:
        kwds = args_dict.pop(args_spec.keywords)
        args_dict.update(kwds)
    args_collections = {arg_name: collections[arg_value]
                        for arg_name, arg_value in args_dict.iteritems()
                        if is_key_in_collections(arg_value, collections)}
    args_dict.update(args_collections)
    # set list of arguments
    args_tuple = tuple([args_dict.pop(argname) for argname in args_spec.args])
    if args_spec.varargs:
        varargs = args_dict.pop(args_spec.varargs)
        # TODO: replace values of varargs by collections if a key is mapped.
        new_varargs = []
        for arg_value in varargs:
            if is_key_in_collections(arg_value, collections):
                new_varargs.append(collections[arg_value])
            else:
                new_varargs.append(arg_value)
        args_tuple += tuple(new_varargs)
    return args_tuple, args_dict


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
        key = kwargs.pop('dask_key_name', None)
        delayed_func = delayed_from_dask_func(
            self.dask, key, func, *args, **kwargs)
        # set key
        if key is None:
            # use tokenize key named setted by delayed
            # in this case dask manage correctly unicity of keys
            key = delayed_func._key
        else:
            assert key not in self.dask, "key is already used"
        # update state
        collections = self.collections
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
