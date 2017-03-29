# import pytest
from ..persist import PersistentDAG
from functools import wraps

# global variable to simulate the fact to have serialize data somewhere
IS_COMPUTED = dict()


def load_data():
    print 'load data ...'
    return 'data'


def clean_data(data):
    assert isinstance(data, str)
    print 'clean data ...'
    return 'cleaned_data'


def analyze_data(cleaned_data, option=1, **other_options):
    assert isinstance(cleaned_data, str)
    print 'analyze data ...'
    return 'analyzed_data'


class Serializer(object):

    def __init__(self):
        pass

    def load(self, key):
        print "load data for key {} ...".format(key)
        return IS_COMPUTED[key]

    def dump(self, key, value):
        print "save data with key {} ...".format(key)
        IS_COMPUTED[key] = value

    def is_computed(self, key):
        return IS_COMPUTED.get(key) is not None

    def delayed_load(self, key):
        def load():
            return self.load(key)
        return load

    def dump_result(self, func, key):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            self.dump(key, result)
            return result
        return wrapped_func


def setup_graph(**kwargs):
    g = PersistentDAG(**kwargs)
    serializer = Serializer()
    for pool in ['pool1', 'pool2']:
        g.add_task(('data', pool), serializer, load_data)
        g.add_task(('cleaned_data', pool), serializer,
                   clean_data, ('data', pool))
        g.add_task(('analyzed_data', pool), serializer,
                   analyze_data, ('cleaned_data', pool))
    return g


def test_delayed():
    from dask import delayed
    data = delayed(load_data)(dask_key_name=('data', 'pool1'))
    cleaned_data = delayed(clean_data)(
        dask_key_name=('cleaned_data', 'pool1'),
        data=data)
    assert cleaned_data.compute() == 'cleaned_data'


def test_get(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    data = g.get(('data', 'pool1'))
    assert data == 'data'
    data = g.get(('cleaned_data', 'pool1'))
    assert data == 'cleaned_data'
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_data'


def test_get_multiple_times(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_data'

    # Checking that it is cached
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_data'

    data = g.get(('cleaned_data', 'pool1'))
    assert data == 'cleaned_data'

    data = g.get(('cleaned_data', 'pool2'))
    assert data == 'cleaned_data'

    data = g.get(('analyzed_data', 'pool2'))
    assert data == 'analyzed_data'

    # get multiple results
    data = g.get([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
    assert isinstance(data, list)

    # check printed messages
    out, err = capsys.readouterr()
    assert out == """load data ...
save data with key ('data', 'pool1') ...
clean data ...
save data with key ('cleaned_data', 'pool1') ...
analyze data ...
save data with key ('analyzed_data', 'pool1') ...
load data for key ('cleaned_data', 'pool1') ...
load data ...
save data with key ('data', 'pool2') ...
clean data ...
save data with key ('cleaned_data', 'pool2') ...
analyze data ...
save data with key ('analyzed_data', 'pool2') ...
"""
    assert not err


def test_run(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    # run the graph
    data = g.run(key=('cleaned_data', 'pool2'))
    assert data == 'cleaned_data'

    data = g.run([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
    assert isinstance(data, dict)
    assert data == {('analyzed_data', 'pool2'): 'analyzed_data',
                    ('analyzed_data', 'pool1'): 'analyzed_data'}


def test_persistent_dsk(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    # the first time the grap is created it has functions
    assert not all(g.is_computed().values())
    assert g.persistent_dsk == g.dsk

    # run the graph
    g.run()
    assert all(g.is_computed().values())
    assert g.persistent_dsk != g.dsk
    # then the graph is replaced by cached data
    assert g.persistent_dsk.values() == \
        ['cleaned_data', 'analyzed_data', 'data', 'cleaned_data',
         'data', 'analyzed_data']

    # We recreate a new graph => the cache is deleted
    g = setup_graph()
    assert all(g.is_computed().values())
    # the graph contains the load methods
    assert g.persistent_dsk != g.dsk
    assert all(map(lambda f: f[0].func_name ==
                   'load', g.persistent_dsk.values()))

    # get multiple results
    data = g.get([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
    assert isinstance(data, list)
    assert data == ['analyzed_data', 'analyzed_data']

    out, err = capsys.readouterr()
    assert out == """load data ...
save data with key ('data', 'pool1') ...
clean data ...
save data with key ('cleaned_data', 'pool1') ...
analyze data ...
save data with key ('analyzed_data', 'pool1') ...
load data ...
save data with key ('data', 'pool2') ...
clean data ...
save data with key ('cleaned_data', 'pool2') ...
analyze data ...
save data with key ('analyzed_data', 'pool2') ...
load data for key ('analyzed_data', 'pool1') ...
load data for key ('analyzed_data', 'pool2') ...
"""
    assert not err


def test_cluster(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph(use_cluster=True)
    data = g.get([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
    assert isinstance(data, list)
    assert data == ['analyzed_data', 'analyzed_data']
    out, err = capsys.readouterr()
    assert not out
    assert not err


def test_async_run(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph(use_cluster=True)
    # persist assert en error because the given collection is not of type
    # dask.base.Base
    data = g.async_run(key=('cleaned_data', 'pool1'))
    assert data.compute() == 'cleaned_data'


def test_async_run_all(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph(use_cluster=True)
    # persist assert en error because the given collection is not of type
    # dask.base.Base
    futures = g.async_run()
    data = map(lambda x: x.compute(), futures)

    assert sorted(data) == ['analyzed_data', 'analyzed_data',
                            'cleaned_data', 'cleaned_data', 'data', 'data']

    data = g.client.gather(futures)
    # here I do not know why gather still return delayed objects...
    # assert isinstance(data[0], str)
