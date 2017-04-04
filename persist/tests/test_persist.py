import pytest
from functools import wraps
# from time import sleep
from ..persist import PersistentDAG
# global variable to simulate the fact to have serialize data somewhere
IS_COMPUTED = dict()


def load_data(*args, **kwargs):
    # sleep(2)
    print 'load data ...'
    if args:
        print args
        return 'data_{}'.format(args)
    if kwargs:
        print kwargs
        return 'data_{}'.format(kwargs)
    return 'data'


def clean_data(data, *args, **kwargs):
    assert isinstance(data, str)
    print 'clean data ...'
    if args:
        print args
        data = data + '_' + '_'.join(map(lambda x: '{}'.format(x), args))
    if kwargs:
        print kwargs
        data = data + '_' + \
            '_'.join(map(lambda kv: '{}_{}'.format(
                kv[0], kv[1]), kwargs.items()))
    return 'cleaned_{}'.format(data)


def analyze_data(cleaned_data, option=1, **other_options):
    assert isinstance(cleaned_data, str)
    print 'analyze data ...'
    return 'analyzed_{}'.format(cleaned_data)


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


def test_submit_api():
    g = PersistentDAG()
    serializer = Serializer()
    for pool in ['pool1', 'pool2']:
        g.submit(load_data,
                 dask_key_name=('data', pool),
                 dask_serializer=serializer)
        g.submit(clean_data, ('data', pool),
                 dask_key_name=('cleaned_data', pool),
                 dask_serializer=serializer)
        g.submit(analyze_data, ('cleaned_data', pool),
                 dask_key_name=('analyzed_data', pool),
                 dask_serializer=serializer)
    data = g.run()
    assert data == {('analyzed_data', 'pool1'): 'analyzed_cleaned_data',
                    ('analyzed_data', 'pool2'): 'analyzed_cleaned_data',
                    ('cleaned_data', 'pool1'): 'cleaned_data',
                    ('cleaned_data', 'pool2'): 'cleaned_data',
                    ('data', 'pool1'): 'data',
                    ('data', 'pool2'): 'data'}


def test_delayed_api():
    g = PersistentDAG()
    serializer = Serializer()
    for pool in ['pool1', 'pool2']:
        g.delayed(load_data, ('data', pool), serializer)()
        g.delayed(clean_data, ('cleaned_data', pool),
                  serializer)(('data', pool))
        g.delayed(analyze_data, ('analyzed_data', pool),
                  serializer)(('cleaned_data', pool))
    data = g.run()
    assert data == {('analyzed_data', 'pool1'): 'analyzed_cleaned_data',
                    ('analyzed_data', 'pool2'): 'analyzed_cleaned_data',
                    ('cleaned_data', 'pool1'): 'cleaned_data',
                    ('cleaned_data', 'pool2'): 'cleaned_data',
                    ('data', 'pool1'): 'data',
                    ('data', 'pool2'): 'data'}


def test_key_none():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task(None, serializer, load_data, option=10)
    data = g.run()
    assert data.values() == ["data_{'option': 10}"]
    assert data.keys()[0].startswith('load_data-')
    keys = g.funcs.keys()
    assert len(keys) == 1
    assert keys[0] is not None
    assert keys[0].startswith('load_data-')


def test_key_none_serializer_none():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    g.add_task(None, None, load_data, option=10)
    data = g.run()
    assert data.values() == ["data_{'option': 10}"]
    assert data.keys()[0].startswith('load_data-')


def test_kwargs():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task('data', serializer, load_data, option=10)
    data = g.run()
    assert data == {'data': "data_{'option': 10}"}


def test_varargs():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    varargs = (10,)
    g.add_task('data', serializer, load_data, *varargs)
    data = g.run()
    assert data == {'data': "data_(10,)"}


def test_use_already_used_key():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task('key_data1', serializer, load_data, option=10)
    with pytest.raises(AssertionError) as err:
        g.add_task('key_data1', serializer, load_data, option=20)
    err = str(err)
    assert err.endswith("key is already used")


def test_varargs_deps():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task('key_data1', serializer, load_data, option=10)
    g.add_task('key_data2', serializer, load_data, option=20)
    varargs = ('key_data1', 'key_data2',)
    g.add_task('cleaned_data', serializer, clean_data, *varargs)
    data = g.run()
    assert data == {'key_data1': "data_{'option': 10}",
                    'key_data2': "data_{'option': 20}",
                    'cleaned_data': "cleaned_data_{'option': 10}_data_{'option': 20}",
                    }


def test_kwargs_deps():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task('key_data1', serializer, load_data, option=10)
    g.add_task('key_data2', serializer, load_data, option=20)
    kwargs = dict(data='key_data1', other='key_data2')
    g.add_task('cleaned_data', serializer, clean_data, **kwargs)
    data = g.run()
    assert data == {'key_data1': "data_{'option': 10}",
                    'key_data2': "data_{'option': 20}",
                    'cleaned_data': "cleaned_data_{'option': 10}_other_data_{'option': 20}",
                    }


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
    assert data == 'analyzed_cleaned_data'


def test_get_multiple_times(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_cleaned_data'

    # Checking that it is cached
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_cleaned_data'

    data = g.get(('cleaned_data', 'pool1'))
    assert data == 'cleaned_data'

    data = g.get(('cleaned_data', 'pool2'))
    assert data == 'cleaned_data'

    data = g.get(('analyzed_data', 'pool2'))
    assert data == 'analyzed_cleaned_data'

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
    assert data == {('analyzed_data', 'pool2'): 'analyzed_cleaned_data',
                    ('analyzed_data', 'pool1'): 'analyzed_cleaned_data'}


def test_persistent_dask(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    # the first time the grap is created it has functions
    assert not all(g.is_computed().values())
    assert g.persistent_dask == g.dask

    # run the graph
    g.run()
    assert all(g.is_computed().values())
    assert g.persistent_dask != g.dask
    # then the graph is replaced by cached data
    values = dict(g.persistent_dask).values()

    assert values == ['cleaned_data', 'analyzed_cleaned_data',
                      'data', 'cleaned_data', 'data', 'analyzed_cleaned_data']

    # We recreate a new graph => the cache is deleted
    g = setup_graph()
    assert all(g.is_computed().values())
    # the graph contains the load methods
    assert g.persistent_dask != g.dask
    assert all(map(lambda f: f[0].func_name ==
                   'load', g.persistent_dask.values()))

    # get multiple results
    data = g.get([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
    assert isinstance(data, list)
    assert data == ['analyzed_cleaned_data', 'analyzed_cleaned_data']

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
    assert data == ['analyzed_cleaned_data', 'analyzed_cleaned_data']
    out, err = capsys.readouterr()
    assert not out
    # assert not err


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

    assert sorted(data) == ['analyzed_cleaned_data', 'analyzed_cleaned_data',
                            'cleaned_data', 'cleaned_data', 'data', 'data']

    data = g.client.gather(futures)
    # here I do not know why gather still return delayed objects...
    # assert isinstance(data[0], str)


def test_visualize():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph(use_cluster=True)
    g.visualize(format='svg')
