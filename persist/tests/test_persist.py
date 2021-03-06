import os
import pytest
import dask
from distributed import Client
from ..persist import PersistentDAG

from .helpers import load_data, clean_data, analyze_data

dask.set_options(get=dask.async.get_sync)
# global variable to simulate the fact to have serialize data somewhere
IS_COMPUTED = dict()


class Serializer(object):

    def __init__(self):
        pass

    def load(self, key):
        print "serialzer load data for key {} ...".format(key)
        return IS_COMPUTED[key]

    def dump(self, key, value):
        print "serialzer dump data for key {} ...".format(key)
        IS_COMPUTED[key] = value

    def is_computed(self, key):
        return IS_COMPUTED.get(key) is not None


def setup_graph(**kwargs):
    g = PersistentDAG(**kwargs)
    serializer = Serializer()
    for pool in ['pool1', 'pool2']:
        g.add_task(load_data,
                   dask_key_name=('data', pool),
                   dask_serializer=serializer)
        g.add_task(clean_data, ('data', pool),
                   dask_key_name=('cleaned_data', pool),
                   dask_serializer=serializer)
        g.add_task(analyze_data, ('cleaned_data', pool),
                   dask_key_name=('analyzed_data', pool),
                   dask_serializer=serializer)
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
    futures = g.run()
    data = g.results(futures)
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
        g.delayed(load_data)(dask_key_name=(
            'data', pool), dask_serializer=serializer)
        g.delayed(clean_data)(('data', pool),
                              dask_key_name=('cleaned_data', pool),
                              dask_serializer=serializer)
        g.delayed(analyze_data)(('cleaned_data', pool),
                                dask_key_name=('analyzed_data', pool),
                                dask_serializer=serializer)
    futures = g.run()
    data = g.results(futures)
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
    g.add_task(dask_serializer=serializer, func=load_data, option=10)
    futures = g.run()
    data = g.results(futures)
    assert data.values() == ["data_{'option': 10}"]
    assert data.keys()[0].startswith('load_data-')
    keys = g.dask.keys()
    assert len(keys) == 1
    assert keys[0].startswith('load_data-')


def test_key_none_serializer_none():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    g.add_task(load_data, option=10)
    futures = g.run()
    data = g.results(futures)
    assert data.values() == ["data_{'option': 10}"]
    assert data.keys()[0].startswith('load_data-')


def test_kwargs():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task(dask_key_name='data', dask_serializer=serializer,
               func=load_data, option=10)
    futures = g.run()
    data = g.results(futures)
    assert data == {'data': "data_{'option': 10}"}


def test_varargs():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    varargs = (10,)
    g.add_task(load_data, *varargs, dask_key_name='data',
               dask_serializer=serializer)
    futures = g.run()
    data = g.results(futures)
    assert data == {'data': "data_(10,)"}


def test_use_already_used_key():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task(dask_key_name='key_data1',
               dask_serializer=serializer, func=load_data, option=10)
    with pytest.raises(AssertionError) as err:
        g.add_task(dask_key_name='key_data1',
                   dask_serializer=serializer, func=load_data, option=20)
    err = str(err)
    assert err.endswith("key is already used")


def test_varargs_deps():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task(func=load_data, option=10,
               dask_key_name='key_data1',
               dask_serializer=serializer)
    g.add_task(func=load_data, option=20,
               dask_key_name='key_data2',
               dask_serializer=serializer)
    varargs = ('key_data1', 'key_data2',)
    g.add_task(clean_data, *varargs,
               dask_key_name='cleaned_data',
               dask_serializer=serializer)
    futures = g.run()
    data = g.results(futures)
    assert data == {'key_data1': "data_{'option': 10}",
                    'key_data2': "data_{'option': 20}",
                    'cleaned_data': "cleaned_data_{'option': 10}_data_{'option': 20}",
                    }


def test_kwargs_deps():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    g.add_task(dask_key_name='key_data1',
               dask_serializer=serializer, func=load_data, option=10)
    g.add_task(dask_key_name='key_data2',
               dask_serializer=serializer, func=load_data, option=20)
    kwargs = dict(data='key_data1', other='key_data2')
    g.add_task(dask_key_name='cleaned_data',
               dask_serializer=serializer, func=clean_data, **kwargs)
    futures = g.run()
    data = g.results(futures)
    assert data == {'key_data1': "data_{'option': 10}",
                    'key_data2': "data_{'option': 20}",
                    'cleaned_data': "cleaned_data_{'option': 10}_other_data_{'option': 20}",
                    }

    # finally add one task without key_name and without serializer
    func = g.add_task(analyze_data, 'cleaned_data')
    futures = g.run()
    data = g.results(futures)
    ref_data = "analyzed_cleaned_data_{'option': 10}_other_data_{'option': 20}"
    assert data[func._key] == ref_data

    data = g.submit(analyze_data, 'cleaned_data')
    assert data.compute() == ref_data


def test_dask_delayed():
    from dask import delayed
    data = delayed(load_data)(dask_key_name=('data', 'pool1'))
    assert data.compute() == 'data'
    cleaned_data = delayed(clean_data)(
        dask_key_name=('cleaned_data', 'pool1'),
        data=data)
    assert cleaned_data.compute() == 'cleaned_data'


def test_persistent_delayed():
    """
    Check with have the same behaviour than with dask delayed.
    """
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    data = g.delayed(load_data)(dask_key_name=('data', 'pool1'))
    assert data.compute() == 'data'
    cleaned_data = g.delayed(clean_data)(
        dask_key_name=('cleaned_data', 'pool1'),
        data=data)
    assert cleaned_data.compute() == 'cleaned_data'


def test_delayed_using_cache():
    """
    Check with have the same behaviour than with dask delayed.
    """
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = PersistentDAG()
    serializer = Serializer()
    data = g.delayed(load_data)(dask_key_name=('data', 'pool1'))
    cleaned_data = g.delayed(clean_data)(
        dask_key_name=('cleaned_data', 'pool1'),
        data=data,
        dask_serializer=serializer)
    assert len(cleaned_data.dask) == 2
    assert cleaned_data.compute() == 'cleaned_data'
    from ..persist import delayed_using_cache
    obj = delayed_using_cache(cleaned_data, g.serializers, g.cache)
    assert len(obj.dask) == 1
    assert obj.compute() == 'cleaned_data'


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
    with dask.set_options(get=dask.async.get_sync):
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
        assert isinstance(data, tuple)

    # check printed messages
    out, err = capsys.readouterr()
    assert out == """load data ...
serialzer dump data for key ('data', 'pool1') ...
clean data ...
serialzer dump data for key ('cleaned_data', 'pool1') ...
analyze data ...
serialzer dump data for key ('analyzed_data', 'pool1') ...
serialzer load data for key ('cleaned_data', 'pool1') ...
load data ...
serialzer dump data for key ('data', 'pool2') ...
clean data ...
serialzer dump data for key ('cleaned_data', 'pool2') ...
analyze data ...
serialzer dump data for key ('analyzed_data', 'pool2') ...
"""
    assert not err


def test_run(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    # run the graph
    futures = g.run(keys=('cleaned_data', 'pool2'))
    data = g.results(futures).values()[0]
    assert data == 'cleaned_data'

    futures = g.run([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
    data = g.results(futures)
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
    with dask.set_options(get=dask.async.get_sync):
        futures = g.run()
        assert IS_COMPUTED
        assert all(g.is_computed().values())
        assert g.persistent_dask != g.dask
        # then the graph is replaced by cached data
        values = g.results(futures).values()

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
    assert isinstance(data, tuple)
    assert data == ('analyzed_cleaned_data', 'analyzed_cleaned_data')

    out, err = capsys.readouterr()
    assert sorted(out.split('\n')) == \
        ['',
         'analyze data ...',
         'analyze data ...',
         'clean data ...',
         'clean data ...',
         'clean data ...',
         'clean data ...',
         'load data ...',
         'load data ...',
         'load data ...',
         'load data ...',
         'load data ...',
         'load data ...',
         "serialzer dump data for key ('analyzed_data', 'pool1') ...",
         "serialzer dump data for key ('analyzed_data', 'pool2') ...",
         "serialzer dump data for key ('cleaned_data', 'pool1') ...",
         "serialzer dump data for key ('cleaned_data', 'pool1') ...",
         "serialzer dump data for key ('cleaned_data', 'pool2') ...",
         "serialzer dump data for key ('cleaned_data', 'pool2') ...",
         "serialzer dump data for key ('data', 'pool1') ...",
         "serialzer dump data for key ('data', 'pool1') ...",
         "serialzer dump data for key ('data', 'pool1') ...",
         "serialzer dump data for key ('data', 'pool2') ...",
         "serialzer dump data for key ('data', 'pool2') ...",
         "serialzer dump data for key ('data', 'pool2') ...",
         "serialzer load data for key ('analyzed_data', 'pool1') ...",
         "serialzer load data for key ('analyzed_data', 'pool2') ...",
         ]
    assert not err


def test_cluster(capsys):
    with Client():
        global IS_COMPUTED
        IS_COMPUTED = dict()
        g = setup_graph()
        data = g.get([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
        assert isinstance(data, list)
        assert data == ['analyzed_cleaned_data', 'analyzed_cleaned_data']
        out, err = capsys.readouterr()
        assert not out
        # assert not err


def test_async_run(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    with Client():
        g = setup_graph()
        # persist assert en error because the given collection is not of type
        # dask.base.Base
        futures = g.run(keys=('cleaned_data', 'pool1'))
        data = g.results(futures).values()[0]
        assert data == 'cleaned_data'


def test_async_run_all(capsys):
    global IS_COMPUTED
    IS_COMPUTED = dict()
    with Client() as client:
        g = setup_graph()
        # persist assert en error because the given collection is not of type
        # dask.base.Base
        futures = g.run()
        data = g.results(futures).values()

        assert sorted(data) == ['analyzed_cleaned_data', 'analyzed_cleaned_data',
                                'cleaned_data', 'cleaned_data', 'data', 'data']

        data = client.gather(futures)
        # here I do not know why gather still return delayed objects...
        # assert isinstance(data[0], str)


def test_visualize_not_computed(tmpdir):
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    g.visualize(format='svg')


def test_compute_method():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    data = g.compute()
    assert data == ('analyzed_cleaned_data', 'analyzed_cleaned_data')


def test_get_no_keys():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    data = g.get()
    assert data == ('analyzed_cleaned_data', 'analyzed_cleaned_data')


def test_persist_method():
    global IS_COMPUTED
    IS_COMPUTED = dict()
    with Client():
        g = setup_graph()
        data = g.persist()
        assert type(data) == PersistentDAG


def test_visualize_computed(tmpdir):
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)
    global IS_COMPUTED
    IS_COMPUTED = dict()
    g = setup_graph()
    g.compute()
    g.visualize(format='svg')
    g.visualize(format='svg', raw_dask=False)


def test_serializer_correctly_setted_with_some_task_not_named(tmpdir):
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)
    global IS_COMPUTED
    IS_COMPUTED = dict()

    g = PersistentDAG()
    serializer = Serializer()
    for pool in ['pool1', 'pool2']:
        g.add_task(load_data,
                   dask_key_name=('data', pool),
                   dask_serializer=serializer)
        g.add_task(clean_data, ('data', pool),
                   dask_key_name=('cleaned_data', pool),
                   # dask_serializer=serializer
                   )
        g.add_task(analyze_data, ('cleaned_data', pool),
                   # dask_key_name=('analyzed_data', pool),
                   dask_serializer=serializer)

    g.compute()
    assert len(g.serializers) == 4
    g.visualize(fogrmat='svg')
    g.visualize(format='svg', raw_dask=False)


def test_df():
    import pandas as pd
    data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    g = PersistentDAG()

    def f(x): return x
    g.add_task(f, data)
    result = g.compute()
    assert isinstance(result, pd.DataFrame)
    assert result.equals(data)
