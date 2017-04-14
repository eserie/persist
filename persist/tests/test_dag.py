import os
import pytest
import dask
from distributed.client import Client
# from time import sleep
from ..dag import DAG
from .helpers import load_data, clean_data, analyze_data
dask.set_options(get=dask.async.get_sync)


def setup_graph(**kwargs):
    g = DAG(**kwargs)
    for pool in ['pool1', 'pool2']:
        g.add_task(load_data, dask_key_name=(
            'data', pool))
        g.add_task(clean_data, ('data', pool),
                   dask_key_name=('cleaned_data', pool))
        g.add_task(analyze_data, ('cleaned_data', pool),
                   dask_key_name=('analyzed_data', pool))
    return g


def test_submit_api():
    g = DAG()

    for pool in ['pool1', 'pool2']:
        g.submit(load_data,
                 dask_key_name=('data', pool),
                 )
        g.submit(clean_data, ('data', pool),
                 dask_key_name=('cleaned_data', pool),
                 )
        g.submit(analyze_data, ('cleaned_data', pool),
                 dask_key_name=('analyzed_data', pool),
                 )
    futures = g.run()
    data = g.results(futures)
    assert data == {('analyzed_data', 'pool1'): 'analyzed_cleaned_data',
                    ('analyzed_data', 'pool2'): 'analyzed_cleaned_data',
                    ('cleaned_data', 'pool1'): 'cleaned_data',
                    ('cleaned_data', 'pool2'): 'cleaned_data',
                    ('data', 'pool1'): 'data',
                    ('data', 'pool2'): 'data'}


def test_delayed_api():
    g = DAG()

    for pool in ['pool1', 'pool2']:
        g.delayed(load_data)(dask_key_name=(
            'data', pool), )
        g.delayed(clean_data)(('data', pool),
                              dask_key_name=('cleaned_data', pool),
                              )
        g.delayed(analyze_data)(('cleaned_data', pool),
                                dask_key_name=('analyzed_data', pool),
                                )
    futures = g.run()
    data = g.results(futures)
    data = g.results(futures)
    assert data == {('analyzed_data', 'pool1'): 'analyzed_cleaned_data',
                    ('analyzed_data', 'pool2'): 'analyzed_cleaned_data',
                    ('cleaned_data', 'pool1'): 'cleaned_data',
                    ('cleaned_data', 'pool2'): 'cleaned_data',
                    ('data', 'pool1'): 'data',
                    ('data', 'pool2'): 'data'}


def test_key_none():
    g = DAG()

    g.add_task(func=load_data, option=10)
    futures = g.run()
    data = g.results(futures)
    assert data.values() == ["data_{'option': 10}"]
    assert data.keys()[0].startswith('load_data-')
    keys = g.dask.keys()
    assert len(keys) == 1
    assert keys[0] is not None
    assert keys[0].startswith('load_data-')


def test_key_none_serializer_none():
    g = DAG()
    g.add_task(load_data, option=10)
    futures = g.run()
    data = g.results(futures)
    assert data.values() == ["data_{'option': 10}"]
    assert data.keys()[0].startswith('load_data-')


def test_kwargs():
    g = DAG()
    g.add_task(dask_key_name='data',
               func=load_data, option=10)
    futures = g.run()
    data = g.results(futures)
    assert data == {'data': "data_{'option': 10}"}


def test_varargs():
    g = DAG()

    varargs = (10,)
    g.add_task(load_data, *varargs, dask_key_name='data')
    futures = g.run()
    data = g.results(futures)
    assert data == {'data': "data_(10,)"}


def test_use_already_used_key():
    g = DAG()

    g.add_task(dask_key_name='key_data1', func=load_data, option=10)
    with pytest.raises(AssertionError) as err:
        g.add_task(dask_key_name='key_data1', func=load_data, option=20)
    err = str(err)
    assert err.endswith("key is already used")


def test_add_task_in_good_order():
    g = DAG()
    g.add_task(func=load_data,
               dask_key_name=('data', 'pool1'))
    g.add_task(clean_data, ('data', 'pool1'),
               dask_key_name=('cleaned_data', 'pool1'))
    g.add_task(analyze_data, ('cleaned_data', 'pool1'),
               dask_key_name=('analyzed_data', 'pool1'))
    data = g.compute()
    assert data == 'analyzed_cleaned_data'


def test_add_task_in_wrong_order():
    with pytest.raises(AssertionError) as err:
        g = DAG()
        g.add_task(analyze_data, ('cleaned_data', 'pool1'),
                   dask_key_name=('analyzed_data', 'pool1'))
        g.add_task(clean_data, ('data', 'pool1'),
                   dask_key_name=('cleaned_data', 'pool1'))
        g.add_task(func=load_data,
                   dask_key_name=('data', 'pool1'))
        g.compute()
    err = str(err)
    assert err.endswith("isinstance(('cleaned_data', 'pool1'), str)")


def test_only_analyze():
    g = DAG()
    g.add_task(analyze_data, 'clean_data',
               dask_key_name='analyze_data')

    g.add_task(func=load_data, option=10,
               dask_key_name='key_data1')


def test_varargs_deps():
    g = DAG()

    g.add_task(func=load_data, option=10,
               dask_key_name='key_data1')
    g.add_task(func=load_data, option=20,
               dask_key_name='key_data2')
    varargs = ('key_data1', 'key_data2',)
    g.add_task(clean_data, *varargs,
               dask_key_name='cleaned_data')
    futures = g.run()
    data = g.results(futures)
    assert data == {'key_data1': "data_{'option': 10}",
                    'key_data2': "data_{'option': 20}",
                    'cleaned_data': "cleaned_data_{'option': 10}_data_{'option': 20}",
                    }


def test_kwargs_deps():
    g = DAG()

    g.add_task(dask_key_name='key_data1', func=load_data, option=10)
    g.add_task(dask_key_name='key_data2', func=load_data, option=20)
    kwargs = dict(data='key_data1', other='key_data2')
    g.add_task(dask_key_name='cleaned_data', func=clean_data, **kwargs)
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
    g = DAG()
    data = g.delayed(load_data)(dask_key_name=('data', 'pool1'))
    assert data.compute() == 'data'
    cleaned_data = g.delayed(clean_data)(
        dask_key_name=('cleaned_data', 'pool1'),
        data=data)
    assert cleaned_data.compute() == 'cleaned_data'


def test_get(capsys):
    g = setup_graph()
    data = g.get(('data', 'pool1'))
    assert data == 'data'
    data = g.get(('cleaned_data', 'pool1'))
    assert data == 'cleaned_data'
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_cleaned_data'


def test_get_multiple_times(capsys):
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
    assert isinstance(data, tuple)

    # check printed messages
    out, err = capsys.readouterr()
    assert out == """load data ...
clean data ...
analyze data ...
load data ...
clean data ...
analyze data ...
load data ...
clean data ...
load data ...
clean data ...
load data ...
clean data ...
analyze data ...
load data ...
clean data ...
analyze data ...
load data ...
clean data ...
analyze data ...
"""
    assert not err


def test_run(capsys):
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


def test_cluster(capsys):
    with Client():
        g = setup_graph()
        data = g.get([('analyzed_data', 'pool1'), ('analyzed_data', 'pool2')])
        assert isinstance(data, list)
        assert data == ['analyzed_cleaned_data', 'analyzed_cleaned_data']
        out, err = capsys.readouterr()
        assert not out
        # assert not err


def test_async_run(capsys):
    with Client():
        g = setup_graph()
        # persist assert en error because the given collection is not of type
        # dask.base.Base
        futures = g.run(keys=('cleaned_data', 'pool1'))
        data = g.results(futures).values()[0]
        assert data == 'cleaned_data'


def test_async_run_all(capsys):
    with Client() as client:
        g = setup_graph()
        # persist assert en error because the given collection is not of type
        # dask.base.Base
        futures = g.run()
        data = g.results(futures)
        assert sorted(data.values()) == ['analyzed_cleaned_data', 'analyzed_cleaned_data',
                                         'cleaned_data', 'cleaned_data', 'data', 'data']

        data = client.gather(futures)
        # TODO: here I do not know why gather still return delayed objects...
        # assert isinstance(data[0], str)


def test_visualize(tmpdir):
    tmpdir = str(tmpdir)
    os.chdir(tmpdir)
    g = setup_graph()
    g.visualize(format='svg')


def test_compute_method():
    g = setup_graph()
    data = g.compute()
    assert data == ('analyzed_cleaned_data', 'analyzed_cleaned_data')


def test_get_with_no_argument():
    g = setup_graph()
    data = g.get()
    assert data == ('analyzed_cleaned_data', 'analyzed_cleaned_data')


def test_persist_method():
    g = setup_graph()
    data = g.persist()
    assert type(data) == DAG


def test_terminal_node():
    g = setup_graph()
    terminal_nodes = g.terminal_nodes
    assert terminal_nodes == [
        ('analyzed_data', 'pool2'), ('analyzed_data', 'pool1')]


def test_dask_to_digraph():
    from ..dag import dask_to_digraph, digraph_to_dask
    g = setup_graph()
    graph = dask_to_digraph(g.dask)
    dsk = digraph_to_dask(graph)
    assert dsk.keys() == g.dask.keys()
    assert dsk == g.dask
    g2 = DAG.from_digraph(graph)
    data = g2.compute()
    assert data == ('analyzed_cleaned_data', 'analyzed_cleaned_data')


def test_df():
    import pandas as pd
    data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    g = DAG()

    def f(x): return x
    g.add_task(f, data)
    result = g.compute()
    assert isinstance(result, pd.DataFrame)
    assert result.equals(data)
