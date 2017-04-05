# import pytest
# from time import sleep
from ..dag import DAG
# global variable to simulate the fact to have serialize data somewhere

from .helpers import load_data, clean_data, analyze_data


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


def test_get(capsys):
    g = setup_graph()
    data = g.get(('data', 'pool1'))
    assert data == 'data'
    data = g.get(('cleaned_data', 'pool1'))
    assert data == 'cleaned_data'
    data = g.get(('analyzed_data', 'pool1'))
    assert data == 'analyzed_cleaned_data'
