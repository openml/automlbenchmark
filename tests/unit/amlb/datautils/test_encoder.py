import numpy as np
import pytest

from amlb.datautils import Encoder


@pytest.mark.parametrize(
    ['labels', 'to_encode', 'encoded'],
    [
        (['c', 'a', 'b'], ['a', 'b', 'c', 'b', 'c', 'a'], [0, 1, 2, 1, 2, 0]),
        (['a', 'b', 'c'], ['a', 'b', 'c', 'b', 'c', 'a'], [0, 1, 2, 1, 2, 0]),
    ])
def test_encoder_sorts_the_labels_in_lexicographic_order(labels, to_encode, encoded):
    e = Encoder().fit(labels)
    transformed = e.transform(to_encode)
    assert (encoded == transformed).all()


@pytest.mark.parametrize(
    ['labels', 'to_encode', 'encoded'],
    [
        (['a', 'A'], ['A', 'a', 'a', 'A'], [0, 1, 1, 0]),
        (['a', ' a'], [' a', 'a', ' a', 'a'], [0, 1, 0, 1]),
        (['a', 'a '], ['a ', 'a', 'a ', 'a'], [1, 0, 1, 0]),
        (['a', ' a '], [' a ', 'a', ' a ', 'a'], [0, 1, 0, 1]),
    ])
def test_encoder_does_not_modify_categorical_values_by_default(labels, to_encode, encoded):
    e = Encoder().fit(labels)
    transformed = e.transform(to_encode)
    assert (encoded == transformed).all()


@pytest.mark.parametrize(
    ['labels', 'to_encode', 'encoded'],
    [
        (['a', ' a'], [' a', 'a', ' a', 'a'], [0, 0, 0, 0]),
        (['a', 'a '], ['a ', 'a', 'a ', 'a'], [0, 0, 0, 0]),
        (['a', ' a '], [' a ', 'a', ' a ', 'a'], [0, 0, 0, 0]),
        (['a', 'b', ' a ', ' a', 'a '], [' a ', 'a', 'b',  'a ', ' a'], [0, 0, 1, 0, 0]),
    ])
def test_encoder_can_trim_categorical_values(labels, to_encode, encoded):
    normalize = lambda v: np.char.strip(np.asarray(v).astype(str))
    e = Encoder(normalize_fn=normalize).fit(labels)
    transformed = e.transform(to_encode)
    assert (encoded == transformed).all()


@pytest.mark.parametrize(
    ['labels', 'to_encode', 'encoded'],
    [
        (['a', 'A'], ['A', 'a', 'a', 'A'], [0, 0, 0, 0]),
        (['a', 'b', 'A'], ['A', 'a', 'b', 'a', 'A'], [0, 0, 1, 0, 0]),
    ])
def test_encoder_can_make_categorical_values_case_insensitive(labels, to_encode, encoded):
    normalize = lambda v: np.char.lower(np.asarray(v).astype(str))
    e = Encoder(normalize_fn=normalize).fit(labels)
    transformed = e.transform(to_encode)
    assert (encoded == transformed).all()
