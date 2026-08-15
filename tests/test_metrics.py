import numpy as np
from src.metrics import coefficient_relative_error, support_scores


def test_zero_coefficient_error():
    A = np.array([[1., 0.], [0., 2.]])
    assert coefficient_relative_error(A, A) == 0.0


def test_perfect_support():
    A = np.array([[1., 0.], [0., 2.]])
    assert support_scores(A, A) == (1.0, 1.0, 1.0)
