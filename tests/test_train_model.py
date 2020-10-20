import numpy as np
import pytest

from src.models.train_model import (
    sigmoid,
    initialize_with_zeros,
    propagate,
    optimize,
    predict
)


def test_sigmoid():
    assert np.allclose(sigmoid(np.array([0, 2])), np.array([0.5, 0.88079708]))


# @pytest.mark.skip
def test_initialize_zeros():
    dim = 2
    w, b = initialize_with_zeros(dim)

    assert np.allclose(w, np.array([[0], [0]]))
    assert b == 0


# @pytest.mark.skip
def test_propagate():
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    grads, cost = propagate(w, b, X, Y)

    assert np.allclose(grads["dw"], np.array([[ 0.99845601], [ 2.39507239]]))
    assert np.allclose(grads["db"], 0.00145557813678)
    assert np.allclose(cost, 5.801545319394553)


# @pytest.mark.skip
def test_optimize():
    w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
    
    assert np.allclose(params["w"], np.array([[ 0.19033591], [ 0.12259159]]))
    assert np.allclose(params["b"], 1.92535983008)
    assert np.allclose(grads["dw"], np.array([[ 0.67752042], [ 1.41625495]]))
    assert np.allclose(grads["db"], 0.219194504541)


@pytest.mark.skip
def test_predict():
    w = np.array([[0.1124579],[0.23106775]])
    b = -0.3
    X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])

    assert np.allclose(predict(w, b, X), [[1.0, 1.0, 0.0]])
