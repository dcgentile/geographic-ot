#!/usr/bin/env python3


import ot
from utils import uniform_measure
import numpy as np
from scipy.optimize import minimize


"""
for a given cost matrix and regularization paramater epsilon,
return e^(-C/epsilon), exponentiation performed componentwise
we perform this regularization so that the entropically regularized
Wasserstein distance can be expressed in terms of a KL Divergence
"""


def regularize_cost(matrix, epsilon):
    return np.exp(-matrix / epsilon)


"""
project a vector onto the simplex
"""


def logarithmic_change_of_variable(coords):
    v = np.exp(coords)
    n = np.linalg.norm(v, 1)
    return v / n


"""
given a pair of measures and cost matrix, build a matrix
whose columns are points along the geodesic path from the first measure to the second
"""


def build_geodesic(
    measures, cost, epsilon=0.1, loss="sqeuc", method="total", steps=10, iters=2048
):
    node_count, measure_count = measures.shape
    assert measure_count == 2
    geodesic_steps = steps
    barycenters = np.empty((node_count, geodesic_steps + 1))
    if method == "total":
        print("computing with TOTAL implementation")
        if loss == "sqeuc":
            loss = sqeuc_loss_grad
        elif loss == "kl":
            loss = kl_loss
        elif loss == "l1":
            loss = ell_one_loss
        for i in range(0, geodesic_steps + 1):
            coordinates = np.array([1 - i / geodesic_steps, i / geodesic_steps])
            barycenters[:, i] = barycenter(
                coordinates, measures, None, cost, epsilon, iters=iters
            )
    else:
        print("computing using Python OT library")
        for i in range(0, geodesic_steps):
            coordinates = np.array([1 - i / geodesic_steps, i / geodesic_steps])
            barycenters[:, i] = ot.barycenter(
                measures, cost, epsilon, weights=coordinates
            )
    return barycenters


def sqeuc_loss_grad(p, q):
    return p - q


def ell_one_loss(p, q):
    return np.sign(p - q)


def kl_loss(p, q):
    return np.log(p / q)


def barycenter(coords, measures, target, cost, epsilon, iters=256):
    barycenter, _ = sinkhorn_differentiate(
        coords, measures, target, cost, epsilon, iters=iters
    )
    return barycenter


def sinkhorn_differentiate(
    coords,
    measures,
    target,
    cost,
    epsilon,
    iters,
):
    num_nodes, num_measures = measures.shape
    b = np.empty((num_nodes, num_measures, iters))
    b[:, :, 0] = np.ones((num_nodes, num_measures))
    w = np.zeros(num_measures)
    r = np.zeros((num_nodes, num_measures))
    phi = np.empty((num_nodes, num_measures, iters))
    k = regularize_cost(cost, epsilon)
    for l in range(1, iters):
        b_l = np.empty((num_nodes, num_measures))
        for s in range(num_measures):
            m = measures[:, s]
            b_m = b[:, s, l - 1]
            kb_m = k @ b_m
            ratio = m / kb_m
            phi[:, s, l] = k.T @ ratio
        p = np.exp(np.dot(np.log(phi[:, :, l]), coords))
        for i in range(num_measures):
            phi_col = phi[:, i, l]
            b_l[:, i] = p / phi_col
        b[:, :, l] = b_l
    # if None was passed as the target, the gradient loop is pointless
    if target is not None:
        # g = loss(p, target) * p
        g = (p - target) * p
        for l in range(iters - 1, 0, -1):
            for m in range(num_measures):
                w[m] = w[m] + np.dot(np.log(phi[:, m, l]), g)
                u = coords[m] * g - r[:, m]
                v = phi[:, m, l]
                b_m = b[:, m, l - 1]
                p_m = measures[:, m]
                x = k @ (u / v)
                y = p_m / ((k @ b_m) ** 2)
                result = (-k.T @ (x * y)) * b_m

                r[:, m] = result
            g = np.sum(r, axis=1)
    else:
        w = None
    return p, w


"""
quantify difference between histograms p and q
"""


def sqeuc_loss(p, q):
    return 0.5 * (np.linalg.norm(p - q, 2) ** 2)


"""
compute the barycenter of measures with coordinates=coordinates,
and return the loss difference between that barycenter and the supplied target
"""


def barycentric_loss(coordinates, measures, target, cost, epsilon):
    # bar = barycenter(logarithmic_change_of_variable(coordinates), measures, target, cost)
    bar, _ = sinkhorn_differentiate(
        logarithmic_change_of_variable(coordinates),
        measures,
        target,
        cost,
        epsilon,
        iters=256,
    )
    return sqeuc_loss(bar, target)


"""
wrapper to return the "gradient" component returned by the Bonneel algorithm
"""


def loss_gradient(coords, measures, cost, target, epsilon):
    return sinkhorn_differentiate(
        logarithmic_change_of_variable(coords),
        measures,
        cost,
        target,
        epsilon,
        iters=256,
    )[1]


"""
given a family of measures, a target histogram, a cost matrix and a regularization parameter epsilon
find the barycentric coordinates that best approximate the target w.r.t the reference measures
"""


def simplex_regression(measures, target, cost, epsilon):
    num_measures = measures.shape[1]
    x0 = uniform_measure(num_measures)
    args = (measures, target, cost, epsilon)
    lam = minimize(
        barycentric_loss, x0, args=args, jac=loss_gradient, method="L-BFGS-B"
    ).x
    return logarithmic_change_of_variable(lam)
