"""
    Multi-Objective Optimization (MOO) util functions for EARN
"""

import numpy as np
from sortedcontainers import SortedSet as sortset
from hvwfg import wfg
from math import log10


def dominates(s1: np.ndarray, s2: np.ndarray) -> bool:
    # Solution s1 dominates s2 if minimizes the objectives
    assert s1.shape[0] == s2.shape[0] and len(s1.shape) == 1 and s1.shape == s2.shape
    return np.all(s1 <= s2) and np.any(s1 < s2)


def non_dominated_selection(obj: np.ndarray) -> np.ndarray:
    assert len(obj.shape) == 2

    dominators = []
    for ri, r in enumerate(obj):
        n = 0
        for qi, q in enumerate(obj):
            if ri != qi:
                if dominates(q, r):
                    n += 1
        if n == 0:
            dominators.append(ri)

    return np.array(dominators)


def fast_non_dominated_sort(obj: np.ndarray) -> np.ndarray:
    assert len(obj.shape) == 2

    rank = -np.ones(obj.shape[0], dtype=np.int)  # Rank of the solution
    N = -np.ones(rank.shape)  # Dominant solutions
    S = []  # Dominated solutions
    F = set()  # Current front

    for ri, r in enumerate(obj):
        Sr = []
        n = 0
        for qi, q in enumerate(obj):
            if ri != qi:
                if dominates(r, q):
                    Sr.append(qi)
                elif dominates(q, r):
                    n += 1
        S.append(Sr)
        N[ri] = n

        if n == 0:
            rank[ri] = 1
            F.add(ri)

    i = 1
    while len(F) > 0:
        Q = set()
        for ri in F:
            for qi in S[ri]:
                N[qi] -= 1
                if N[qi] == 0:
                    rank[qi] = i+1
                    Q.add(qi)
        i += 1
        F = Q

    return rank


def compute_crowding_distance(obj: np.ndarray) -> np.ndarray:
    assert len(obj.shape) == 2

    D = np.zeros(obj.shape[0])
    for j in range(obj.shape[1]):
        obj_id_sorted = np.argsort(obj[:, j])
        D[obj_id_sorted[0]] = D[obj_id_sorted[-1]] = float("inf")
        for i in range(1, obj.shape[0] - 1):
            D[obj_id_sorted[i]] += obj[obj_id_sorted[i+1], j] - obj[obj_id_sorted[i-1], j]
    return D


def hvolume(obj: np.ndarray, ref: np.ndarray=None) -> float:
    assert len(obj.shape) == 2

    if ref is None:
        ref = np.ones(obj.shape[1])

    valid = np.all(np.less(obj, ref), axis=1)
    obj = obj[valid]
    obj = obj[non_dominated_selection(obj)]
    hv = wfg(obj, ref)

    return hv


def area_rect(h, w):
    a = 0
    if h > 0 and w > 0:
        magnitude = pow(10, -round(log10(min(h, w))-0.5))
        a = (h*magnitude) * (w*magnitude)
        a /= magnitude
    return a


def compute_hvolume(obj: np.ndarray, r: np.ndarray) -> float:

    """
    Hypervolume computation algorithm for 3 objectives (Beume.N et al 2009)
    :param r: 1x3 np array. Reference point
    :param obj: Nx3 nparray. Volume contributing points (Ensemble evlauations).
    :return: Hyper volume generated by points in obj with respect to r
    """

    assert obj.shape[1] == 3, "Hypervolume computation only for 3 objectives"

    obj = r-obj  # Time, accuracy and params normalized values relative to reference NN
    valid = np.logical_and(obj[:, 0] > 0, obj[:, 1] > 0)
    valid = np.logical_and(obj[:, 2] > 0, valid)
    valid = np.where(valid)[0]
    obj = obj[valid]
    obj = np.unique(obj, axis=0)
    V = 0

    if len(valid) > 0:
        # Numpy not hashable, work with list and tuple
        order_x3 = np.argsort(-obj[:, 0], axis=0)
        points_x12 = [(arow[1], arow[2]) for arow in obj]
        points_x3 = list(obj[:, 0])

        # Step 1: Initialize algorithm
        s = sortset([(0, float("inf")), (float("inf"), 0)])

        # Step 2: Process the first point
        p1i = order_x3[0]
        p1 = points_x12[p1i]
        A = area_rect(p1[0], p1[1])
        zi = p1i
        s.add(p1)

        # Step 3: Process all other points
        for i in order_x3[1:]:
            pi = i
            p = points_x12[pi]
            qi = s.bisect_right(p)
            q = s[qi]
            if p[1] > q[1]:
                V += area_rect(A, points_x3[zi] - points_x3[pi])  # Volume re-using area formula
                zi = pi
                ti = qi-1
                while p[1] > s[ti][1]:
                    t = s[ti]
                    A -= area_rect(t[0] - s[ti-1][0], t[1] - q[1])
                    s.remove(t)
                    ti = ti - 1
                t = s[ti]
                A += area_rect(p[0] - t[0], p[1] - q[1])
                s.add(p)

        V += area_rect(A, points_x3[zi])
    return V


