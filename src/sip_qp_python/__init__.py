from typing import Optional, Union

import numpy as np
import scipy.sparse as spa

from .sip_qp_python_ext import (
    FAILED_CHECK,
    FACTORIZATION_FAILURE,
    ITERATION_LIMIT,
    LINE_SEARCH_FAILURE,
    LINE_SEARCH_ITERATION_LIMIT,
    LOCALLY_INFEASIBLE,
    SOLVED,
    SUBOPTIMAL,
    TIMEOUT,
    BarrierSettings,
    LineSearchSettings,
    LoggingSettings,
    Mode,
    PenaltySettings,
    RegularizationSettings,
    Result,
    ScalingSettings,
    Settings,
    SipSettings,
    SipTerminationSettings,
    Solver,
    SolverInfo,
    Status,
    TerminationSettings,
)

__all__ = [
    "FAILED_CHECK",
    "FACTORIZATION_FAILURE",
    "ITERATION_LIMIT",
    "LINE_SEARCH_FAILURE",
    "LINE_SEARCH_ITERATION_LIMIT",
    "LOCALLY_INFEASIBLE",
    "SOLVED",
    "SUBOPTIMAL",
    "TIMEOUT",
    "BarrierSettings",
    "LineSearchSettings",
    "LoggingSettings",
    "Mode",
    "PenaltySettings",
    "RegularizationSettings",
    "Result",
    "ScalingSettings",
    "Settings",
    "SipSettings",
    "SipTerminationSettings",
    "Solver",
    "SolverInfo",
    "Status",
    "TerminationSettings",
    "solve_qp",
]


def solve_qp(
    P: Union[np.ndarray, spa.spmatrix],
    q: np.ndarray,
    G: Optional[Union[np.ndarray, spa.spmatrix]] = None,
    h: Optional[np.ndarray] = None,
    A: Optional[Union[np.ndarray, spa.spmatrix]] = None,
    b: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    initvals: Optional[np.ndarray] = None,
    *,
    kkt_inverse_permutation: np.ndarray,
    settings: Optional[Settings] = None,
    time_limit_s: float = float("inf"),
) -> Result:
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError("q must be a vector")
    n = q.shape[0]
    P = spa.csc_matrix(P, dtype=np.float64)
    G = spa.csr_matrix((0, n)) if G is None else spa.csr_matrix(G, dtype=np.float64)
    A = spa.csr_matrix((0, n)) if A is None else spa.csr_matrix(A, dtype=np.float64)
    if P.shape != (n, n) or G.shape[1] != n or A.shape[1] != n:
        raise ValueError("inconsistent QP matrix dimensions")

    def vector_or_default(value, size, default):
        if value is None:
            return np.full(size, default, dtype=np.float64)
        result = np.ascontiguousarray(value, dtype=np.float64)
        if result.ndim != 1 or result.shape[0] != size:
            raise ValueError("inconsistent QP vector dimensions")
        return result

    h = vector_or_default(h, G.shape[0], 0.0)
    b = vector_or_default(b, A.shape[0], 0.0)
    lb = vector_or_default(lb, n, -np.inf)
    ub = vector_or_default(ub, n, np.inf)
    initvals = vector_or_default(initvals, n, 0.0)
    kkt_inverse_permutation = np.ascontiguousarray(
        kkt_inverse_permutation, dtype=np.int32
    )

    P.sum_duplicates()
    P.eliminate_zeros()
    P.sort_indices()
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()
    G.sum_duplicates()
    G.eliminate_zeros()
    G.sort_indices()

    solver = Solver(
        P,
        q,
        G,
        h,
        A,
        b,
        lb,
        ub,
        kkt_inverse_permutation,
        Settings() if settings is None else settings,
        time_limit_s,
    )
    return solver.solve(initvals)
