import math

import numpy as np
import pytest
import scipy.sparse as spa

from sip_qp_python import Settings, Status, solve_qp


def test_solve_qp_uses_native_variable_bounds():
    result = solve_qp(
        P=spa.csc_matrix([[4.0, 1.0], [1.0, 2.0]]),
        q=np.array([1.0, 1.0]),
        A=spa.csr_matrix([[1.0, 1.0]]),
        b=np.array([1.0]),
        lb=np.array([0.0, 0.0]),
        ub=np.array([0.7, 0.7]),
        kkt_inverse_permutation=np.array([0, 1, 2]),
    )

    assert result.info.exit_status == Status.SOLVED
    assert result.x == pytest.approx([0.3, 0.7], abs=1e-5)
    assert result.z.shape == (0,)
    assert result.z_box[0] == pytest.approx(0.0, abs=1e-5)
    assert result.z_box[1] > 0.0


def test_solve_qp_accepts_user_defined_kkt_permutation():
    result = solve_qp(
        P=spa.csc_matrix([[4.0, 1.0], [1.0, 2.0]]),
        q=np.array([1.0, 1.0]),
        A=spa.csr_matrix([[1.0, 1.0]]),
        b=np.array([1.0]),
        lb=np.array([0.0, 0.0]),
        ub=np.array([0.7, 0.7]),
        kkt_inverse_permutation=np.array([0, 1, 2]),
    )

    assert result.info.exit_status == Status.SOLVED
    assert result.x == pytest.approx([0.3, 0.7], abs=1e-5)


def test_termination_validates_gap_reconstructed_from_returned_bound_dual():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    b = np.array([1.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([0.7, 0.7])
    tolerance = 1e-3
    settings = Settings()
    settings.termination.max_absolute_residual = tolerance
    settings.termination.max_relative_residual = 0.0
    settings.termination.max_absolute_duality_gap = tolerance
    settings.termination.max_relative_duality_gap = 0.0

    result = solve_qp(
        P=spa.csc_matrix(P),
        q=q,
        A=spa.csr_matrix([[1.0, 1.0]]),
        b=b,
        lb=lb,
        ub=ub,
        kkt_inverse_permutation=np.array([0, 1, 2]),
        settings=settings,
    )

    returned_gap = abs(
        math.fsum(
            [
                *(
                    float(P[row, column])
                    * float(result.x[row])
                    * float(result.x[column])
                    for row, column in zip(*np.nonzero(P))
                ),
                float(q @ result.x),
                float(b @ result.y),
                *(
                    float(bound) * min(float(dual), 0.0)
                    for bound, dual in zip(lb, result.z_box)
                ),
                *(
                    float(bound) * max(float(dual), 0.0)
                    for bound, dual in zip(ub, result.z_box)
                ),
            ]
        )
    )

    assert result.info.exit_status == Status.SOLVED
    assert returned_gap < tolerance


def test_solve_qp_equilibrates_large_constraint_coefficients():
    result = solve_qp(
        P=spa.eye(1, format="csc"),
        q=np.zeros(1),
        G=spa.csr_matrix([[-1e6]]),
        h=np.array([-1e6]),
        kkt_inverse_permutation=np.array([0, 1]),
    )

    assert result.info.exit_status == Status.SOLVED
    assert result.x == pytest.approx([1.0], abs=1e-5)
