import numpy as np
import pytest
import scipy.sparse as spa

from sip_qp_python import Status, solve_qp


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
