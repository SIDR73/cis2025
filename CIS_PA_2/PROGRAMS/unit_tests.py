import numpy as np
import pytest
import os
import sys
from scipy.spatial.transform import Rotation
from scipy.special import comb

# cartesian math tests

def test_pos_3D():
    """test 3D position vector creation"""
    result = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_Rot_X_identity():
    """test x-axis rotation at 0 degrees returns identity"""
    def Rot_X(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    
    R = Rot_X(0)
    np.testing.assert_array_almost_equal(R, np.eye(3))

def test_Rot_X_90_degrees():
    """Test X-axis rotation at 90 degrees"""
    def Rot_X(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    
    R = Rot_X(np.pi/2)
    expected = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    np.testing.assert_array_almost_equal(R, expected)

def test_Rot_Y_identity():
    """Test Y-axis rotation at 0 degrees returns identity"""
    def Rot_Y(theta):
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    R = Rot_Y(0)
    np.testing.assert_array_almost_equal(R, np.eye(3))

def test_Rot_Z_identity():
    """Test Z-axis rotation at 0 degrees returns identity"""
    def Rot_Z(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    R = Rot_Z(0)
    np.testing.assert_array_almost_equal(R, np.eye(3))

def test_rotation_orthogonality():
    """Test that rotation matrices are orthogonal (R^T * R = I)"""
    def Rot_X(theta):
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])
    
    theta = np.pi/4
    R = Rot_X(theta)
    result = R.T @ R
    np.testing.assert_array_almost_equal(result, np.eye(3))

def test_rotation_determinant():
    """Test that rotation matrices have determinant = 1"""
    def Rot_Y(theta):
        return np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    
    theta = np.pi/3
    R = Rot_Y(theta)
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 1e-10

def test_rotate_simple():
    """Test rotation of a point"""
    def rotate(R, P):
        P = np.asarray(P).reshape(3,)
        return R @ P
    
    R = np.eye(3)  # rot. identity
    P = np.array([1, 2, 3])
    result = rotate(R, P)
    np.testing.assert_array_almost_equal(result, P)

def test_translate_simple():
    """translation of a point test"""
    def translate(P, d):
        P, d = np.asarray(P).reshape(3,), np.asarray(d).reshape(3,)
        return P + d
    
    P = np.array([1, 2, 3])
    d = np.array([4, 5, 6])
    result = translate(P, d)
    expected = np.array([5, 7, 9])
    np.testing.assert_array_almost_equal(result, expected)

def test_frame_transformation_identity():
    """frame transformation with identity rotation and zero translation test"""
    def frame_transformation(R, Pin, d):
        Pin, d = np.asarray(Pin).reshape(3,), np.asarray(d).reshape(3,)
        R = np.asarray(R).reshape(3, 3)
        return R @ Pin + d
    
    R = np.eye(3)
    P = np.array([1, 2, 3])
    d = np.zeros(3)
    result = frame_transformation(R, P, d)
    np.testing.assert_array_almost_equal(result, P)

def test_frame_transformation_combined():
    """frame transformation with rotation and translation"""
    def frame_transformation(R, Pin, d):
        Pin, d = np.asarray(Pin).reshape(3,), np.asarray(d).reshape(3,)
        R = np.asarray(R).reshape(3, 3)
        return R @ Pin + d
    
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° rotation around Z
    P = np.array([1, 0, 0])
    d = np.array([1, 1, 1])
    result = frame_transformation(R, P, d)
    expected = np.array([1, 2, 1])
    np.testing.assert_array_almost_equal(result, expected)

# POINT REGISTRATION TESTS

def test_registration_identity():
    """Test registration with identical point sets (non-collinear)"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.shape != B.shape or A.shape[1] != 3:
            raise ValueError("A and B must have same shape (N,3)")
        
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        p = cB - R @ cA
        return R, p
    
    #non-collinear points
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    B = A.copy()
    R, p = point2point_3Dregistration(A, B)
    
    np.testing.assert_array_almost_equal(R, np.eye(3))
    np.testing.assert_array_almost_equal(p, np.zeros(3))

def test_registration_translation_only():
    """registration with only translation"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.shape != B.shape or A.shape[1] != 3:
            raise ValueError("A and B must have same shape (N,3)")
        
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        p = cB - R @ cA
        return R, p
    
    A = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    translation = np.array([5, 3, 2])
    B = A + translation
    
    R, p = point2point_3Dregistration(A, B)
    
    np.testing.assert_array_almost_equal(R, np.eye(3))
    np.testing.assert_array_almost_equal(p, translation)

def test_registration_rotation_only():
    """registration with only rotation"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.shape != B.shape or A.shape[1] != 3:
            raise ValueError("A and B must have same shape (N,3)")
        
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        p = cB - R @ cA
        return R, p
    
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    #90 deg rotation around Z-axis
    R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    B = (R_true @ A.T).T
    
    R, p = point2point_3Dregistration(A, B)
    
    np.testing.assert_array_almost_equal(R, R_true)
    np.testing.assert_array_almost_equal(p, np.zeros(3))

def test_registration_combined():
    """registration with both rotation and translation"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.shape != B.shape or A.shape[1] != 3:
            raise ValueError("A and B must have same shape (N,3)")
        
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        p = cB - R @ cA
        return R, p
    
    A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    R_true = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t_true = np.array([2, 3, 4])
    B = (R_true @ A.T).T + t_true
    
    R, p = point2point_3Dregistration(A, B)
    
    np.testing.assert_array_almost_equal(R, R_true)
    np.testing.assert_array_almost_equal(p, t_true)

def test_registration_reconstruction_accuracy():
    """registration reconstruction test"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.shape != B.shape or A.shape[1] != 3:
            raise ValueError("A and B must have same shape (N,3)")
        
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        p = cB - R @ cA
        return R, p
    
    #get random points
    np.random.seed(42)
    A = np.random.randn(10, 3)
    R_true = Rotation.random(random_state=42).as_matrix()
    t_true = np.array([5, -3, 2])
    B = (R_true @ A.T).T + t_true
    
    R, p = point2point_3Dregistration(A, B)
    
    #reconstruct B from A
    B_reconstructed = (R @ A.T).T + p
    
    np.testing.assert_array_almost_equal(B_reconstructed, B, decimal=10)

def test_registration_shape_mismatch():
    """registration for mismatched shapes check"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        if A.shape != B.shape or A.shape[1] != 3:
            raise ValueError("A and B must have same shape (N,3)")
        
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        
        p = cB - R @ cA
        return R, p
    
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2, 3]])
    
    with pytest.raises(ValueError):
        point2point_3Dregistration(A, B)

# PIVOT CALIBRATION TESTS

def test_pivot_calibration_stationary_point():
    """pivot calibration with fake data"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        p = cB - R @ cA
        return R, p
    
    def pivot_calibration(frames):
        if len(frames) < 2:
            raise ValueError("Need at least two frames.")
        
        ref = np.asarray(frames[0], float)
        G0 = ref.mean(axis=0)
        g_local = ref - G0
        
        A_blocks, b_list = [], []
        for F in frames:
            F = np.asarray(F, float)
            Rk, tk = point2point_3Dregistration(g_local, F - G0)
            A_blocks.append(np.hstack([Rk, -np.eye(3)]))
            b_list.append(-(G0 + tk))
        
        A, b = np.vstack(A_blocks), np.concatenate(b_list)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        p_dimple = sol[3:]
        p_tip = sol[:3]
        return p_tip, g_local
    
    #generate fake data
    pivot_true = np.array([10, 20, 30])
    tip_local_true = np.array([0, 0, 5])
    
    #generate random orientations and translations
    np.random.seed(42)
    frames = []
    for i in range(10):
        R = Rotation.random(random_state=i).as_matrix()
        markers_local = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0]
        ])
        t = pivot_true - R @ tip_local_true
        markers_world = (R @ markers_local.T).T + t
        frames.append(markers_world)
    
    p_tip, g_local = pivot_calibration(frames)
    
    #check function runs
    assert p_tip is not None
    assert g_local is not None
    assert g_local.shape[1] == 3

def test_pivot_calibration_minimum_frames():
    """pivot calibration frame requirement test"""
    def pivot_calibration(frames):
        if len(frames) < 2:
            raise ValueError("Need at least two frames.")
        return None, None
    
    frames = [np.array([[1, 2, 3]])]
    
    with pytest.raises(ValueError):
        pivot_calibration(frames)

# BERNSTEIN POLYNOMIAL TESTS

def test_bernstein_polynomial_degree_0():
    """Bernstein polynomial of degree 0 test"""
    def bernstein_polynomial(n, k, v):
        return comb(n, k) * (v** k) * ((1 - v) ** (n - k))
    
    result = bernstein_polynomial(0, 0, 0.5)
    assert abs(result - 1.0) < 1e-10

def test_bernstein_polynomial_partition_of_unity():
    """Bernstein polynomials sum to 1 test"""
    def bernstein_polynomial(n, k, v):
        return comb(n, k) * (v** k) * ((1 - v) ** (n - k))
    
    degree = 3
    v = 0.6
    total = sum(bernstein_polynomial(degree, k, v) for k in range(degree + 1))
    assert abs(total - 1.0) < 1e-10

def test_bernstein_polynomial_endpoints():
    """Bernstein polynomial at endpoints"""
    def bernstein_polynomial(n, k, v):
        return comb(n, k) * (v** k) * ((1 - v) ** (n - k))
    
    degree = 3
    # At v=0, only B_{n,0} should be non-zero
    assert abs(bernstein_polynomial(degree, 0, 0) - 1.0) < 1e-10
    assert abs(bernstein_polynomial(degree, 1, 0)) < 1e-10
    
    # At v=1, only B_{n,n} should be non-zero
    assert abs(bernstein_polynomial(degree, degree, 1) - 1.0) < 1e-10
    assert abs(bernstein_polynomial(degree, 0, 1)) < 1e-10

def test_create_bernstein_tensors_shape():
    """check Bernstein tensor shape"""
    def bernstein_polynomial(n, k, v):
        return comb(n, k) * (v** k) * ((1 - v) ** (n - k))
    
    def create_bernstein_tensors(q, degree=3, q_min=None, q_max=None):
        q = np.asarray(q)
        N = q.shape[0]
        
        if q_min is None:
            q_min = q.min(axis=0)
        if q_max is None:
            q_max = q.max(axis=0)
        
        scale = np.maximum(q_max - q_min, 1e-8)
        u = (q - q_min) / scale
        
        Bx = np.array([bernstein_polynomial(degree, i, u[:,0]) for i in range(degree+1)]).T
        By = np.array([bernstein_polynomial(degree, j, u[:,1]) for j in range(degree+1)]).T
        Bz = np.array([bernstein_polynomial(degree, k, u[:,2]) for k in range(degree+1)]).T
        
        tensors = np.einsum('pi,pj,pk->pijk', Bx, By, Bz).reshape(N, -1)
        
        return tensors, q_min, q_max
    
    q = np.random.randn(10, 3)
    degree = 3
    
    tensors, q_min, q_max = create_bernstein_tensors(q, degree=degree)
    
    expected_cols = (degree + 1) ** 3
    assert tensors.shape == (10, expected_cols)
    assert q_min.shape == (3,)
    assert q_max.shape == (3,)

def test_bernstein_correction_identity():
    """Bernstein correction with no distortion"""
    def bernstein_polynomial(n, k, v):
        return comb(n, k) * (v** k) * ((1 - v) ** (n - k))
    
    def create_bernstein_tensors(q, degree=3, q_min=None, q_max=None):
        q = np.asarray(q)
        N = q.shape[0]
        
        if q_min is None:
            q_min = q.min(axis=0)
        if q_max is None:
            q_max = q.max(axis=0)
        
        scale = np.maximum(q_max - q_min, 1e-8)
        u = (q - q_min) / scale
        
        Bx = np.array([bernstein_polynomial(degree, i, u[:,0]) for i in range(degree+1)]).T
        By = np.array([bernstein_polynomial(degree, j, u[:,1]) for j in range(degree+1)]).T
        Bz = np.array([bernstein_polynomial(degree, k, u[:,2]) for k in range(degree+1)]).T
        
        tensors = np.einsum('pi,pj,pk->pijk', Bx, By, Bz).reshape(N, -1)
        
        return tensors, q_min, q_max
    
    def bernstein_distortion_correction(C_measured_frames, C_expected_frames, degree=3, lambda_reg=0.0):
        C_meas_all = np.vstack(C_measured_frames)
        C_exp_all = np.vstack(C_expected_frames)
        
        F, q_min, q_max = create_bernstein_tensors(C_meas_all, degree)
        
        correction_models = {
            'degree': degree,
            'points_min': q_min,
            'points_max': q_max,
            'coefficients': []
        }
        
        for coord in range(3):
            P = C_exp_all[:, coord]
            if lambda_reg > 0:
                C = np.linalg.solve(F.T @ F + lambda_reg * np.eye(F.shape[1]), F.T @ P)
            else:
                C, _, _, _ = np.linalg.lstsq(F, P, rcond=None)
            correction_models['coefficients'].append(C)
        
        return correction_models
    
    def applyBernsteincorrection(q, correction_models):
        q = np.asarray(q)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        
        F, _, _ = create_bernstein_tensors(
            q,
            degree=correction_models['degree'],
            q_min=correction_models['points_min'],
            q_max=correction_models['points_max']
        )
        
        corrected = np.zeros_like(q)
        for coord in range(3):
            C = correction_models['coefficients'][coord]
            corrected[:, coord] = F @ C
        
        return corrected
    
    #data with no distortion 
    np.random.seed(42)
    C_measured = [np.random.randn(10, 3) for _ in range(5)]
    C_expected = [frame.copy() for frame in C_measured]  #copy
    
    correction_model = bernstein_distortion_correction(C_measured, C_expected, degree=3)
    
    #test reconstruction
    test_points = C_measured[0][:3]
    corrected = applyBernsteincorrection(test_points, correction_model)
    
    #use decimal=1 tolerance
    np.testing.assert_array_almost_equal(corrected, test_points, decimal=3)

# COMPUTE EXPECTED C TESTS

def test_compute_expected_C_identity():
    """Test expected C computation with identity"""
    def point2point_3Dregistration(A, B):
        A, B = np.asarray(A, float), np.asarray(B, float)
        cA, cB = A.mean(axis=0), B.mean(axis=0)
        A0, B0 = A - cA, B - cB
        H = A0.T @ B0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        p = cB - R @ cA
        return R, p
    
    def frame_transformation(R, Pin, d):
        Pin, d = np.asarray(Pin).reshape(3,), np.asarray(d).reshape(3,)
        R = np.asarray(R).reshape(3, 3)
        return R @ Pin + d
    
    def compute_expected_C(d, a, c, D_frames, A_frames):
        C_expected_frames = []
        for Dj, Aj in zip(D_frames, A_frames):
            R_d, p_d = point2point_3Dregistration(d, Dj)
            R_a, p_a = point2point_3Dregistration(a, Aj)
            R_d_inv, p_d_inv = R_d.T, -R_d.T @ p_d
            R_comb, p_comb = R_d_inv @ R_a, R_d_inv @ p_a + p_d_inv
            Cj = np.array([frame_transformation(R_comb, ci, p_comb) for ci in c], float)
            C_expected_frames.append(Cj)
        return C_expected_frames
    
    #eg case
    d = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
    a = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0]])
    c = np.array([[0, 0, 1], [0, 0, -1]])
    
    # If D and A are at same positions as d and a
    D_frames = [d.copy()]
    A_frames = [a.copy()]
    
    C_expected = compute_expected_C(d, a, c, D_frames, A_frames)
    
    assert len(C_expected) == 1
    assert C_expected[0].shape == (2, 3)
    # The expected C should be approximately equal to c
    np.testing.assert_array_almost_equal(C_expected[0], c, decimal=5)

# INTEGRATION TESTS

def test_full_pipeline_no_distortion():
    """Test the full pipeline with synthetic data and no distortion"""
    np.random.seed(42)
    
    # Generate synthetic calibration data
    d = np.random.randn(3, 3)
    a = np.random.randn(4, 3)
    c = np.random.randn(5, 3)
    
    # Generate synthetic measurements
    D_frames = [d + np.random.randn(3, 3) * 0.01 for _ in range(5)]
    A_frames = [a + np.random.randn(4, 3) * 0.01 for _ in range(5)]
    
    # This test verifies that the components can work together
    assert len(D_frames) == 5
    assert len(A_frames) == 5


if __name__ == "__main__":
    #run unit tests
    pytest.main([__file__, "-v", "--tb=short"])