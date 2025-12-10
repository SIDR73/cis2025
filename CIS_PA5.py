import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import os

# ===================== 3D GEOMETRY UTILITIES =====================
def pos_3D(x, y, z):
    """Return a 3D vector as NumPy array."""
    return np.array([x, y, z])

# --- Rotation matrices around X, Y, Z axes respectively ---
def Rot_X(theta):
    """Rotation matrix about X-axis by angle theta (radians)."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def Rot_Y(theta):
    """Rotation matrix about Y-axis by angle theta (radians)."""
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rot_Z(theta):
    """Rotation matrix about Z-axis by angle theta (radians)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

def rotate(R, P):
    """Apply rotation R to point/vector P."""
    P = np.asarray(P).reshape(3,)
    return R @ P

def translate(P, d):
    """Apply translation d to point P."""
    P, d = np.asarray(P).reshape(3,), np.asarray(d).reshape(3,)
    return P + d

def frame_transformation(R, Pin, d):
    """Perform full rigid-body transform: output = R * Pin + d."""
    Pin, d = np.asarray(Pin).reshape(3,), np.asarray(d).reshape(3,)
    R = np.asarray(R).reshape(3, 3)
    return R @ Pin + d


# ==================== 3D POINT REGISTRATION ======================
def point2point_3Dregistration(A, B):
    """
    Compute rotation (R) and translation (p) aligning A -> B using SVD.
    Solves min_{R,p} ||B - (R A + p)||^2.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    if A.shape != B.shape or A.shape[1] != 3:
        raise ValueError("A and B must have same shape (N,3)")

    # Compute centroids of both sets
    cA, cB = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - cA, B - cB

    # Covariance matrix and SVD decomposition
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # Ensure right-handed rotation
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Translation vector
    p = cB - R @ cA
    return R, p


# ======================== FILE PARSERS ===========================
def parseBodyA(path):
    """Read BodyA file for A markers coordinates and A tip coordinates."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    Na= int(lines[0].split()[0]) # read values from header
    idx = 1
    A = [[float(x) for x in lines[idx+i].split()] for i in range (Na)] #parse all LED marker coordinates
    A_tip = [[float(x) for x in lines[idx + Na].split()]] #parse tip coordinate
    return Na, np.array(A), np.array(A_tip)

def parseBodyB(path):
    """Read BodyB file for B markers coordinates and B tip coordinates."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    Nb= int(lines[0].split()[0]) #read data amount from header
    idx = 1
    B = [[float(x) for x in lines[idx+i].split()] for i in range (Nb)] #parse LED marker coordinates
    B_tip = [[float(x) for x in lines[idx + Nb].split()]] #parse tip coordinate
    return Nb, np.array(B), np.array(B_tip)

def parseMesh(path):
    """Parse Mesh file for vertices and indices"""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    Nv= int(lines[0].split()[0]) #Read number of vertices from header
    V = [[float(x) for x in lines[1 + i].split()] for i in range (Nv)] #Parse coordinates for Vertices
    Nt = int(lines[1 + Nv].split()[0]) #Read number of triangles
    Indices = [[int(x) for x in lines[Nv + i + 2].split()[:3]] for i in range (Nt)] #Pares vertex indices for each triangle
    return Nv, np.array(V), Nt, np.array(Indices)

def parseModes(path):
    """Parse modes file for atlas deformation modes."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    # Parse header: "Problem5Modes.txt Nvertices=1568 Nmodes=6"
    header_parts = lines[0].split()
    Nvertices = None
    Nmodes = None
    for part in header_parts:
        if part.startswith("Nvertices="):
            Nvertices = int(part.split("=")[1])
        elif part.startswith("Nmodes="):
            Nmodes = int(part.split("=")[1])
    
    if Nvertices is None or Nmodes is None:
        raise ValueError("Could not parse Nvertices or Nmodes from modes file header")
    
    # Parse Mode 0 (average)
    idx = 1
    if not lines[idx].startswith("Mode 0"):
        raise ValueError("Expected Mode 0 header")
    idx += 1
    
    mode0 = []
    for i in range(Nvertices):
        coords = [float(x.strip()) for x in lines[idx + i].split(",")]
        mode0.append(coords)
    idx += Nvertices
    
    # Parse remaining modes
    modes = []
    for m in range(1, Nmodes + 1):
        if idx >= len(lines) or not lines[idx].startswith(f"Mode {m}"):
            raise ValueError(f"Expected Mode {m} header at line {idx}")
        idx += 1
        
        mode_displacements = []
        for i in range(Nvertices):
            if idx + i >= len(lines):
                raise ValueError(f"Not enough lines for mode {m}")
            coords = [float(x.strip()) for x in lines[idx + i].split(",")]
            mode_displacements.append(coords)
        modes.append(mode_displacements)
        idx += Nvertices
    
    return Nvertices, Nmodes, np.array(mode0), [np.array(m) for m in modes]

def parseSampleReadings(path, Na, Nb):
    """Parse sample readings for measurements of A, B, and D markers in tracker coordinates.
    Returns Nmodes from header as well."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    # Parse header: "16, 150, PA5-A-Debug-SampleReadingsTest.txt 6"
    header_parts = lines[0].split(",")
    Nsum = int(header_parts[0].strip())
    Ns = int(header_parts[1].strip())
    
    # Extract Nmodes from the end of the header line
    last_part = header_parts[-1].strip().split()
    Nmodes = int(last_part[-1]) if last_part else None
    
    if Nmodes is None:
        raise ValueError("Could not parse Nmodes from sample readings header")
    
    Nd = Nsum - Na - Nb #find number of dummy frames
    idx = 1
    A_frames = []
    B_frames = []
    D_frames = []
    for _ in range(Ns): #parse measured coordinates for LED markers on A, B, and Dummy coordinates
        A = np.array([[float(x) for x in lines[idx + i].split(",")] for i in range(Na)], float); idx += Na
        B = np.array([[float(x) for x in lines[idx + i].split(",")] for i in range(Nb)], float); idx += Nb
        D = np.array([[float(x) for x in lines[idx + i].split(",")] for i in range(Nd)], float); idx += Nd
        A_frames.append(A), B_frames.append(B), D_frames.append(D)
    return A_frames, B_frames, D_frames, Nd, Ns, Nmodes

# ==================== CLOSEST POINT SEARCH  =====================

def project_on_segment(a, p, q):
    """Find closest point on line segment pq to point a."""
    ap = a - p 
    qp = q - p
    denom = np.dot(qp, qp) #check magnitude of segment pq
    if denom == 0.0: # if magnitude is 0, set closest point to a vertex
        c = p.copy()
    else: 
        lambd = np.dot(ap, qp) / np.dot(qp, qp) #calculate lambda
        lam_seg = np.maximum(0, np.minimum(lambd, 1))  #Bind lambda to find lambda for segment
        c = p + lam_seg * qp # find projected point 
    return c, np.sum((a - c) ** 2)

def closest_point_on_triangle(a, p, q, r):
    """
    Barycentric-based closest point on triangle pqr to point a.
    Returns closest point, distance squared, and barycentric coordinates (zeta, xi, psi).
    """
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)

    #Set up system to solve for coordinates λ, μ
    M = np.column_stack((q - p, r - p)) 
    rhs = a - p

    # Least-squares solve for λ and μ
    lam_mu, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    lam, mu = lam_mu
    nu = 1 - lam - mu
 
    # Check if point projects inside triangle 
    if lam >= 0.0 and mu >= 0.0 and (lam + mu) <= 1.0:
        c = lam * q + mu * r + nu * p #project inside triangle
        return c, np.sum((a - c) ** 2), (nu, lam, mu)  # (zeta, xi, psi) = (nu, lam, mu)

    # If not inside, project on edge, checking region by region to determine projection side
    if (lam < 0):
        c, d = project_on_segment(a, r, p)
        # Compute barycentric for edge r-p
        t = np.dot(c - r, p - r) / np.dot(p - r, p - r) if np.dot(p - r, p - r) > 0 else 0
        t = np.clip(t, 0, 1)
        bary = (t, 0, 1 - t)  # (zeta, xi, psi)
    elif (mu < 0):
        c, d = project_on_segment(a, p, q)
        # Compute barycentric for edge p-q
        t = np.dot(c - p, q - p) / np.dot(q - p, q - p) if np.dot(q - p, q - p) > 0 else 0
        t = np.clip(t, 0, 1)
        bary = (1 - t, t, 0)  # (zeta, xi, psi)
    else: 
        c, d = project_on_segment(a, q, r)
        # Compute barycentric for edge q-r
        t = np.dot(c - q, r - q) / np.dot(r - q, r - q) if np.dot(r - q, r - q) > 0 else 0
        t = np.clip(t, 0, 1)
        bary = (0, 1 - t, t)  # (zeta, xi, psi)
    return c, float(d), bary


# ==================== KD-TREE IMPLEMENTATION =====================

class KDTreeNode:
    """Node for KD-tree structure."""
    def __init__(self, index):
        self.index = index  # index of vertex in vertices array
        self.lessThan = None  # child node with smaller value on current axis
        self.greaterThan = None  # child node with larger value on current axis


def make_tree_point(vertices):
    """
    Create a KD-tree from vertices: Nx3 array of vertex coord., output root note
    """
    # add indices as 4th column
    vertices_with_indices = np.column_stack([vertices, np.arange(len(vertices))])
    return build_tree(vertices_with_indices, 1)


def build_tree(vertices_with_indices, axis):
    """
    Recursively build KD-tree.
    """
    if len(vertices_with_indices) == 0:
        return None
    
    # Sort by current axis
    sorted_indices = np.argsort(vertices_with_indices[:, axis - 1])
    sorted_vertices = vertices_with_indices[sorted_indices]
    
    # find middle point
    middle = len(sorted_vertices) // 2
    
    # create node
    root = KDTreeNode(int(sorted_vertices[middle, 3]))
    
    # Recursively build left and right subtrees 
    next_axis = (axis % 3) + 1
    if middle > 0:
        root.lessThan = build_tree(sorted_vertices[:middle], next_axis)
    if middle + 1 < len(sorted_vertices):
        root.greaterThan = build_tree(sorted_vertices[middle + 1:], next_axis)
    
    return root


def test_triangles(vert_index, vertices, indices, point):
    """
    Find closest point on all triangles that contain vertex vert_index.
    Returns closest point, distance squared, triangle index, and barycentric coordinates.
    """
    # Find all triangles that contain this vertex
    triangle_indices = np.where((indices == vert_index).any(axis=1))[0]
    
    #initialize values
    closest_point = None
    min_dist = np.inf
    best_tri_idx = None
    best_bary = None
    
    #For triangles that contain vertex
    for tri_idx in triangle_indices:
        curr_indices = indices[tri_idx]
        #Extract vertices
        p1 = vertices[curr_indices[0]]
        p2 = vertices[curr_indices[1]]
        p3 = vertices[curr_indices[2]]
        
        #Find closest point and distance
        closest_temp, dist_temp, bary_temp = closest_point_on_triangle(point, p1, p2, p3)
        
        #If less then replace best distance
        if dist_temp < min_dist:
            min_dist = dist_temp
            closest_point = closest_temp
            best_tri_idx = tri_idx
            best_bary = bary_temp
    
    if closest_point is None:
        closest_point = vertices[vert_index]
        min_dist = np.sum((point - closest_point) ** 2)
        best_tri_idx = None
        best_bary = None
    
    return closest_point, min_dist, best_tri_idx, best_bary


def search_tree_point(root, vertices, indices, point, axis):
    """
    Search KD-tree to find closest point on mesh to given point.
    Outputs: (1) closest_point on mesh, (2) minimum distance squared, 
    (3) triangle index, (4) barycentric coordinates
    """
    if root is None:
        return None, np.inf, None, None
    
    vertex = vertices[root.index]
    
    # determine which child to search first
    if point[axis - 1] < vertex[axis - 1]:
        first = root.lessThan
        second = root.greaterThan
    else:
        first = root.greaterThan
        second = root.lessThan
    
    #search for the first child
    closest_point, min_dist, best_tri_idx, best_bary = search_tree_point(first, vertices, indices, point, (axis % 3) + 1)
    
    #check triangles
    closest_temp, dist_temp, tri_idx_temp, bary_temp = test_triangles(root.index, vertices, indices, point)
    if dist_temp < min_dist:
        min_dist = dist_temp
        closest_point = closest_temp
        best_tri_idx = tri_idx_temp
        best_bary = bary_temp
    
    #check if we need to search the other side
    if abs(vertex[axis - 1] - point[axis - 1]) < np.sqrt(min_dist):
        closest_temp, dist_temp, tri_idx_temp, bary_temp = search_tree_point(second, vertices, indices, point, (axis % 3) + 1)
        if dist_temp < min_dist:
            min_dist = dist_temp
            closest_point = closest_temp
            best_tri_idx = tri_idx_temp
            best_bary = bary_temp
    
    return closest_point, min_dist, best_tri_idx, best_bary

# ========================= DEFORMABLE REGISTRATION ============================

def compute_d(A, B, A_tip, A_frames, B_frames, Ns):
    """
    Calculate coordinates of A_tip with respect to Body B coordinates for each sample frame 
    """
    
    d = [] 
    A_tip = np.asarray(A_tip).reshape(3,)

    #For each sample find point A_tip in Body B coordinates
    for k in range(Ns): 
        A_meas = np.asarray(A_frames[k])
        B_meas = np.asarray(B_frames[k])
        # Find transformation between measured LED trackers and trackers in body coordinates 
        R_ak, p_ak = point2point_3Dregistration(A, A_meas)
        R_bk, p_bk = point2point_3Dregistration(B, B_meas)
        # Apply transformation to A_tip to find A_tip in tracker coordinates 
        a_tracker = frame_transformation(R_ak, A_tip, p_ak)

        R_bk_inv = R_bk.T
        p_bk_inv = -R_bk.T@p_bk
        # Apply inverse transformation to A_tracker to find A_tracker with respect to Body B coordinates (d_k)
        d_k = frame_transformation(R_bk_inv , a_tracker, p_bk_inv)
        d.append(d_k)
    
    return np.array(d)

def update_mesh_vertices(mode0, modes, lambda_vals, Nmodes):
    """
    Update mesh vertices using mode weights: V = mode0 + sum(lambda_m * modes[m])
    """
    V = mode0.copy()
    for m in range(Nmodes):
        if m < len(modes):
            V += lambda_vals[m] * modes[m]
    return V

def compute_q_vectors(mode0, modes, tri_indices, bary_coords, Nmodes):
    """
    Compute q_0,k and q_m,k vectors for a point on a triangle using barycentric coordinates.
    tri_indices: [s, t, u] - vertex indices of triangle
    bary_coords: (zeta, xi, psi) - barycentric coordinates
    """
    zeta, xi, psi = bary_coords
    s, t, u = tri_indices
    
    # q_0,k = zeta * mode0[s] + xi * mode0[t] + psi * mode0[u]
    q0 = zeta * mode0[s] + xi * mode0[t] + psi * mode0[u]
    
    # q_m,k = zeta * modes[m-1][s] + xi * modes[m-1][t] + psi * modes[m-1][u]
    qm_list = []
    for m in range(1, Nmodes + 1):
        if m - 1 < len(modes):
            qm = zeta * modes[m-1][s] + xi * modes[m-1][t] + psi * modes[m-1][u]
            qm_list.append(qm)
        else:
            qm_list.append(np.zeros(3))
    
    return q0, np.array(qm_list)

def initial_rigid_registration(d, root, V, Indices):
    """
    Perform initial rigid registration using PA4 ICP method.
    Returns R_reg, p_reg, s_k values.
    """
    Q = d
    Ns = d.shape[0]

    #Initial guess for R_reg and p_reg
    n = 0
    R_reg = np.eye(3)
    p_reg = np.zeros(3)

    #Define sigma_delta bound and initial eta_n
    eta_n = np.inf
    sigma_delta_thresh = 0.000001

    prev_err = None
    prevsigma = 0

    min_val = int(0.3 * Ns)

    gamma_count = 0

    while True:
        A_adjusted = []
        B_adjusted = []

        for k in range(Ns):
            q_k = Q[k]
            #Compute initial sample point 
            s_k = frame_transformation(R_reg, q_k, p_reg)

            #Find closest point on mesh to sample point 
            c_k, d2_k, _, _ = search_tree_point(root, V, Indices, s_k, 1)

            if c_k is None:
                continue
            
            d_k = np.sqrt(d2_k)

            #If distance is less than current error, add to A_adjusted 
            if (d_k < eta_n):
                A_adjusted.append(q_k)
                B_adjusted.append(c_k)

        
        A_adjusted = np.asarray(A_adjusted) 
        B_adjusted = np.asarray(B_adjusted)           

        #Compute new guess for R_reg and p_reg
        R_adjust, p_adjust = point2point_3Dregistration(A_adjusted, B_adjusted)

        #Compute new sample point 
        s_adjust = (R_adjust@A_adjusted.T).T + p_adjust

        #Compute residuals and calculate mean error and sigma 
        E = B_adjusted - s_adjust
        E_norm = np.linalg.norm(E, axis=1)

        mean_E = E_norm.mean()
        sigma_n = np.sqrt(np.mean(E_norm**2))

        #If eliminating too many values, increase bound
        if len(A_adjusted) < min_val:
            eta_n *=1.3    
        else:
            #Otherwise, increase as normal 
            eta_n = 3.0 * mean_E
        
        #Calculate change between sigma values 
        sigma_delt = abs(sigma_n - prevsigma)
        
        #If change is within threshold 
        if sigma_delt < sigma_delta_thresh and prev_err is not None:
            #Check Gamma Ratio 
            gamma_ratio = mean_E / prev_err
           
           #If gamma ratio remains within threshold for 3 iterations, assume convergence
            if (gamma_count < 3):
                if gamma_ratio >= 0.95 and gamma_ratio <= 1:
                    R_reg, p_reg = R_adjust, p_adjust
                    gamma_count += 1
                else:
                    gamma_count = 0
            else:
                break
        else:
            gamma_count = 0

        #Set final value to R_reg and p_reg 
        R_reg, p_reg = R_adjust, p_adjust

        prev_err = mean_E
        prevsigma = sigma_n

        n += 1

        #If Iterations exceed maximum, break 
        if n == 250:
            break

    # Compute final s_k values
    S_list = []
    for k in range(Ns):
        q_k = Q[k]
        s_k = frame_transformation(R_reg, q_k, p_reg)
        S_list.append(s_k)

    return R_reg, p_reg, np.array(S_list)

def deformable_registration(d, mode0, modes, Indices, Nmodes, max_iter=50):
    """
    Perform deformable registration using iterative mode matching and rigid transformation.
    """
    Ns = d.shape[0]
    
    # Initialize mesh with mode0
    V = mode0.copy()
    root = make_tree_point(V)
    
    # Initial rigid registration
    print("Performing initial rigid registration...")
    R_reg, p_reg, s_k = initial_rigid_registration(d, root, V, Indices)
    
    # Initialize mode weights
    lambda_vals = np.zeros(Nmodes)
    
    print("Starting deformable registration iterations...")
    
    for outer_iter in range(max_iter):
        # Step 1: Find closest points on current deformed mesh
        c_k_list = []
        q0_k_list = []
        qm_k_list = []
        valid_samples = []
        
        for k in range(Ns):
            s_k_current = frame_transformation(R_reg, d[k], p_reg)
            c_k, d2_k, tri_idx, bary_coords = search_tree_point(root, V, Indices, s_k_current, 1)
            
            if c_k is None or tri_idx is None:
                continue
            
            # Get triangle vertex indices
            tri_verts = Indices[tri_idx]
            
            # Compute q vectors
            q0_k, qm_k = compute_q_vectors(mode0, modes, tri_verts, bary_coords, Nmodes)
            
            c_k_list.append(c_k)
            q0_k_list.append(q0_k)
            qm_k_list.append(qm_k)
            valid_samples.append(k)
        
        if len(valid_samples) == 0:
            print("Warning: No valid samples found!")
            break
        
        c_k_array = np.array(c_k_list)
        q0_k_array = np.array(q0_k_list)
        qm_k_array = np.array(qm_k_list)  # Shape: (N_valid, Nmodes, 3)
        s_k_valid = np.array([frame_transformation(R_reg, d[k], p_reg) for k in valid_samples])
        
        # Step 2: Solve least squares for lambda^(t+1)
        # Equation: s_k ≈ q_0,k + sum(lambda_m * q_m,k)
        # Rearrange: sum(lambda_m * q_m,k) ≈ s_k - q_0,k
        
        # Build system: A * lambda = b
        # A has shape (3*N_valid, Nmodes), b has shape (3*N_valid,)
        N_valid = len(valid_samples)
        A = np.zeros((3 * N_valid, Nmodes))
        b = np.zeros(3 * N_valid)
        
        for i, k in enumerate(valid_samples):
            s_k_vec = s_k_valid[i]
            q0_k_vec = q0_k_array[i]
            residual = s_k_vec - q0_k_vec
            
            # Fill A matrix: each mode contributes 3 rows
            for m in range(Nmodes):
                qm_k_vec = qm_k_array[i, m]
                A[3*i:3*(i+1), m] = qm_k_vec
            
            # Fill b vector
            b[3*i:3*(i+1)] = residual
        
        # Solve least squares
        lambda_old = lambda_vals.copy()
        lambda_new, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
        lambda_vals = lambda_new
        
        # Step 3: Update mesh vertices
        V = update_mesh_vertices(mode0, modes, lambda_vals, Nmodes)
        
        # Rebuild KD-tree
        root = make_tree_point(V)
        
        # Step 4: Re-estimate rigid transformation
        # Find new closest points with updated mesh
        A_points = []
        B_points = []
        
        for k in range(Ns):
            s_k_current = frame_transformation(R_reg, d[k], p_reg)
            c_k_new, _, _, _ = search_tree_point(root, V, Indices, s_k_current, 1)
            
            if c_k_new is not None:
                A_points.append(d[k])
                B_points.append(c_k_new)
        
        if len(A_points) < 3:
            print("Warning: Not enough points for registration!")
            break
        
        A_points = np.array(A_points)
        B_points = np.array(B_points)
        
        # Re-estimate rigid transformation
        R_new, p_new = point2point_3Dregistration(A_points, B_points)
        
        # Check convergence
        R_change = np.linalg.norm(R_new - R_reg)
        p_change = np.linalg.norm(p_new - p_reg)
        lambda_change = np.linalg.norm(lambda_new - lambda_old) if outer_iter > 0 else np.inf
        
        # Update transformations
        R_reg = R_new
        p_reg = p_new
        
        # Compute error
        errors = []
        for k in range(Ns):
            s_k_current = frame_transformation(R_reg, d[k], p_reg)
            c_k_current, d2_k, _, _ = search_tree_point(root, V, Indices, s_k_current, 1)
            if c_k_current is not None:
                errors.append(np.sqrt(d2_k))
        
        mean_error = np.mean(errors) if errors else np.inf
        
        print(f"Outer iteration {outer_iter + 1}: mean_error={mean_error:.6f}, "
              f"R_change={R_change:.6f}, p_change={p_change:.6f}, lambda_change={lambda_change:.6f}")
        
        # Check convergence
        if R_change < 1e-6 and p_change < 1e-6 and lambda_change < 1e-6:
            print("Converged!")
            break
    
    # Final computation of s_k and c_k
    S_list = []
    C_list = []
    
    for k in range(Ns):
        s_k_final = frame_transformation(R_reg, d[k], p_reg)
        c_k_final, _, _, _ = search_tree_point(root, V, Indices, s_k_final, 1)
        
        if c_k_final is None:
            # Fallback: use vertex
            c_k_final = V[0]
        
        S_list.append(s_k_final)
        C_list.append(c_k_final)
    
    return lambda_vals, R_reg, p_reg, np.array(S_list), np.array(C_list)

# =========================OUTPUT FILES ============================

def write_output_files(filename, lambda_vals, s_k, c_k, Ns, Nmodes):
    """Write output file with mode weights, tip positions s_k and closest point on mesh c_k, as well as distance between the two"""
    with open(filename, "w") as f:
        # Header: Nsamps filename Nmodes
        f.write(f"{Ns} {filename} {Nmodes}\n")
        
        # Mode weights line - format with spaces (matches: "  112.6664   50.3573   57.0491   46.8895  -58.5305  -22.1712")
        lambda_str = " ".join([f"{lam:10.4f}" for lam in lambda_vals])
        f.write(f"{lambda_str}\n")
        
        # Sample points: s_x s_y s_z c_x c_y c_z ||s_k - c_k||
        # Format matches: "   -3.85    -0.55   -36.69        -3.85    -0.55   -36.69     0.000"
        for s in range(Ns):
            diff_vec = s_k[s] - c_k[s]
            diff_mag = np.linalg.norm(diff_vec)
            f.write(f"{s_k[s][0]:8.2f} {s_k[s][1]:8.2f} {s_k[s][2]:8.2f}        "
                   f"{c_k[s][0]:8.2f} {c_k[s][1]:8.2f} {c_k[s][2]:8.2f}     {diff_mag:.3f}\n")


# ======================= DATASET PIPELINE ========================
def process_dataset(data_prefix, extract_path):
    """Run full deformable registration algorithm for a single dataset prefix."""
    print(f"Processing {data_prefix}...")
    try:
        Na, A, A_tip = parseBodyA(os.path.join(extract_path, "Problem5-BodyA.txt"))
        Nb, B, B_tip = parseBodyB(os.path.join(extract_path, "Problem5-BodyB.txt"))
        Nv, V, Nt, Indices = parseMesh(os.path.join(extract_path, "Problem5MeshFile.sur"))
        Nvertices, Nmodes_total, mode0, modes = parseModes(os.path.join(extract_path, "Problem5Modes.txt"))
        A_frames, B_frames, D_frames, Nd, Ns, Nmodes = parseSampleReadings(
            os.path.join(extract_path, f"{data_prefix}-SampleReadingsTest.txt"), Na, Nb)
        
        # Verify mesh vertices match mode0
        if not np.allclose(V, mode0, atol=1e-4):
            print("Warning: Mesh vertices do not match Mode 0!")
            print(f"Max difference: {np.max(np.abs(V - mode0))}")
        
        # Use only the requested number of modes
        modes_to_use = modes[:Nmodes]
        
    except Exception as e:
        print(f"  Error processing {data_prefix}: {e}")
        import traceback
        traceback.print_exc()
        return

    #Calculate d_k for each sample (pointer tip in B coordinate frame)
    A_tip = np.asarray(A_tip).reshape(3,)
    ds = compute_d(A, B, A_tip, A_frames, B_frames, Ns)
    print("Computed d_k values")
    
    #Perform deformable registration
    lambda_vals, R_reg, p_reg, ss, cs = deformable_registration(ds, mode0, modes_to_use, Indices, Nmodes)
    
    #write outputs to file
    parts = data_prefix.split('-')
    if len(parts) >= 2:
        letter = parts[1].upper()  #get letters
        if "Debug" in data_prefix:
            out_name = f"PA5-{letter}-Debug-Output.txt"
        else:
            out_name = f"PA5-{letter}-Unknown-Output.txt"
    else:
        out_name = f"{data_prefix}-Output.txt"
    
    #Write to output file 
    write_output_files(out_name, lambda_vals, ss, cs, Ns, Nmodes)
    print(f"  Completed {data_prefix}")
    print(f"  Mode weights: {lambda_vals}")


def main():
    #Get path from user 
    extract_path = input("enter dataset path:").strip()
    
    # Default to Data directory if empty
    if not extract_path:
        extract_path = "Data"
    
    if not os.path.exists(extract_path):
        print(f"Error: Path {extract_path} does not exist!")
        return
    
    datasets = os.listdir(extract_path)

    # run pipeline for each dataset - SKIP DEMO FILES
    for fname in datasets:
        if fname.endswith("-SampleReadingsTest.txt"):
            data_prefix = fname[:-len("-SampleReadingsTest.txt")]
            if "Demo" in data_prefix:
                print(f"Skipping {fname} (Demo file)")
                continue
            process_dataset(data_prefix, extract_path)
        else:
            print(f"Skipping {fname}")

if __name__ == "__main__":
    main()

