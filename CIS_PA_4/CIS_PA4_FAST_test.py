import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import os

# ======================== CONFIGURATION ==========================
# Path to the dataset folder (contains all PA1 debug and unknown files)
extract_path = r"C:\Users\aarus\Documents\CIS_PA_4\DATA\\"
datasets = os.listdir(extract_path)

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

def parseSampleReadings(path, Na, Nb):
    """Parse sample readings for measurements of A, B, and D markers in tracker coordinates"""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    Nsum, Ns = map(int, lines[0].split(",")[:2]) #Read Number of samples and total number of marker frames
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
    return A_frames, B_frames, D_frames, Nd, Ns

# ==================== OCTTREE AND SEARCH  =====================

class OctreeNode:
    def __init__(self, center, half_size, tri_indices):
        self.center = center
        self.half_size = half_size
        self.tri_indices = tri_indices
        self.children = [None] * 8

def distance(a, center, half_size):
    dx = max(0.0, abs(a[0] - center[0]) - half_size)
    dy = max(0.0, abs(a[1]- center[1]) - half_size)
    dz = max(0.0, abs(a[2]- center[2]) - half_size)      

    return dx**2 + dy**2 + dz**2

def octree(tris, tri_indices, center, half_size, depth = 0):
    tri_indices = list(tri_indices)
    node = OctreeNode(center, half_size, tri_indices)

    if depth >= 10 or len(tri_indices) <= 20:
        return node
    
    indices_min = np.array([np.min(tris[i], axis=0) for i in tri_indices])
    indices_max = np.array([np.max(tris[i], axis=0) for i in tri_indices])
    child_half = half_size/2.0
    offsets = np.array([
        [-1,-1,-1], [ 1,-1,-1],
        [-1, 1,-1], [ 1, 1,-1],
        [-1,-1, 1], [ 1,-1, 1],
        [-1, 1, 1], [ 1, 1, 1]
    ], dtype=float)

    for c in range(8):
        child_center = center + offsets[c]*child_half
        c_min = child_center - child_half
        c_max = child_center + child_half

        child_tris = []

        for idx, tri_idx in enumerate(tri_indices):
            tmin = indices_min[idx]
            tmax = indices_max[idx]

            if not (tmax[0] < c_min[0] or tmin[0] > c_max[0] or
                    tmax[1] < c_min[1] or tmin[1] > c_max[1] or
                    tmax[2] < c_min[2] or tmin[2] > c_max[2]):
                child_tris.append(tri_idx)
        
        if child_tris:
            node.children[c] = octree(tris,child_tris,child_center,child_half,depth + 1)

    return node

def octree_search(node, a, tris, d2, c):
    if node is None:
        return d2, c
    
    if distance(a, node.center, node.half_size) > d2:
        return d2, c
    
    if all(child is None for child in node.children):
        for idx in node.tri_indices:
            p,q,r = tris[idx]
            c_new, d2_new= closest_point_on_triangle(a,p,q,r)
            if d2_new < d2:
                d2 = d2_new
                c = c_new
            
        return d2, c
    
    for child in node.children:
        d2, c = octree_search(child, a, tris, d2, c)
    return d2, c



# ==================== CLOSEST POINT SEARCH  =====================

def compute_bounds(V, Indices):
    """
    Compute lower and upper bounds for each triangle in mesh
    """
    V = np.asarray(V, dtype=float)
    Indices = np.asarray(Indices, dtype=int)
    tris = V[Indices] # Make an array with coordinates for vertices of each triangle in each row (shape (nt, 3, 3))
    lower = np.min(tris, axis=1) #array with minimum bound for each triangle
    upper = np.max(tris, axis=1) # array with maximum bound for each triangle
    return lower, upper, tris

def project_on_segment(a, p, q):
    #Find line segments
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
    """
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)

    #Set up system to solve for oordinates λ, μ
    M = np.column_stack((q - p, r - p)) 
    rhs = a - p

    # Least-squares solve for λ and μ
    lam_mu, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    lam, mu = lam_mu
    nu = 1 - lam - mu
 
    # Check if point projects inside triangle 
    if lam >= 0.0 and mu >= 0.0 and (lam + mu) <= 1.0:
        c = lam * q + mu * r + nu * p #project inside triangle
        return c, np.sum((a - c) ** 2)

    # If not inside, project on edge, checking region by region to determine projection side
    if (lam < 0):
        c, d = project_on_segment(a, r, p)
    elif (mu < 0):
        c, d = project_on_segment(a, p, q)
    else: 
        c, d = project_on_segment(a, q, r)
    return c, float(d)

def closest_point_octree(a, octree_root, tris):
    d2 = np.inf
    c = None
    d2, c = octree_search(octree_root, a, tris, d2, c)
    return c, d2

# ========================= ICP Algorithms ============================

def compute_d(A, B, A_tip, A_frames, B_frames, Ns):
    
    d = [] 
    A_tip = np.asarray(A_tip).reshape(3,)

    #For each sample find point A_tip in Body B coordinates
    for k in range(Ns): 
        A_meas = np.asarray(A_frames[k])
        B_meas = np.asarray(B_frames[k])
        # 1. Find transformation between measured LED trackers and trackers in body coordinates 
        R_ak, p_ak = point2point_3Dregistration(A, A_meas)
        R_bk, p_bk = point2point_3Dregistration(B, B_meas)
        # 2. Apply transformation to A_tip to find A_tip in tracker coordinates 
        a_tracker = frame_transformation(R_ak, A_tip, p_ak)

        R_bk_inv = R_bk.T
        p_bk_inv = -R_bk.T@p_bk
        # 3. Apply inverse transformation to A_tracker to find A_tracker with respect to Body B coordinates (d_k)
        d_k = frame_transformation(R_bk_inv , a_tracker, p_bk_inv)
        d.append(d_k)
    
    return np.array(d)

def icp_algorithm(d, tris, octree_root):
    Q = d
    Ns = d.shape[0]

    n = 0
    R_reg = np.eye(3)
    p_reg = np.zeros(3)

    eta_n = np.inf
    sigma_delta_thresh = 0.000001

    prev_err = None
    prevsigma = 0

    min_val = int(0.3 * Ns)

    gamma_count = 0

    while True:
        A_adjusted = []
        B_adjusted = []
        C =[]
        D = []

        for k in range(Ns):
            q_k = Q[k]
            s_k = frame_transformation(R_reg, q_k, p_reg)

            c_k, d2_k = closest_point_octree(s_k, octree_root, tris)

            #line may not be needed
            if c_k is None:
                continue
            
            d_k = np.sqrt(d2_k)
            C.append(c_k)
            D.append(d_k)

            if (d_k < eta_n):
                A_adjusted.append(q_k)
                B_adjusted.append(c_k)

        #may not need
        
        A_adjusted = np.asarray(A_adjusted) 
        B_adjusted = np.asarray(B_adjusted)           

        if A_adjusted.shape[0] < 3:
            break
        
        R_adjust, p_adjust = point2point_3Dregistration(A_adjusted, B_adjusted)

        s_adjust = (R_adjust@A_adjusted.T).T + p_adjust
        E = B_adjusted - s_adjust
        E_norm = np.linalg.norm(E, axis=1)

        mean_E = E_norm.mean()
        max_err = E_norm.max()
        sigma_n = np.sqrt(np.mean(E_norm**2))
        print(f"Iteration {n+1}")
        print(f"Sigma_n: {sigma_n}")


# do we need to rebound to sigma n 

        if len(A_adjusted) < min_val:
            eta_n *=1.3    
        else:
            eta_n = 3.0 * mean_E
        
        sigma_delt = abs(sigma_n - prevsigma)

        if sigma_delt < sigma_delta_thresh and prev_err is not None:
            gamma_ratio = mean_E / prev_err
            print(f"Sigma delta less than threshold!")
            if (gamma_count < 3):
                if gamma_ratio >= 0.95 and gamma_ratio <= 1:
                    R_reg, p_reg = R_adjust, p_adjust
                    print(f"Gamma Ratio Within Threshold!: {gamma_ratio}")
                    gamma_count += 1
                else:
                    gamma_count = 0
            else:
                break

        
        R_reg, p_reg = R_adjust, p_adjust
        prev_err = mean_E
        prevsigma = sigma_n

        n += 1
        if n == 230:
            print("Maximum iterations exceed, assuming convergence")
            break

    S_list = []
    C_list = []
    for k in range(Ns):
        q_k = Q[k]
        s_adjust_k = R_reg @ q_k + p_reg
        S_list.append(s_adjust_k)
        c_adjust_k, _ = closest_point_octree(s_adjust_k, octree_root, tris)

        C_list.append(c_adjust_k)

    return R_reg, p_reg, np.array(S_list), np.array(C_list)







# =========================OUTPUT FILES ============================

def write_output_files(filename, d, c, Ns):
    """Write output file with tip positions d and closest point on mesh c, as well as distance between the two"""
    #Write Output File 1
    with open(filename, "w") as f:
        f.write(f"{Ns} {filename}\n")
        for s in range(Ns):
            magdiff = np.sum((d[s] - c[s])**2)
            diff = magdiff**(0.5)
            f.write(f"{d[s][0]:.2f}\t{d[s][1]:.2f}\t{d[s][2]:.2f}\t\t{c[s][0]:.2f}\t{c[s][1]:.2f}\t{c[s][2]:.2f}\t{diff:.3f}\n")
        


# ======================= DATASET PIPELINE ========================
def process_dataset(data_prefix):
    """Run full matching algorithm for a single dataset prefix."""
    print(f"Processing {data_prefix}...")
    try:
    # --- 1. Parse input files ---
        Na, A, A_tip = parseBodyA(os.path.join(extract_path,"Problem4-BodyA.txt"))
        Nb, B, B_tip = parseBodyB(os.path.join(extract_path,"Problem4-BodyB.txt"))
        Nv , V, Nt, Indices = parseMesh(os.path.join(extract_path,"Problem4MeshFile.sur"))
        A_frames, B_frames, D_frames, Nd, Ns = parseSampleReadings(os.path.join(extract_path, f"{data_prefix}-SampleReadingsTest.txt"), Na, Nb)
    except Exception as e:
        print(f"  Error processing {data_prefix}: {e}")
    print("Read in Files")
    # --- 2. Find bounds for each triangle ---
    lower, upper, tris = compute_bounds(V, Indices)
    tri_indices = list(range(tris.shape[0]))


    global_lower = np.min(lower, axis = 0)
    global_upper = np.max(upper, axis = 0)
    center = 0.5 * (global_lower + global_upper)
    half_size = 0.5* np.max(global_upper - global_lower)

    octree_root = octree(tris, tri_indices, center, half_size)


    d = compute_d(A, B, A_tip, A_frames, B_frames, Ns)
    print("computed D")
    print("Starting ICP:")
    R_reg, p_reg, s, c = icp_algorithm(d,tris, octree_root)
    print("Finished ICP")
    out_name = f"{data_prefix}-myOutput.txt"
    write_output_files(out_name, s , c, Ns)

def main():
    """Main loop — process all datasets in directory."""
    # Ask the user for the dataset path
    '''
    extract_path = input("Please enter the path to the dataset folder: ").strip()  # Get user input
    
    # ensure the path is valid
    while not os.path.isdir(extract_path):
        print("The path you entered is invalid. Please try again.")
        extract_path = input("Please enter the path to the dataset folder: ").strip()

    datasets = os.listdir(extract_path) #extract datasets  
    '''
    '''

    
    #Run pipeline for each dataset in folder
    for fname in datasets:
        if fname.endswith("-SampleReadingsTest.txt"):
            data_prefix = fname[:-len("-SampleReadingsTest.txt")]
            process_dataset(data_prefix)
        else:
            print(f"Skipping {fname}")
    '''
    process_dataset("PA4-K-Unknown")


if __name__ == "__main__":
    main()