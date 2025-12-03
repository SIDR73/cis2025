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
        return c, np.sum((a - c) ** 2)

    # If not inside, project on edge, checking region by region to determine projection side
    if (lam < 0):
        c, d = project_on_segment(a, r, p)
    elif (mu < 0):
        c, d = project_on_segment(a, p, q)
    else: 
        c, d = project_on_segment(a, q, r)
    return c, float(d)


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
    
    # Recursively build left and right subtrees *** KEY
    next_axis = (axis % 3) + 1
    if middle > 0:
        root.lessThan = build_tree(sorted_vertices[:middle], next_axis)
    if middle + 1 < len(sorted_vertices):
        root.greaterThan = build_tree(sorted_vertices[middle + 1:], next_axis)
    
    return root


def test_triangles(vert_index, vertices, indices, point):
    """
    Find closest point on all triangles that contain vertex vert_index.
    """
    # Find all triangles that contain this vertex
    triangle_indices = np.where((indices == vert_index).any(axis=1))[0]
    
    closest_point = None
    min_dist = np.inf
    
    for tri_idx in triangle_indices:
        curr_indices = indices[tri_idx]
        p1 = vertices[curr_indices[0]]
        p2 = vertices[curr_indices[1]]
        p3 = vertices[curr_indices[2]]
        
        closest_temp, dist_temp = closest_point_on_triangle(point, p1, p2, p3)
        
        if dist_temp < min_dist:
            min_dist = dist_temp
            closest_point = closest_temp
    
    if closest_point is None:
        closest_point = vertices[vert_index]
        min_dist = np.sum((point - closest_point) ** 2)
    
    return closest_point, min_dist


def search_tree_point(root, vertices, indices, point, axis):
    """
    Search KD-tree to find closest point on mesh to given point.
    Outputs: (1) closest_point on mesh, (2) minimum distance squared
    """
    if root is None:
        return None, np.inf
    
    vertex = vertices[root.index]
    
    # determine which child to search first
    if point[axis - 1] < vertex[axis - 1]:
        first = root.lessThan
        second = root.greaterThan
    else:
        first = root.greaterThan
        second = root.lessThan
    
    #search for the first child
    closest_point, min_dist = search_tree_point(first, vertices, indices, point, (axis % 3) + 1)
    
    #check triangles
    closest_temp, dist_temp = test_triangles(root.index, vertices, indices, point)
    if dist_temp < min_dist:
        min_dist = dist_temp
        closest_point = closest_temp
    
    #check if we need to search the other side
    if abs(vertex[axis - 1] - point[axis - 1]) < np.sqrt(min_dist):
        closest_temp, dist_temp = search_tree_point(second, vertices, indices, point, (axis % 3) + 1)
        if dist_temp < min_dist:
            min_dist = dist_temp
            closest_point = closest_temp
    
    return closest_point, min_dist


# =========================OUTPUT FILES ============================

def write_output_files(filename, s, c, Ns):
    """
    Write output file with transformed tip positions s and closest point on mesh c.
    Format: s_x, s_y, s_z, c_x, c_y, c_z, ||s - c||
    """
    with open(filename, "w") as f:
        f.write(f"{Ns}, {filename}\n")
        for i in range(Ns):
            diff = np.sqrt(np.sum((s[i] - c[i]) ** 2))
            f.write(f"{s[i][0]:.3f}, {s[i][1]:.3f}, {s[i][2]:.3f}, ")
            f.write(f"{c[i][0]:.3f}, {c[i][1]:.3f}, {c[i][2]:.3f}, ")
            f.write(f"{diff:.3f}\n")


# ======================= DATASET PIPELINE ========================
def process_dataset(data_prefix, extract_path):
    """Run full ICP algorithm for a single dataset prefix."""
    print(f"Processing {data_prefix}...")
    try:
        Na, A, A_tip = parseBodyA(os.path.join(extract_path, "Problem4-BodyA.txt"))
        Nb, B, B_tip = parseBodyB(os.path.join(extract_path, "Problem4-BodyB.txt"))
        Nv, V, Nt, Indices = parseMesh(os.path.join(extract_path, "Problem4MeshFile.sur"))
        A_frames, B_frames, D_frames, Nd, Ns = parseSampleReadings(
            os.path.join(extract_path, f"{data_prefix}-SampleReadingsTest.txt"), Na, Nb)
    except Exception as e:
        print(f"  Error processing {data_prefix}: {e}")
        return

    # Build KD-tree for closest point search
    root = make_tree_point(V)
    
    #Calculate d_k for each sample (pointer tip in B coordinate frame)
    ds = []
    A_tip = np.asarray(A_tip).reshape(3,)
    
    for k in range(Ns):
        A_meas = np.asarray(A_frames[k])
        B_meas = np.asarray(B_frames[k])
        
        #transformation btwn measured LED trackers and trackers in body coordinates
        R_ak, p_ak = point2point_3Dregistration(A, A_meas)
        R_bk, p_bk = point2point_3Dregistration(B, B_meas)
        
        #transformation to A_tip to find A_tip in tracker coordinates
        a_tracker = frame_transformation(R_ak, A_tip, p_ak)
        
        #use inverse transformation to find A_tracker with respect to Body B coordinates (d_k)
        R_bk_inv = R_bk.T
        p_bk_inv = -R_bk.T @ p_bk
        d_k = frame_transformation(R_bk_inv, a_tracker, p_bk_inv)
        
        ds.append(d_k)
    
    ds = np.array(ds)
    
    #ICP
    R_reg = np.eye(3)
    p_reg = np.zeros(3)
    
    #iteration parameters to play with
    eta = 100.0
    sigma = 100.0
    epsilon_avg = 100.0
    prev_epsilon_avg = 1.0
    counter = 0
    
    #iterate until convergence
    while sigma > 0.01 or (epsilon_avg / prev_epsilon_avg < 0.95 or epsilon_avg / prev_epsilon_avg > 1.0):
        A_correspondences = []
        B_correspondences = []
        prev_epsilon_avg = epsilon_avg
        
        #for each sample, find closest point on mesh
        cs = []
        for k in range(Ns):
            d_k = ds[k]
            
            s_k = frame_transformation(R_reg, d_k, p_reg)
            
            # Find closest point on mesh to s_k
            c_k, min_dist = search_tree_point(root, V, Indices, s_k, 1)
            cs.append(c_k)
            
            #save if correspondences are close enough
            if min_dist < eta:
                A_correspondences.append(d_k)
                B_correspondences.append(c_k)
        
        cs = np.array(cs)
        
        # Refine registration transformation using correspondences
        if len(A_correspondences) > 0:
            A_corr = np.array(A_correspondences)
            B_corr = np.array(B_correspondences)
            R_reg, p_reg = point2point_3Dregistration(A_corr, B_corr)
        
        # Calculate error metrics
        if len(A_correspondences) > 0:
            transformed_A = np.array([frame_transformation(R_reg, d, p_reg) for d in A_correspondences])
            e = B_corr - transformed_A
            # sigma = RMS error per element
            sigma = np.sqrt(np.sum(e * e) / e.size)
            # average absolute error per element
            epsilon_avg = np.sum(np.abs(e)) / e.size
            eta = max(3 * epsilon_avg, 0.1175)
        else:
            break
        
        counter += 1
        if counter > 1000:
            print(f"Max iterations reached for {data_prefix}")
            break
    
    # compute final s_k values
    ss = np.array([frame_transformation(R_reg, d_k, p_reg) for d_k in ds])
    
    #write outputs to file
    parts = data_prefix.split('-')
    if len(parts) >= 2:
        letter = parts[1].upper()  #get letters
        if letter <= 'F':
            out_name = f"PA4-{letter}-Debug-Output.txt"
        else:
            out_name = f"PA4-{letter}-Unknown-Output.txt"
    else:
        out_name = f"{data_prefix}-Output.txt"
    
    write_output_files(out_name, ss, cs, Ns)
    print(f"  Completed {data_prefix}: {counter} iterations")


def main():
    extract_path = input("enter dataset path:").strip()
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