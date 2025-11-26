
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

def search_with_boxes(a, lower, upper, tris):
    """
    Perform search for closest point on mesh with bounding boxes
    """
    a = np.asarray(a)
    Nt = tris.shape[0]
    bound2 = np.inf #set initial bound to infinity
    closest = None #Variable for holding closest point 

    for i in range(Nt):
        #For each triangle, check if a is within the bounding box (determined with current smallest distance)
        if np.any((a < (lower[i] - np.sqrt(bound2))) | (a > (upper[i] + np.sqrt(bound2)))):
            continue #if it is then keep checking, otherwise discard and go to next triangle

        p, q, r = tris[i]
        c, d2 = closest_point_on_triangle(a, p, q, r) #find closest point on current triangle 
        if d2 < bound2:#if distance to point is less than current smallest distance, update closest point and distance
            bound2 = d2
            closest = c

    return closest, bound2

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
def process_dataset(data_prefix, extract_path):
    """Run full matching algorithm for a single dataset prefix."""
    print(f"Processing {data_prefix}...")
    try:
    # --- 1. Parse input files ---
        Na, A, A_tip = parseBodyA(os.path.join(extract_path,"Problem3-BodyA.txt"))
        Nb, B, B_tip = parseBodyB(os.path.join(extract_path,"Problem3-BodyB.txt"))
        Nv , V, Nt, Indices = parseMesh(os.path.join(extract_path,"Problem3MeshFile.sur"))
        A_frames, B_frames, D_frames, Nd, Ns = parseSampleReadings(os.path.join(extract_path, f"{data_prefix}-SampleReadingsTest.txt"), Na, Nb)
    except Exception as e:
        print(f"  Error processing {data_prefix}: {e}")

    # --- 2. Find bounds for each triangle ---
    lower, upper, tris = compute_bounds(V, Indices)
    c = []
    d = [] 

    A_tip = np.asarray(A_tip).reshape(3,)

    # 3. For each sample find closest points to mesh 
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

        # Find closest point to d_k on mesh 
        c_k, _ = search_with_boxes(d_k, lower, upper, tris)

        c.append(c_k)
        d.append(d_k)
    
    # 4. Write results to output file
    out_name = f"{data_prefix}-myOutput.txt"
    write_output_files(out_name, d , c, Ns)

def main():
    """Main loop — process all datasets in directory."""
    # Ask the user for the dataset path

    extract_path = input("Please enter the path to the dataset folder: ").strip()  # Get user input
    
    # ensure the path is valid
    while not os.path.isdir(extract_path):
        print("The path you entered is invalid. Please try again.")
        extract_path = input("Please enter the path to the dataset folder: ").strip()

    datasets = os.listdir(extract_path) #extract datasets  

    #Run pipeline for each dataset in folder
    for fname in datasets:
        if fname.endswith("-SampleReadingsTest.txt"):
            data_prefix = fname[:-len("-SampleReadingsTest.txt")]
            process_dataset(data_prefix, extract_path)
        else:
            print(f"Skipping {fname}")


if __name__ == "__main__":
    main()