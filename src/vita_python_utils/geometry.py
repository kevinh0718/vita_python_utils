import numpy as np
from scipy.ndimage import binary_dilation
from skimage import measure
import trimesh
import pyvista as pv
import vtk


def define_dilate_region(skull_mask, thickness=3.5, voxelsize=0.375):
    """
    Define the scalp area by dilating the skull mask.

    Args:
        skull_mask (3D numpy array): Binary mask where skull is 1 and background is 0.
        thickness (mm): Number of pixels to extend outward.

    Returns:
        scalp_mask (3D numpy array): Binary mask where the scalp area is 1.
    """
    # Create a structuring element for dilation (3D cube)
    structuring_element = np.ones((3, 3, 3))  # 3x3x3 cube, can be modified

    # Dilate the skull by the given thickness
    dilated_skull = binary_dilation(skull_mask, structure=structuring_element, iterations=int(thickness/voxelsize))

    # Subtract the original skull to get only the added layer (scalp)
    scalp_mask = dilated_skull.astype(np.uint8) - skull_mask.astype(np.uint8)

    return scalp_mask

def random_index_of_one(arr):
    """
    Selects a random index (x, y, z) where the value in the 3D array is 1.
    
    Args:
        arr (numpy.ndarray): A 3D array containing 0s and 1s.
    
    Returns:
        tuple: A randomly selected index (x, y, z) where arr[x, y, z] == 1.
        Returns None if there are no 1s in the array.
    """
    indices = np.argwhere(arr == 1)  # Find all (x, y, z) indices where value is 1
    if indices.size == 0:
        return None  # No 1s found
    return indices[np.random.choice(indices.shape[0])]  # Randomly select one

def save_np_vtk(tar_region, save_path, spacing=0.375, smooth=True, filetype='vtk'):
    # Define padding width
    PAD_WIDTH = 5
    # --- 1. Pad the input data ---
    tar_region_int = tar_region.astype(np.uint8)
    padded_mask = np.pad(tar_region_int, pad_width=PAD_WIDTH, mode='constant', constant_values=0)
    
    # --- 2. Run `marching_cubes` on the padded volume ---
    print("Running marching cubes to generate mesh...")
    verts, faces, normals, values = measure.marching_cubes(padded_mask, level=0.5)
    # --- FIX 1: Correct Origin and Voxel Spacing ---
    # A. Origin Correction (Move the mesh back by the padding amount)
    verts = verts - PAD_WIDTH + 0.5
    # B. Spacing Correction (Scale the mesh according to voxel size)
    # verts is shape (N, 3). spacing is shape (3,). Use NumPy broadcasting.
    verts[:, 0] *= spacing # Scale X
    verts[:, 1] *= spacing # Scale Y
    verts[:, 2] *= spacing # Scale Z
    # --- 3. Use `trimesh` to load the mesh and smooth it ---
    # Create a trimesh object from the vertices and faces
    original_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # Assuming you have a `my_mesh` object from `trimesh`
    # loaded with your vertices and faces.
    if original_mesh.is_watertight:
        print("The mesh is watertight (closed).")
    else:
        print("The mesh is NOT watertight (it has holes).")
        original_mesh.fill_holes()
        if original_mesh.is_watertight:
            print("Holes successfully filled.")
        else:
            print("Warning: Mesh still has holes after fill_holes().")
    # Perform smoothing using the correct function from the `smoothing` module
    # `trimesh.smoothing.filter_laplacian` returns a new set of vertices
    print(f"Original mesh: {original_mesh.vertices.shape[0]} vertices.")
    if smooth:
        print("Smoothing the generated mesh using `trimesh`...")
        smoothed_mesh = trimesh.smoothing.filter_laplacian(original_mesh, iterations=10, lamb=0.5)
        print(f"Smoothed mesh: {smoothed_mesh.vertices.shape[0]} vertices.")
    else:
        smoothed_mesh = original_mesh
    # --- 4. Save the smoothed mesh to an STL file ---
    if filetype=='stl':
        save_path += ".stl"
        smoothed_mesh.export(save_path, file_type='stl')
    else:
        save_path += ".vtk"
        pv_mesh = pv.wrap(smoothed_mesh)
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputDataObject(pv_mesh)
        writer.SetFileVersion(42)  # Set the file version to 4.2, 42 corresponds to version 4.2
        writer.SetFileName(save_path)
        writer.Write()
    print(f"Workflow complete. The file is ready. Saved as '{save_path}'")
    return save_path

def tubify_mesh(path: str, radius: float = 0.0040) -> pv.PolyData:
    """
    Converts a network of lines or poly-lines (e.g., vessel skeleton data) 
    into a 3D tubular surface mesh with a specified radius.

    This process is known as 'tubification' or 'sweeping' and is essential 
    for visualizing 1D geometry in 3D. The output format is PyVista PolyData.

    Parameters:
    ----------
    path : str
        The file path to the input mesh/poly-line data. The file should 
        ideally be in a format supported by VTK/PyVista (e.g., .vtk, .vtp).
        The data must contain point-data named 'radius' for variable thickness.
        
    radius : float, optional
        The default radius to use for the tube operation (e.g., in meters).
        This value is used as a fallback if the 'radius' scalar data is missing
        or for parts of the poly-line where the scalar data is zero.
        Defaults to 0.0040.

    Returns:
    -------
    pv.PolyData
        The resulting 3D tubular surface mesh (PolyData object).
    """
    
    # Read the input poly-line data from the specified path.
    ex1 = pv.read(path) 
    
    # Set the 'radius' array as the active scalar data. This tells PyVista
    # to prioritize this array when looking up thickness values for the tube operation.
    ex1.set_active_scalars('radius', preference='point') 
    
    # Generate the tubular mesh by sweeping a circle along the poly-lines.
    # The radius argument sets the base radius.
    # The 'scalars='radius'' part instructs the tube filter to use the active
    # 'radius' point data array to modulate the tube radius along its length.
    mesh = ex1.tube(radius=radius, scalars='radius')
    
    return mesh
