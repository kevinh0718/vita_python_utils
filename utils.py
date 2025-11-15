import numpy as np
from scipy.ndimage import binary_dilation
import pyvista as pv
import vtk
from skimage import measure
import trimesh
import subprocess
import os
import sys # Used for flushing output
from multiprocessing import Pool, cpu_count

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
    return tuple(indices[np.random.choice(indices.shape[0])])  # Randomly select one

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

def gen_vita_docker_com(
    ves_pos, num_ves, gen_prefix="scalp_left", vtk_name="scalp_vas_right.vtk",
    vita_path="/scratch/ciml_nhp/vita_headphantom"):
    """
    Executes the vessel_synthesis binary inside the Docker container.
    """
    # --- 1. Define Environment and Parameters ---
    
    # Path where your input/output files will live on the HOST machine
    host_volume_path = os.path.abspath(vita_path)
    
    # Path inside the container (defined by the -v flag)
    container_work_dir = "/app/vita_example"

    # Construct the parameters list (this is what you generated in Python)
    # The actual parameters depend on your Vita usage (e.g., number of vessels, iterations)
    vita_params = [
        str(ves_pos[0]),
        str(ves_pos[1]),
        str(ves_pos[2]),
        str(num_ves),
        "data/generated/"+gen_prefix,
        "data/vas_region/"+vtk_name,
        "0"
    ]

    # --- 2. Construct the Shell Command (The Trick) ---
    # We use 'sh -c "..."' to execute a string of commands, which lets us use 'cd' and '&&'
    vita_command_string = (
        f"cd {container_work_dir} && ./vessel_synthesis " + " ".join(vita_params)
    )

    # --- 3. Construct the Full Docker Command ---
    docker_command = [
        "docker", "run",
        "--rm",                                 # Automatically remove the container after it exits
        "-v", f"{host_volume_path}:{container_work_dir}", # Mount the host directory to the container
        "kevinh0718/vita_talou_cco:latest",     # The Docker image
        "sh",                                   # Execute the shell
        "-c",                                   # Pass the command string to the shell
        vita_command_string                     # The command string itself
    ]
    return docker_command

def run_vita_with_monitoring(docker_command,
                             max_count=3, print_all=0,
                             termination_message="Testing inside domain condition for 10000 points",):
    """
    Executes the Docker command and monitors its STDOUT for a specific message,
    terminating the process if the message count exceeds max_count.
    """
    
    # --- 1. Start the Docker process asynchronously ---
    if print_all>1:
        print("Starting VItA CCO process with live monitoring...")
    
    # We use Popen, which starts the process and returns immediately.
    # stdout=subprocess.PIPE is crucial: it redirects the process's output 
    # so Python can read it.
    try:
        process = subprocess.Popen(
            docker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for simpler reading
            text=True,
            bufsize=1,                 # Line-buffering (read output line-by-line)
            universal_newlines=True    # Ensure proper text decoding
        )
    except Exception as e:
        print(f"Error starting process: {e}")
        return False
        
    # --- 2. Implement Live Monitoring and Termination Logic ---
    message_count = 0
    
    # process.stdout is a file-like object we can read from
    while True:
        # Read a line from the process's output
        line = process.stdout.readline()
        
        # If the line is empty and the process has finished, break the loop
        if not line and process.poll() is not None:
            break
            
        # Print the output live (optional, but good for debugging)
        if print_all>1:
            sys.stdout.write(line)
        sys.stdout.flush() # Force output to appear immediately

        # Check for the termination condition
        if termination_message in line:
            message_count += 1
            if print_all>0:
                print(f"[MONITOR] Found termination trigger. Count: {message_count}/{max_count}")
            
            if message_count >= max_count:
                if print_all>0:
                    print(f"[MONITOR] Count reached {max_count}. Terminating process early...")
                process.terminate() # Send SIGTERM signal
                try:
                    # Wait briefly to see if it terminates
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # If it doesn't terminate gracefully, kill it
                    process.kill()
                break
        else:
            message_count = 0
        
    # --- 3. Check Final Status (If the process finished normally) ---
    
    # poll() returns the exit code if the process has terminated
    exit_code = process.wait() 
    
    if exit_code == 0:
        if print_all>1:
            print("--- EXECUTION SUCCESSFUL (Finished Normally) ---")
        return True
    else:
        print(f"--- EXECUTION FAILED (Exit Code: {exit_code}) ---")
        return False

def _process_subvolume(args):
    """
    Worker function to process a single sub-volume and return its mask.
    This function will be executed by multiple CPU cores in parallel.
    """
    # Unpack arguments: mesh, (slice_start, slice_size), fixed_axes_sizes, spacing, axis_index
    mesh, slice_info, fixed_sizes, dx, parallel_axis = args
    # print("input: ", slice_info, fixed_sizes)
    
    start, size = slice_info
    full_shape = [0, 0, 0]
    full_shape[parallel_axis] = size # Set the size for the slicing axis
    
    # Reconstruct the full shape list for numpy.zeros/reshape
    for i in range(3):
        if i != parallel_axis:
            full_shape[i] = fixed_sizes[i][1]
    Nx, Ny, Nz = full_shape
    # --- 1. Generate Local Coordinates for the Sub-volume ---
    
    # Calculate the coordinate ranges for this specific slice
    coords = []
    for i, (full_size, fixed_size) in enumerate(zip(full_shape, fixed_sizes)):
        # print("each axis: ", full_size, fixed_size)
        if i == parallel_axis:
            # Slicing axis: starts at the global offset
            c_start = (fixed_size[0] + start) * dx 
            c_end = (fixed_size[0] + start + size) * dx
            # print("slice axis: ", c_start, c_end, start, size)
            coord_range = np.linspace(c_start, c_end, num=size, endpoint=False)
        else:
            # Fixed axes: uses the global coordinate range
            c_start = fixed_size[0] * dx 
            c_end = (fixed_size[0] + fixed_size[1]) * dx
            coord_range = np.linspace(c_start, c_end, num=fixed_size[1], endpoint=False)
        coords.append(coord_range.astype(np.float32))

    # Create the local meshgrid
    x, y, z = np.meshgrid(coords[0], coords[1], coords[2])

    # --- 2. Geometric Query ---
    ugrid = pv.StructuredGrid(x, y, z).cast_to_unstructured_grid()
    
    # Run the point-in-mesh test
    selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0, check_surface=False)
    
    # --- 3. Return the Mask ---
    # Reshape to the local logical shape and ensure correct memory order (Nx, Ny, Nz)
    mask = selection.point_data['SelectedPoints'].view(np.bool_)
    mask_np = mask.reshape(Nz, Nx, Ny).transpose(1,2,0)

    return mask_np

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

def to_structured_grid_parallel(mesh, shape=(725, 248, 248),
                                origin=(0, 0, 0), dx=0.015,
                                parallel_axis=0, num_slices=4, num_workers=None):
    """
    Splits the volume along parallel_axis and processes each slice in parallel.
    
    Parameters:
    - mesh: The input trimesh object.
    - shape: Global shape (Nx, Ny, Nz).
    - origin: Global start coordinates (Nx0, Ny0, Nz0).
    - dx: Voxel size.
    - parallel_axis: The axis to slice along (0, 1, or 2).
    - num_workers: Number of processes to use. Defaults to CPU count.
    - slice_size: The thickness of each sub-volume slice.
    """
    
    num_workers = num_workers or cpu_count()
    total_len = shape[parallel_axis]
    
    # Determine the number of slices needed
    slice_size = int(np.ceil(total_len / num_slices))
    
    # --- 1. Prepare Arguments for the Worker Pool ---
    task_args = []
    
    for i in range(num_slices):
        start_index = i * slice_size
        if total_len - start_index <= 0:
            print(f"Slice {i + 1} exceeds volume bounds. Stopping slice generation.")
            break
        current_size = min(slice_size, total_len - start_index)
        
        # Determine the sizes of the non-slicing axes (needed for worker function)
        fixed_sizes_with_start = []
        for j in range(3):
            if j == parallel_axis:
                # Store the start index and size for the slicing axis
                fixed_sizes_with_start.append((origin[j], shape[j])) 
            else:
                # Store the start index and size for the fixed axes
                fixed_sizes_with_start.append((origin[j], shape[j]))
        
        # Package the arguments for the worker function
        # Arguments: mesh, (start_index, current_size), fixed_axes_sizes, dx, parallel_axis
        task_args.append((mesh, (start_index, current_size), 
                          fixed_sizes_with_start, dx, parallel_axis))

    # --- 2. Execute Parallel Processing ---
    print(f"Splitting volume into {len(task_args)} slices and processing with {num_workers} workers.")
    
    with Pool(processes=num_workers) as pool:
        # pool.map applies the worker function to all task arguments
        results = pool.map(_process_subvolume, task_args)
        
    # --- 3. Combine Results and Finalize ---
    
    # Use np.concatenate to stitch the resulting mask arrays back together
    # Stitching is done along the original parallel_axis
    final_mask = np.concatenate(results, axis=parallel_axis)
    
    print("Parallel processing complete. Volume stitched.")

    # Final check on shape
    if final_mask.shape != shape:
         print(f"Warning: Final shape {final_mask.shape} does not match target shape {shape}. Check slicing logic.")

    return final_mask
