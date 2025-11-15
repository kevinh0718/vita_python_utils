import numpy as np
import pyvista as pv
from multiprocessing import Pool, cpu_count

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
