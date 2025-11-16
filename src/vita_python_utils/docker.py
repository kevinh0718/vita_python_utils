import subprocess
import sys
import os

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
        gen_prefix,
        vtk_name,
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
