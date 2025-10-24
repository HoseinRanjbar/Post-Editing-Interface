import paramiko
import os
from scp import SCPClient

def find_least_busy_gpu(ssh):
    """
    Query all GPUs' memory usage via nvidia-smi, parse, return the GPU index
    with the least memory used (or any other criterion).
    Returns an integer GPU index, e.g. 0, 1, 2...
    """
    # 1) Run remote nvidia-smi
    #    We'll ask for index and memory.used in CSV for easy parsing
    command = (
        "nvidia-smi --query-gpu=index,memory.used "
        "--format=csv,noheader,nounits"
    )
    stdin, stdout, stderr = ssh.exec_command(command)
    output = stdout.read().decode().strip()
    if not output:
        # if for some reason we didn't get data, fallback to GPU 0
        return 0

    """
    Example output lines from nvidia-smi:
    0, 1024
    1, 0
    2, 4003
    ...
    """
    lines = output.split('\n')
    gpu_usages = []
    for line in lines:
        # each line is something like "0, 234"
        parts = line.split(',')
        if len(parts) < 2:
            continue
        gpu_index_str = parts[0].strip()
        mem_used_str  = parts[1].strip()
        try:
            gpu_index = int(gpu_index_str)
            mem_used  = int(mem_used_str)
            gpu_usages.append((gpu_index, mem_used))
        except ValueError:
            pass

    # 2) pick the GPU with the minimal memory usage
    if not gpu_usages:
        return 0  # fallback
    gpu_usages.sort(key=lambda x: x[1])  # sort by memory used ascending
    best_gpu = gpu_usages[0][0]
    return best_gpu

def facefusion(video_address, ref_img_address, output_address, config):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname='habra.cl.uzh.ch', username=config['s3it']['username'], password=config['s3it']['password'])


    # Step 1: Transfer the file to the remote server
    ref_img_remote_file_path = 'facefusion_directory/.assets/reference_image/Image.png'
    video_remote_file_path = 'facefusion_directory/.assets/target_video/Video.mp4'
    video_frame_folder =  'facefusion_directory/.assets/target_video/Video'
    remote_output_path =  'facefusion_directory/.assets/outputs/output.mp4'
    # Define the directory where all commands will run
    # Define commands
    working_directory = 'facefusion_directory_path'
    remote_env_activate = 'source environment_directory/bin/activate'
    ffmpeg = 'export PATH="/home/hranjb/bin:$PATH"'

    sftp = ssh.open_sftp()
    print(ref_img_remote_file_path)
    sftp.put(ref_img_address, ref_img_remote_file_path)
    sftp.put(video_address, video_remote_file_path)
    sftp.close()

    best_gpu_idx = find_least_busy_gpu(ssh)
    print(f"Least busy GPU is index {best_gpu_idx}")
    

    remote_face_swapping = f'rm -rf {video_frame_folder} && cd {working_directory} && CUDA_VISIBLE_DEVICES={best_gpu_idx} python face_swapper.py --source_path {ref_img_remote_file_path} --target_path {video_remote_file_path} --output_path {remote_output_path}'
    
    # Combine all commands into one script executed by `srun`
    combined_command = f'{remote_env_activate} && {ffmpeg} &&{remote_face_swapping}'
    
    stdin, stdout, stderr = ssh.exec_command(combined_command)
    
    print("Post Etiting:")
    print(stdout.read().decode())
    print(stderr.read().decode())  # Also print any errors

    # Create SCP client to transfer files
    scp = SCPClient(ssh.get_transport())
    
    # Transfer the synthesized video from the remote machine to your local machine
    scp.get(remote_output_path, output_address)
    
    # Close the SCP and SSH connections
    scp.close()
    ssh.close()


