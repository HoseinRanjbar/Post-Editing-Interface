import paramiko
import os
from scp import SCPClient
import yaml

def run_remote(video_address, ref_img_address, output_path, video_size):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname='cluster.s3it.uzh.ch', username = config['s3it']['username'], password = config['s3it']['password'])

    # Define the directory where all commands will run
    working_directory = config['s3it']['working_directory'] #'/home/username/data/MusePose'
    # Step 1: Transfer the file to the remote server
    video_remote_file_path = os.path.join(config['s3it']['working_directory'],'assets/videos/video.mp4')
    ref_img_remote_file_path = os.path.join(config['s3it']['working_directory'],'assets/images/image.png')

    sftp = ssh.open_sftp()
    sftp.put(ref_img_address, ref_img_remote_file_path)
    sftp.put(video_address, video_remote_file_path)
    sftp.close()

    # Define commands
    #remote_allocate_gpu = 'srun --pty --time=0:10:0 --mem-per-cpu=32G --cpus-per-task=10 --partition=lowprio --gpus=V100:1 --constraint=GPUMEM32GB bash -l'
    remote_allocate_gpu = 'srun --pty --time=0:10:0 --mem-per-cpu=32G --cpus-per-task=5 --partition=lowprio --gpus=A100:1 bash -c'
    environment_activation = config['s3it']['environment_activation']
    remote_env_activate = f'source {environment_activation}'

    remote_pose_align = (
        f'cd {working_directory} && '
        f'python {working_directory}/pose_align.py '
        f'--imgfn_refer {working_directory}/assets/images/image.png '
        f'--vidfn {working_directory}/assets/videos/video.mp4')
    
    # Define the command for video synthesis
    remote_video_synthesis = (
        f'python {working_directory}/test_stage_2.py '
        f'--config {working_directory}/configs/test_stage_2.yaml '
        f'-W {video_size[0]} '
        f'-H {video_size[1]} '
        f'--skip 0')

    print(remote_video_synthesis)
    # Combine all commands into one script executed by `srun`
    combined_command = f'{remote_allocate_gpu} "{remote_env_activate} && {remote_pose_align} && {remote_video_synthesis}"'
    
    stdin, stdout, stderr = ssh.exec_command(combined_command)
    
    # Print outputs for debugging
    print("Pose Alignment and Video Synthesis Output:")
    print(stdout.read().decode())
    print(stderr.read().decode())  # Also print any errors

    # Step 3: Find the last modified directory and retrieve the output video
    remote_script_command = f'python {working_directory}/scripts/find_last_modified.py'
    combined_command = f'{remote_env_activate} && {remote_script_command}'
    stdin, stdout, stderr = ssh.exec_command(combined_command)
    last_modified_directory = stdout.read().decode().strip()
    print(last_modified_directory)

    # Create SCP client to transfer files
    scp = SCPClient(ssh.get_transport())
    
    # Transfer the synthesized video from the remote machine to your local machine
    scp.get(last_modified_directory, output_path)
    
    # Close the SCP and SSH connections
    scp.close()
    ssh.close()

