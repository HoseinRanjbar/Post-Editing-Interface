import paramiko
import os
from scp import SCPClient

def frame_interpolate(fgf, lgf, ff, lf, output_path, frame_number,config, video_size):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname='cluster.s3it.uzh.ch', username=config['s3it']['username'], password=config['s3it']['password'])


    # Step 1: Transfer the file to the remote server
    fgf_remote_file_path = ''
    lgf_remote_file_path = ''
    ff_remote_file_path = ''
    lf_remote_file_path = ''
    result_folder1 = ''
    result_folder2 = ''
    # Define the directory where all commands will run
    # Define commands
    working_directory = ''
    remote_env_activate = 'source /home/environment_path/env/bin/activate'
    remote_allocate_gpu = 'srun --pty --time=0:15:0 --mem-per-cpu=32G --cpus-per-task=7 --partition=lowprio --gpus=A100:1 bash -c'
    matplotlib = 'export MPLCONFIGDIR=/tmp/$USER/mpl_cache'


    sftp = ssh.open_sftp()

    sftp.put(fgf, fgf_remote_file_path)
    sftp.put(lgf, lgf_remote_file_path)
    sftp.put(ff, ff_remote_file_path)
    sftp.put(lf, lf_remote_file_path)
    sftp.close()

    remote_first_frame_interpolation = f'rm -rf {result_folder1} && rm -rf {result_folder2} && cd {working_directory} && python demo_FCVG.py --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
        --controlnext_path checkpoints/controlnext.safetensors --output_dir results --height {video_size[1]} --width {video_size[0]} --interp_frames_number {frame_number}\
        --unet_path checkpoints/unet.safetensors --num_inference_steps 25 --control_weight 1.0 --image1_path {ff_remote_file_path} --image2_path {fgf_remote_file_path}'
    
    remote_second_frame_interpolation = f'cd {working_directory} && python demo_FCVG.py --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
        --controlnext_path checkpoints/controlnext.safetensors --output_dir results --height {video_size[1]} --width {video_size[0]} --interp_frames_number {frame_number}\
        --unet_path checkpoints/unet.safetensors --num_inference_steps 25 --control_weight 1.0 --image1_path {lgf_remote_file_path} --image2_path {lf_remote_file_path}'

    # Combine all commands into one script executed by `srun`
    combined_command = f'{remote_allocate_gpu} "{remote_env_activate} && {matplotlib} &&{remote_first_frame_interpolation} && {remote_second_frame_interpolation}"'
    
    stdin, stdout, stderr = ssh.exec_command(combined_command)
    
    print("Frame Interpolation:")
    print(stdout.read().decode())
    print(stderr.read().decode())  # Also print any errors

    # Create SCP client to transfer files
    scp = SCPClient(ssh.get_transport())
    
    # Transfer the synthesized video from the remote machine to your local machine
    scp.get(result_folder1+'/result.mp4', output_path+'/f_trans.mp4')
    scp.get(result_folder2+'/result.mp4', output_path+'/s_trans.mp4')
    
    # Close the SCP and SSH connections
    scp.close()
    ssh.close()

