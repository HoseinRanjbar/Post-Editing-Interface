import paramiko
import os
from scp import SCPClient

def ControlNeXt(video_address, inserted_video_address, segment_time, ff_address, ef_address, ref_img_address, output_path, video_size, method, sample_stride, s3it):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname='cluster.s3it.uzh.ch', username=s3it['username'], password=s3it['password'])
    # Step 1: Transfer the file to the remote server
    ff_remote_path = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2/examples/ref_imgs/ff.png'
    ef_remote_path = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2/examples/ref_imgs/ef.png'
    ref_img_remote_file_path = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2/examples/ref_imgs/Image.png'
    inserted_video_remote_file_path = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2/examples/video/Segment_Video.mp4'
    video_remote_file_path = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2/examples/video/Video.mp4'
    remote_output_path = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2/outputs/output.mp4'
    # Define the directory where all commands will run
    # Define commands
    working_directory = '/home/hranjb/data/ControlNeXt/ControlNeXt-SVD-v2'
    remote_env_activate = 'source /home/hranjb/data/ControlNeXt/.env/bin/activate'
    #remote_allocate_gpu = 'srun --pty --time=0:15:0 --mem-per-cpu=32G --cpus-per-task=10 --partition=lowprio --gpus=V100:1 --constraint=GPUMEM32GB bash -l'
    remote_allocate_gpu = 'srun --pty --time=1:30:0 --mem-per-cpu=32G --cpus-per-task=7 --partition=lowprio --gpus=A100:1 bash -c'


    sftp = ssh.open_sftp()

    sftp.put(ff_address, ff_remote_path)
    sftp.put(ef_address, ef_remote_path)
    sftp.put(ref_img_address, ref_img_remote_file_path)
    sftp.put(inserted_video_address, inserted_video_remote_file_path)
    if method == 'entire':
        sftp.put(video_address, video_remote_file_path)
    else:
        video_remote_file_path = ""
    sftp.close()
    

    remote_video_synthesis = f'cd {working_directory} && python interface_controlnext.py --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
        --output_dir outputs --max_frame_num 240 --guidance_scale 3 --batch_frames 24 --sample_stride {sample_stride} --overlap 6 --height {video_size[1]} --width {video_size[0]} \
        --controlnext_path pretrained/controlnet.bin --unet_path pretrained/unet.bin {"--video_path " + video_remote_file_path if video_remote_file_path else ""} --validation_control_video_path {inserted_video_remote_file_path} \
        --ref_image_path {ff_remote_path} --first_frame_path {ff_remote_path} --end_frame_path {ef_remote_path} --start_flag {segment_time[0]} --end_flag {segment_time[1]}'
    
    # Combine all commands into one script executed by `srun`
    combined_command = f'{remote_allocate_gpu} "{remote_env_activate} && {remote_video_synthesis}"'
    
    stdin, stdout, stderr = ssh.exec_command(combined_command)
    
    # Print outputs for debugging
    print("Video Generation:")
    print(stdout.read().decode())
    print(stderr.read().decode())  # Also print any errors

    # Create SCP client to transfer files
    scp = SCPClient(ssh.get_transport())
    
    # Transfer the synthesized video from the remote machine to your local machine
    scp.get(remote_output_path, output_path)
    
    # Close the SCP and SSH connections
    scp.close()
    ssh.close()

