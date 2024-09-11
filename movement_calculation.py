import numpy as np

def calculate_speed(p1_x, p1_y, p2_x, p2_y, time_interval):
    """
    Calculate the speed between two points.
    
    Parameters:
    p1 (tuple): Coordinates (x, y) at time t1.
    p2 (tuple): Coordinates (x, y) at time t2.
    time_interval (float): Time interval between t1 and t2.
    
    Returns:
    float: Speed
    """
    distance = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
    return distance / time_interval

def calculate_acceleration(v1, v2, time_interval):
    """
    Calculate the acceleration between two speeds.
    
    Parameters:
    v1 (float): Speed at time t1.
    v2 (float): Speed at time t2.
    time_interval (float): Time interval between t1 and t2.
    
    Returns:
    float: Acceleration
    """
    return (v2 - v1) / time_interval


def movement(pose_data, fps):
    """
    Process pose data to calculate speed and acceleration for a specific body part.
    
    Parameters:
    pose_data (list): List of dictionaries containing pose information for each frame.
    body_part (str): Key name of the body part to track.
    fps (int): Frames per second of the video.
    
    Returns:
    list: Speed values for each frame.
    list: Acceleration values for each frame.
    """
    time_interval = 1 / fps

    right_hand_speeds = []
    left_hand_speeds = []
    right_hand_accelerations = []
    left_hand_accelerations = []
    
    for idx in range(len(pose_data)-1):

        #print(pose_data[idx].shape)
        right_hand_speed = 0
        left_hand_speed = 0
        right_hand_acceleration = 0
        left_hand_acceleration = 0
        for i in range(pose_data[idx].shape[1]):

            if i in list(range(6, 27)) or i == 3:
                p1_x = pose_data[idx][0,i]
                p1_y = pose_data[idx][1,i]
                p2_y = pose_data[idx+1][1,i]
                p2_x = pose_data[idx+1][0,i]
                
                speed = calculate_speed(p1_x, p1_y, p2_x, p2_y, time_interval)
                right_hand_speed += speed

                
                if idx > 1:
                    acceleration = calculate_acceleration(right_hand_speeds[-2], right_hand_speeds[-1], time_interval)
                    right_hand_acceleration += acceleration


            if i in list(range(27,48)) or i == 2:
                p1_x = pose_data[idx][0,i]
                p1_y = pose_data[idx][1,i]
                p2_y = pose_data[idx+1][1,i]
                p2_x = pose_data[idx+1][0,i]
                
                speed = calculate_speed(p1_x, p1_y, p2_x, p2_y, time_interval)
                left_hand_speed += speed
                
                if idx > 1:
                    acceleration = calculate_acceleration(left_hand_speeds[-2], left_hand_speeds[-1], time_interval)
                    left_hand_acceleration += acceleration
        
        right_hand_speeds.append(right_hand_speed)
        left_hand_speeds.append(left_hand_speed)
        right_hand_accelerations.append(right_hand_acceleration)
        left_hand_accelerations.append(left_hand_acceleration)
        # pose_movement.append({'right_hand_accelerate':right_hand_accelerates,
        #                       'right_hand_speed':right_hand_speeds,
        #                       'left_hand_accelerate':left_hand_accelerates,
        #                       'left_hand_speed':left_hand_speeds})
      
    return right_hand_speeds, right_hand_accelerations, left_hand_speeds, left_hand_accelerations



