import cv2
import mediapipe as mp
import numpy as np


face_skeleton = {}
skeleton_data = {}

def mediapipe(video_address):

    mp_holistic = mp.solutions.holistic # Holistic model
    holistic = mp_holistic.Holistic(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    frame_index = 0
    cap = cv2.VideoCapture(video_address)
    #cap = video
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = holistic.process(frame)

        skeleton = {'face': features.face_landmarks, 'pose': features.pose_landmarks, 'right_hand': features.right_hand_landmarks, 'left_hand': features.left_hand_landmarks, 'segmentation' : features.segmentation_mask }

        data_numpy = np.zeros((3,48))
        face = np.zeros((3,11))
        pose = skeleton['pose']
        right_hand = skeleton['right_hand']
        left_hand = skeleton['left_hand']
        v_num = 0

        if pose:

            # pose keypoints
            for idx, landmark in enumerate(pose.landmark):
                if idx in range(0,11):
                    face[0, idx] = landmark.x
                    face[1, idx] = landmark.y
                    face[2, idx] = landmark.z
                    
                if idx in [11,12,13,14,23,24]:
                    data_numpy[0, v_num] = landmark.x
                    data_numpy[1, v_num] = landmark.y
                    data_numpy[2, v_num] = landmark.z
                    v_num += 1

            # right hand keypoints
            if right_hand:
                for idx, landmark in enumerate(right_hand.landmark):

                    data_numpy[0, v_num] = landmark.x
                    data_numpy[1, v_num] = landmark.y
                    data_numpy[2, v_num] = landmark.z
                    v_num += 1

            else:

                for j in range(0,21):

                    if j == 0:
                    
                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z
                        v_num += 1
                        

                    elif j in [1,2,3,4]:

                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].z
                        v_num += 1

                    elif j in [5,6,7,8,9,10,11,12]:

                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].z
                        v_num += 1

                    else:

                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].z
                        v_num += 1


            # left hand keypoints
            if left_hand:
                for idx, landmark in enumerate(left_hand.landmark):

                    data_numpy[0, v_num] = landmark.x
                    data_numpy[1, v_num] = landmark.y
                    data_numpy[2, v_num] = landmark.z
                    v_num += 1

            else:

                for j in range(0,21):

                    if j == 0:
                    
                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z
                        v_num += 1
                        

                    elif j in [1,2,3,4]:

                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].z
                        v_num += 1

                    elif j in [5,6,7,8,9,10,11,12]:

                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].z
                        v_num += 1

                    else:

                        data_numpy[0, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].x
                        data_numpy[1, v_num] = pose.landmark[mp_holistic.PoseLandmark.RIGHT_PINKY].y
                        data_numpy[2, v_num] = pose.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].z
                        v_num += 1
            

        else:

            data_numpy[0, v_num] = 0
            data_numpy[1, v_num] = 0
            data_numpy[2, v_num] = 0
            v_num += 1

        skeleton_data[frame_index] = data_numpy
        face_skeleton[frame_index] = face
        frame_index += 1


    return(skeleton_data, face_skeleton)




