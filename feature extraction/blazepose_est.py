# Based code on https://www.analyticsvidhya.com/blog/2021/10/human-pose-estimation-using-machine-learning-in-python/

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from path_vars import root 

# Run video with markers created with Open CV
# Gets x, y, z, and visibility for each joint 
# Commented out sections correspond to rendering the graphics 
def runModelWithVideo(subject, move, path, samplingRate=1): 
    mpPose = mp.solutions.pose
    pose = mpPose.Pose(model_complexity=2)
    mpDraw = mp.solutions.drawing_utils
    allTime = []

    cap = cv2.VideoCapture(path)

    t_msec = 0 

    while True:
        t_msec += 1000*(samplingRate)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_msec)
        success, img = cap.read()

        if not success:
            if len(allTime) != 0:
                print("video completed ", path) 
                return allTime
            else:
                print("error occured provessing video ", path)
                break
        else:
            perFrame = []    
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            # consider POSE_WORLD_LANDMARKS for distance as well, world vs normal
            if results.pose_landmarks:
                perFrame.append(subject)
                perFrame.append(move)
                perFrame.append(t_msec)
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                i = 0
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w,c = img.shape
                    # store data for joints of interest
                    if id in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                        # store in perFrame  
                        perFrame.append(lm.x)
                        perFrame.append(lm.y)
                        perFrame.append(lm.z)
                        i += 1
                    #cx, cy = int(lm.x*w), int(lm.y*h)
                    #cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

            #cv2.imshow("Image", img)
            #cv2.waitKey(1)
            allTime.append(perFrame)
    cap.release()
    return allTime


# Combine 2D arrays in python 
def combineArrays(a1, a2):
    # add each row in a2 to a1
    for row in a2:
        if len(row) != 0:
            a1.append(row)
    return a1

# testing run with model
def main():
    subject = ['TCOA01', 'TCOA02', 'TCOA03', 'TCOA04', 'TCOA05', 'TCOA06', 'TCOA07', 'TCOA08', 'TCOA09', 'TCOA10', 'TCOA11', 'TCOA12', 'TCOA13',
     'TCOA14', 'TCOA15', 'TCOA16', 'TCOA17', 'TCOA18', 'TCOA19', 'TCOA20', 'TCOA21', 'TCOA22', 'TCOA23', 'TCOA24', 'TCOA26', 'TCOA27', 'TCOA28', 'TCOA29', 'TCOA30', 'TCOA31', 'TCOA32', 'TCOA33']
    
    videoFront = ['Brush_knee_twist_step_left_ft_front', 'Brush_knee_twist_step_right_ft_front', 'Golden_rooster_left_ft_front', 'Golden_rooster_right_ft_front', 
    'Grasp_the_sparrow_tail_left_ft_front', 'Grasp_the_sparrow_tail_right_ft_front', 'Push_Left_ft_front', 'Push_Right_ft_front', 'Raising_the_power_front', 
    'Wave_hands_like_clouds_both_hand_front', 'Wave_hands_like_clouds_left_hand_front']
    
    p = root + subject[0] + '\\' + videoFront[-1] + '.mp4'
    p2 = root + subject[1] + '\\' + videoFront[0] + '.mp4'
    column_vals = ['subject', 'move', 'timestamp', 'marker_0_x', 'marker_0_y', 'marker_0_z', 'marker_11_x', 'marker_11_y', 'marker_11_z', 'marker_12_x', 'marker_12_y', 'marker_12_z', 'marker_13_x',  'marker_13_y',  'marker_13_z', 
    'marker_14_x', 'marker_14_y', 'marker_14_z','marker_15_x', 'marker_15_y', 'marker_15_z', 'marker_16_x', 'marker_16_y', 'marker_16_z', 'marker_23_x', 'marker_23_y', 'marker_23_z', 'marker_24_x', 'marker_24_y', 'marker_24_z', 
    'marker_25_x', 'marker_25_y', 'marker_25_z','marker_26_x', 'marker_26_y', 'marker_26_z', 'marker_27_x', 'marker_27_y', 'marker_27_z', 'marker_28_x', 'marker_28_y', 'marker_28_z']
    data = runModelWithVideo(subject[0], videoFront[-1], p, 4)
    print(len(data), len(data[0]))
    data2 = runModelWithVideo(subject[1], videoFront[0], p2, 4)
    print(len(data2), len(data2[0]))
    result = combineArrays(data, data2)
    df2 = pd.DataFrame(result, columns = column_vals)
    print(df2)

if __name__ == "__main__":
    main()