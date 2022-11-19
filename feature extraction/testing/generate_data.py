from blazepose_est import runModelWithVideo, combineArrays
import numpy as np
import pandas as pd 
import os
from path_vars import root, root2

subject = ['TCOA01', 'TCOA02', 'TCOA03', 'TCOA04', 'TCOA05', 'TCOA06', 'TCOA07', 'TCOA08', 'TCOA09', 'TCOA10', 'TCOA11', 'TCOA12', 'TCOA13',
     'TCOA14', 'TCOA15', 'TCOA16', 'TCOA17', 'TCOA18', 'TCOA19', 'TCOA20', 'TCOA21', 'TCOA22', 'TCOA23', 'TCOA24', 'TCOA26', 'TCOA27', 'TCOA28', 'TCOA29', 'TCOA30', 'TCOA31', 'TCOA32', 'TCOA33']
    
videoFront = ['Brush_knee_twist_step_left_ft_front_Trim', 'Brush_knee_twist_step_right_ft_front', 'Golden_rooster_left_ft_front', 'Golden_rooster_right_ft_front', 
    'Grasp_the_sparrow_tail_left_ft_front', 'Grasp_the_sparrow_tail_right_ft_front', 'Push_Left_ft_front', 'Push_Right_ft_front', 'Raising_the_power_front', 
    'Wave_hands_like_clouds_both_hand_front', 'Wave_hands_like_clouds_left_hand_front']
    
videoLateral = ['Brush_knee_twist_step_left_ft_lateral', 'Brush_knee_twist_step_right_ft_lateral', 'Golden_rooster_left_ft_lateral', 'Golden_rooster_right_ft_lateral', 
    'Grasp_the_sparrow_tail_left_ft_lateral', 'Grasp_the_sparrow_tail_right_ft_lateral', 'Push_Left_ft_lateral', 'Push_Right_ft_lateral',
    'Raising_the_power_lateral', 'Wave_hands_like_clouds_both_hand_lateral', 'Wave_hands_like_clouds_left_hand_lateral']

# make note of expections
def generateData(v):
    data = []
    if v == "videolateral":
        view = videoLateral
    elif v == "videofront": 
        view = videoFront

    for i in range(len(subject)):
        for k in range(len(view)):
            # exceptions on 8, 9, 14, 26, 28, 29, 31, 32; should be caught by if
                path = root + subject[i] + '\\' + view[k] + '.mp4'
                if os.path.exists(path) and os.path.isfile(path):
                    print('exists')
                    if len(data) == 0: 
                        data = runModelWithVideo(subject[i], view[k], path, 0.06)
                    else: 
                        data = combineArrays(data, runModelWithVideo(subject[i], view[k], path, 0.06))
    return data

# export final data 
column_vals = ['subject', 'move', 'timestamp', 'marker_0_x', 'marker_0_y', 'marker_0_z', 'marker_11_x', 'marker_11_y', 'marker_11_z', 'marker_12_x', 'marker_12_y', 'marker_12_z', 'marker_13_x',  'marker_13_y',  'marker_13_z', 
    'marker_14_x', 'marker_14_y', 'marker_14_z','marker_15_x', 'marker_15_y', 'marker_15_z', 'marker_16_x', 'marker_16_y', 'marker_16_z', 'marker_23_x', 'marker_23_y', 'marker_23_z', 'marker_24_x', 'marker_24_y', 'marker_24_z', 
    'marker_25_x', 'marker_25_y', 'marker_25_z','marker_26_x', 'marker_26_y', 'marker_26_z', 'marker_27_x', 'marker_27_y', 'marker_27_z', 'marker_28_x', 'marker_28_y', 'marker_28_z']

data_ant = generateData("videofront")
df_ant = pd.DataFrame(data_ant, columns=column_vals)
df_ant.to_csv(root2 + 'Subject0_Anterior.csv')


