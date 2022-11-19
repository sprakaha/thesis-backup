from cmath import nan
from helpers import selectIfGreater, selectIfLess, getDistance, noDivideByZero
import pandas as pd
import feature_ext as fe
import numpy as np
import os
from scipy.signal import find_peaks, peak_widths,  argrelextrema
from path_vars import root2, base

# Script extracts pose markers at a high samplin rate ~15 Hz
# Upon extraction, the features are directly calculated
# Then, they are adjusted with Savitzky-Golay / FFT

subject = ['TCOA01', 'TCOA02', 'TCOA03', 'TCOA04', 'TCOA05', 'TCOA06', 'TCOA07', 'TCOA08', 'TCOA09', 'TCOA10', 'TCOA11', 'TCOA12', 'TCOA13',
     'TCOA14', 'TCOA15', 'TCOA16', 'TCOA17', 'TCOA18', 'TCOA19', 'TCOA20', 'TCOA21', 'TCOA22', 'TCOA23', 'TCOA24', 'TCOA26', 'TCOA27', 'TCOA28', 'TCOA29', 'TCOA30', 'TCOA31', 'TCOA32', 'TCOA33']
    
moves = ['Brush_knee_twist_step_left', 'Brush_knee_twist_step_right', 'Golden_rooster_left', 'Golden_rooster_right_ft_front', 
    'Grasp_the_sparrow_tail_left', 'Grasp_the_sparrow_tail_right', 'Push_Left', 'Push_Right', 'Raising_the_power', 
    'Wave_hands_like_clouds_both_hand', 'Wave_hands_like_clouds_left_hand', 'Wave_hands_like_clouds_right_hand_']

videoFront = ['Brush_knee_twist_step_left_ft_front', 'Brush_knee_twist_step_right_ft_front', 'Golden_rooster_left_ft_front', 'Golden_rooster_right_ft_front', 
    'Grasp_the_sparrow_tail_left_ft_front', 'Grasp_the_sparrow_tail_right_ft_front', 'Push_Left_ft_front', 'Push_Right_ft_front', 'Raising_the_power_front', 
    'Wave_hands_like_clouds_both_hand_front', 'Wave_hands_like_clouds_left_hand_front', 'Wave_hands_like_clouds_right_hand_front']
    
videoLateral = ['Brush_knee_twist_step_left_ft_lateral', 'Brush_knee_twist_step_right_ft_lateral', 'Golden_rooster_left_ft_lateral', 'Golden_rooster_right_ft_lateral', 
    'Grasp_the_sparrow_tail_left_ft_lateral', 'Grasp_the_sparrow_tail_right_ft_lateral', 'Push_Left_ft_lateral', 'Push_Right_ft_lateral',
    'Raising_the_power_lateral', 'Wave_hands_like_clouds_both_hand_lateral', 'Wave_hands_like_clouds_left_hand_lateral']

column_vals = ['subject', 'move', 'timestamp', 'marker_0_x', 'marker_0_y', 'marker_0_z', 'marker_11_x', 'marker_11_y', 'marker_11_z', 'marker_12_x', 'marker_12_y', 'marker_12_z', 'marker_13_x',  'marker_13_y',  'marker_13_z', 
    'marker_14_x', 'marker_14_y', 'marker_14_z','marker_15_x', 'marker_15_y', 'marker_15_z', 'marker_16_x', 'marker_16_y', 'marker_16_z', 'marker_23_x', 'marker_23_y', 'marker_23_z', 'marker_24_x', 'marker_24_y', 'marker_24_z', 
    'marker_25_x', 'marker_25_y', 'marker_25_z','marker_26_x', 'marker_26_y', 'marker_26_z', 'marker_27_x', 'marker_27_y', 'marker_27_z', 'marker_28_x', 'marker_28_y', 'marker_28_z']

# frontal 
newcols = ['subject', 'move', 'timestamp','h_slope_xz', 's_slope_xz', 'lws_xdist_xy', 'rws_xdist_xy', 'h_slope_xy', 'stoh_slope_xy', 'ktoa_slope_xy', 'htok_slope_xy', 
'lws_height_xy', 'rws_height_xy', 'ls_angle_xy', 'rs_angle_xy', 'ls_adisp_xy', 'rs_adisp_xy', 'lk_angle_xy', 'rk_angle_xy', 
'lk_adisp_xy', 'rk_adisp_xy', 'la_xdisp_xy', 'ra_xdisp_xy', 'lktla_height_xy', 'rktra_height_xy', 'hand_sym_fr', 
'letlw_height_xy', 'retrw_height_xy', 'le_angle_xy', 're_angle_xy', 'le_adisp_xy', 're_adisp_xy', 'lhtla_height_xy', 'rhtra_height_xy', 
'a_dist_xy', 'la_ydisp_xy', 'ra_ydisp_xy', 'atos_dist_xy']

# lateral
newcols_lat = ['subject', 'move', 'timestamp', 'lws_ydist_yz', 'rws_ydist_yz', 'lws_xdist_yz', 'rws_xdist_yz',
 'n_angle_yz', 'lara_dist_yz', 'la_angle_yz', 'ra_angle_yz', 'lk_angle_yz']

features_all = ['subject', 'move', 'h_slope_xz_pMeanDist','h_slope_xz_pMedDist','h_slope_xz_pStdDist','h_slope_xz_tMeanDist','h_slope_xz_tMedDist','h_slope_xz_tStdDist','h_slope_xz_rMean','h_slope_xz_sMean','h_slope_xz_peakMean','h_slope_xz_widthMean','h_slope_xz_troMean','h_slope_xz_rMed','h_slope_xz_sMed','h_slope_xz_peakMed','h_slope_xz_widthMed','h_slope_xz_troMed','h_slope_xz_rMin','h_slope_xz_rMax','h_slope_xz_rAmp','h_slope_xz_rStd','h_slope_xz_sStd','h_slope_xz_peakStd','h_slope_xz_widthStd','h_slope_xz_troStd','h_slope_xz_rVar','h_slope_xz_sVar','h_slope_xz_priFreq','h_slope_xz_meanFreq','h_slope_xz_Entrop','h_slope_xz_meanPower',
's_slope_xz_pMeanDist','s_slope_xz_pMedDist','s_slope_xz_pStdDist','s_slope_xz_tMeanDist','s_slope_xz_tMedDist','s_slope_xz_tStdDist','s_slope_xz_rMean','s_slope_xz_sMean','s_slope_xz_peakMean','s_slope_xz_widthMean','s_slope_xz_troMean','s_slope_xz_rMed','s_slope_xz_sMed','s_slope_xz_peakMed','s_slope_xz_widthMed','s_slope_xz_troMed','s_slope_xz_rMin','s_slope_xz_rMax','s_slope_xz_rAmp','s_slope_xz_rStd','s_slope_xz_sStd','s_slope_xz_peakStd','s_slope_xz_widthStd','s_slope_xz_troStd','s_slope_xz_rVar','s_slope_xz_sVar','s_slope_xz_priFreq','s_slope_xz_meanFreq','s_slope_xz_Entrop','s_slope_xz_meanPower',
'lws_xdist_xy_pMeanDist','lws_xdist_xy_pMedDist','lws_xdist_xy_pStdDist','lws_xdist_xy_tMeanDist','lws_xdist_xy_tMedDist','lws_xdist_xy_tStdDist','lws_xdist_xy_rMean','lws_xdist_xy_sMean','lws_xdist_xy_peakMean','lws_xdist_xy_widthMean','lws_xdist_xy_troMean','lws_xdist_xy_rMed','lws_xdist_xy_sMed','lws_xdist_xy_peakMed','lws_xdist_xy_widthMed','lws_xdist_xy_troMed','lws_xdist_xy_rMin','lws_xdist_xy_rMax','lws_xdist_xy_rAmp','lws_xdist_xy_rStd','lws_xdist_xy_sStd','lws_xdist_xy_peakStd','lws_xdist_xy_widthStd','lws_xdist_xy_troStd','lws_xdist_xy_rVar','lws_xdist_xy_sVar','lws_xdist_xy_priFreq','lws_xdist_xy_meanFreq','lws_xdist_xy_Entrop','lws_xdist_xy_meanPower',
'rws_xdist_xy_pMeanDist','rws_xdist_xy_pMedDist','rws_xdist_xy_pStdDist','rws_xdist_xy_tMeanDist','rws_xdist_xy_tMedDist','rws_xdist_xy_tStdDist','rws_xdist_xy_rMean','rws_xdist_xy_sMean','rws_xdist_xy_peakMean','rws_xdist_xy_widthMean','rws_xdist_xy_troMean','rws_xdist_xy_rMed','rws_xdist_xy_sMed','rws_xdist_xy_peakMed','rws_xdist_xy_widthMed','rws_xdist_xy_troMed','rws_xdist_xy_rMin','rws_xdist_xy_rMax','rws_xdist_xy_rAmp','rws_xdist_xy_rStd','rws_xdist_xy_sStd','rws_xdist_xy_peakStd','rws_xdist_xy_widthStd','rws_xdist_xy_troStd','rws_xdist_xy_rVar','rws_xdist_xy_sVar','rws_xdist_xy_priFreq','rws_xdist_xy_meanFreq','rws_xdist_xy_Entrop','rws_xdist_xy_meanPower',
'h_slope_xy_pMeanDist','h_slope_xy_pMedDist','h_slope_xy_pStdDist','h_slope_xy_tMeanDist','h_slope_xy_tMedDist','h_slope_xy_tStdDist','h_slope_xy_rMean','h_slope_xy_sMean','h_slope_xy_peakMean','h_slope_xy_widthMean','h_slope_xy_troMean','h_slope_xy_rMed','h_slope_xy_sMed','h_slope_xy_peakMed','h_slope_xy_widthMed','h_slope_xy_troMed','h_slope_xy_rMin','h_slope_xy_rMax','h_slope_xy_rAmp','h_slope_xy_rStd','h_slope_xy_sStd','h_slope_xy_peakStd','h_slope_xy_widthStd','h_slope_xy_troStd','h_slope_xy_rVar','h_slope_xy_sVar','h_slope_xy_priFreq','h_slope_xy_meanFreq','h_slope_xy_Entrop','h_slope_xy_meanPower',
'stoh_slope_xy_pMeanDist','stoh_slope_xy_pMedDist','stoh_slope_xy_pStdDist','stoh_slope_xy_tMeanDist','stoh_slope_xy_tMedDist','stoh_slope_xy_tStdDist','stoh_slope_xy_rMean','stoh_slope_xy_sMean','stoh_slope_xy_peakMean','stoh_slope_xy_widthMean','stoh_slope_xy_troMean','stoh_slope_xy_rMed','stoh_slope_xy_sMed','stoh_slope_xy_peakMed','stoh_slope_xy_widthMed','stoh_slope_xy_troMed','stoh_slope_xy_rMin','stoh_slope_xy_rMax','stoh_slope_xy_rAmp','stoh_slope_xy_rStd','stoh_slope_xy_sStd','stoh_slope_xy_peakStd','stoh_slope_xy_widthStd','stoh_slope_xy_troStd','stoh_slope_xy_rVar','stoh_slope_xy_sVar','stoh_slope_xy_priFreq','stoh_slope_xy_meanFreq','stoh_slope_xy_Entrop','stoh_slope_xy_meanPower',
'ktoa_slope_xy_pMeanDist','ktoa_slope_xy_pMedDist','ktoa_slope_xy_pStdDist','ktoa_slope_xy_tMeanDist','ktoa_slope_xy_tMedDist','ktoa_slope_xy_tStdDist','ktoa_slope_xy_rMean','ktoa_slope_xy_sMean','ktoa_slope_xy_peakMean','ktoa_slope_xy_widthMean','ktoa_slope_xy_troMean','ktoa_slope_xy_rMed','ktoa_slope_xy_sMed','ktoa_slope_xy_peakMed','ktoa_slope_xy_widthMed','ktoa_slope_xy_troMed','ktoa_slope_xy_rMin','ktoa_slope_xy_rMax','ktoa_slope_xy_rAmp','ktoa_slope_xy_rStd','ktoa_slope_xy_sStd','ktoa_slope_xy_peakStd','ktoa_slope_xy_widthStd','ktoa_slope_xy_troStd','ktoa_slope_xy_rVar','ktoa_slope_xy_sVar','ktoa_slope_xy_priFreq','ktoa_slope_xy_meanFreq','ktoa_slope_xy_Entrop','ktoa_slope_xy_meanPower',
'htok_slope_xy_pMeanDist','htok_slope_xy_pMedDist','htok_slope_xy_pStdDist','htok_slope_xy_tMeanDist','htok_slope_xy_tMedDist','htok_slope_xy_tStdDist','htok_slope_xy_rMean','htok_slope_xy_sMean','htok_slope_xy_peakMean','htok_slope_xy_widthMean','htok_slope_xy_troMean','htok_slope_xy_rMed','htok_slope_xy_sMed','htok_slope_xy_peakMed','htok_slope_xy_widthMed','htok_slope_xy_troMed','htok_slope_xy_rMin','htok_slope_xy_rMax','htok_slope_xy_rAmp','htok_slope_xy_rStd','htok_slope_xy_sStd','htok_slope_xy_peakStd','htok_slope_xy_widthStd','htok_slope_xy_troStd','htok_slope_xy_rVar','htok_slope_xy_sVar','htok_slope_xy_priFreq','htok_slope_xy_meanFreq','htok_slope_xy_Entrop','htok_slope_xy_meanPower',
'lws_height_xy_pMeanDist','lws_height_xy_pMedDist','lws_height_xy_pStdDist','lws_height_xy_tMeanDist','lws_height_xy_tMedDist','lws_height_xy_tStdDist','lws_height_xy_rMean','lws_height_xy_sMean','lws_height_xy_peakMean','lws_height_xy_widthMean','lws_height_xy_troMean','lws_height_xy_rMed','lws_height_xy_sMed','lws_height_xy_peakMed','lws_height_xy_widthMed','lws_height_xy_troMed','lws_height_xy_rMin','lws_height_xy_rMax','lws_height_xy_rAmp','lws_height_xy_rStd','lws_height_xy_sStd','lws_height_xy_peakStd','lws_height_xy_widthStd','lws_height_xy_troStd','lws_height_xy_rVar','lws_height_xy_sVar','lws_height_xy_priFreq','lws_height_xy_meanFreq','lws_height_xy_Entrop','lws_height_xy_meanPower',
'rws_height_xy_pMeanDist','rws_height_xy_pMedDist','rws_height_xy_pStdDist','rws_height_xy_tMeanDist','rws_height_xy_tMedDist','rws_height_xy_tStdDist','rws_height_xy_rMean','rws_height_xy_sMean','rws_height_xy_peakMean','rws_height_xy_widthMean','rws_height_xy_troMean','rws_height_xy_rMed','rws_height_xy_sMed','rws_height_xy_peakMed','rws_height_xy_widthMed','rws_height_xy_troMed','rws_height_xy_rMin','rws_height_xy_rMax','rws_height_xy_rAmp','rws_height_xy_rStd','rws_height_xy_sStd','rws_height_xy_peakStd','rws_height_xy_widthStd','rws_height_xy_troStd','rws_height_xy_rVar','rws_height_xy_sVar','rws_height_xy_priFreq','rws_height_xy_meanFreq','rws_height_xy_Entrop','rws_height_xy_meanPower',
'ls_angle_xy_pMeanDist','ls_angle_xy_pMedDist','ls_angle_xy_pStdDist','ls_angle_xy_tMeanDist','ls_angle_xy_tMedDist','ls_angle_xy_tStdDist','ls_angle_xy_rMean','ls_angle_xy_sMean','ls_angle_xy_peakMean','ls_angle_xy_widthMean','ls_angle_xy_troMean','ls_angle_xy_rMed','ls_angle_xy_sMed','ls_angle_xy_peakMed','ls_angle_xy_widthMed','ls_angle_xy_troMed','ls_angle_xy_rMin','ls_angle_xy_rMax','ls_angle_xy_rAmp','ls_angle_xy_rStd','ls_angle_xy_sStd','ls_angle_xy_peakStd','ls_angle_xy_widthStd','ls_angle_xy_troStd','ls_angle_xy_rVar','ls_angle_xy_sVar','ls_angle_xy_priFreq','ls_angle_xy_meanFreq','ls_angle_xy_Entrop','ls_angle_xy_meanPower',
'rs_angle_xy_pMeanDist','rs_angle_xy_pMedDist','rs_angle_xy_pStdDist','rs_angle_xy_tMeanDist','rs_angle_xy_tMedDist','rs_angle_xy_tStdDist','rs_angle_xy_rMean','rs_angle_xy_sMean','rs_angle_xy_peakMean','rs_angle_xy_widthMean','rs_angle_xy_troMean','rs_angle_xy_rMed','rs_angle_xy_sMed','rs_angle_xy_peakMed','rs_angle_xy_widthMed','rs_angle_xy_troMed','rs_angle_xy_rMin','rs_angle_xy_rMax','rs_angle_xy_rAmp','rs_angle_xy_rStd','rs_angle_xy_sStd','rs_angle_xy_peakStd','rs_angle_xy_widthStd','rs_angle_xy_troStd','rs_angle_xy_rVar','rs_angle_xy_sVar','rs_angle_xy_priFreq','rs_angle_xy_meanFreq','rs_angle_xy_Entrop','rs_angle_xy_meanPower',
'ls_adisp_xy_pMeanDist','ls_adisp_xy_pMedDist','ls_adisp_xy_pStdDist','ls_adisp_xy_tMeanDist','ls_adisp_xy_tMedDist','ls_adisp_xy_tStdDist','ls_adisp_xy_rMean','ls_adisp_xy_sMean','ls_adisp_xy_peakMean','ls_adisp_xy_widthMean','ls_adisp_xy_troMean','ls_adisp_xy_rMed','ls_adisp_xy_sMed','ls_adisp_xy_peakMed','ls_adisp_xy_widthMed','ls_adisp_xy_troMed','ls_adisp_xy_rMin','ls_adisp_xy_rMax','ls_adisp_xy_rAmp','ls_adisp_xy_rStd','ls_adisp_xy_sStd','ls_adisp_xy_peakStd','ls_adisp_xy_widthStd','ls_adisp_xy_troStd','ls_adisp_xy_rVar','ls_adisp_xy_sVar','ls_adisp_xy_priFreq','ls_adisp_xy_meanFreq','ls_adisp_xy_Entrop','ls_adisp_xy_meanPower',
'rs_adisp_xy_pMeanDist','rs_adisp_xy_pMedDist','rs_adisp_xy_pStdDist','rs_adisp_xy_tMeanDist','rs_adisp_xy_tMedDist','rs_adisp_xy_tStdDist','rs_adisp_xy_rMean','rs_adisp_xy_sMean','rs_adisp_xy_peakMean','rs_adisp_xy_widthMean','rs_adisp_xy_troMean','rs_adisp_xy_rMed','rs_adisp_xy_sMed','rs_adisp_xy_peakMed','rs_adisp_xy_widthMed','rs_adisp_xy_troMed','rs_adisp_xy_rMin','rs_adisp_xy_rMax','rs_adisp_xy_rAmp','rs_adisp_xy_rStd','rs_adisp_xy_sStd','rs_adisp_xy_peakStd','rs_adisp_xy_widthStd','rs_adisp_xy_troStd','rs_adisp_xy_rVar','rs_adisp_xy_sVar','rs_adisp_xy_priFreq','rs_adisp_xy_meanFreq','rs_adisp_xy_Entrop','rs_adisp_xy_meanPower',
'lk_angle_xy_pMeanDist','lk_angle_xy_pMedDist','lk_angle_xy_pStdDist','lk_angle_xy_tMeanDist','lk_angle_xy_tMedDist','lk_angle_xy_tStdDist','lk_angle_xy_rMean','lk_angle_xy_sMean','lk_angle_xy_peakMean','lk_angle_xy_widthMean','lk_angle_xy_troMean','lk_angle_xy_rMed','lk_angle_xy_sMed','lk_angle_xy_peakMed','lk_angle_xy_widthMed','lk_angle_xy_troMed','lk_angle_xy_rMin','lk_angle_xy_rMax','lk_angle_xy_rAmp','lk_angle_xy_rStd','lk_angle_xy_sStd','lk_angle_xy_peakStd','lk_angle_xy_widthStd','lk_angle_xy_troStd','lk_angle_xy_rVar','lk_angle_xy_sVar','lk_angle_xy_priFreq','lk_angle_xy_meanFreq','lk_angle_xy_Entrop','lk_angle_xy_meanPower',
'rk_angle_xy_pMeanDist','rk_angle_xy_pMedDist','rk_angle_xy_pStdDist','rk_angle_xy_tMeanDist','rk_angle_xy_tMedDist','rk_angle_xy_tStdDist','rk_angle_xy_rMean','rk_angle_xy_sMean','rk_angle_xy_peakMean','rk_angle_xy_widthMean','rk_angle_xy_troMean','rk_angle_xy_rMed','rk_angle_xy_sMed','rk_angle_xy_peakMed','rk_angle_xy_widthMed','rk_angle_xy_troMed','rk_angle_xy_rMin','rk_angle_xy_rMax','rk_angle_xy_rAmp','rk_angle_xy_rStd','rk_angle_xy_sStd','rk_angle_xy_peakStd','rk_angle_xy_widthStd','rk_angle_xy_troStd','rk_angle_xy_rVar','rk_angle_xy_sVar','rk_angle_xy_priFreq','rk_angle_xy_meanFreq','rk_angle_xy_Entrop','rk_angle_xy_meanPower',
'lk_adisp_xy_pMeanDist','lk_adisp_xy_pMedDist','lk_adisp_xy_pStdDist','lk_adisp_xy_tMeanDist','lk_adisp_xy_tMedDist','lk_adisp_xy_tStdDist','lk_adisp_xy_rMean','lk_adisp_xy_sMean','lk_adisp_xy_peakMean','lk_adisp_xy_widthMean','lk_adisp_xy_troMean','lk_adisp_xy_rMed','lk_adisp_xy_sMed','lk_adisp_xy_peakMed','lk_adisp_xy_widthMed','lk_adisp_xy_troMed','lk_adisp_xy_rMin','lk_adisp_xy_rMax','lk_adisp_xy_rAmp','lk_adisp_xy_rStd','lk_adisp_xy_sStd','lk_adisp_xy_peakStd','lk_adisp_xy_widthStd','lk_adisp_xy_troStd','lk_adisp_xy_rVar','lk_adisp_xy_sVar','lk_adisp_xy_priFreq','lk_adisp_xy_meanFreq','lk_adisp_xy_Entrop','lk_adisp_xy_meanPower',
'rk_adisp_xy_pMeanDist','rk_adisp_xy_pMedDist','rk_adisp_xy_pStdDist','rk_adisp_xy_tMeanDist','rk_adisp_xy_tMedDist','rk_adisp_xy_tStdDist','rk_adisp_xy_rMean','rk_adisp_xy_sMean','rk_adisp_xy_peakMean','rk_adisp_xy_widthMean','rk_adisp_xy_troMean','rk_adisp_xy_rMed','rk_adisp_xy_sMed','rk_adisp_xy_peakMed','rk_adisp_xy_widthMed','rk_adisp_xy_troMed','rk_adisp_xy_rMin','rk_adisp_xy_rMax','rk_adisp_xy_rAmp','rk_adisp_xy_rStd','rk_adisp_xy_sStd','rk_adisp_xy_peakStd','rk_adisp_xy_widthStd','rk_adisp_xy_troStd','rk_adisp_xy_rVar','rk_adisp_xy_sVar','rk_adisp_xy_priFreq','rk_adisp_xy_meanFreq','rk_adisp_xy_Entrop','rk_adisp_xy_meanPower',
'la_xdisp_xy_pMeanDist','la_xdisp_xy_pMedDist','la_xdisp_xy_pStdDist','la_xdisp_xy_tMeanDist','la_xdisp_xy_tMedDist','la_xdisp_xy_tStdDist','la_xdisp_xy_rMean','la_xdisp_xy_sMean','la_xdisp_xy_peakMean','la_xdisp_xy_widthMean','la_xdisp_xy_troMean','la_xdisp_xy_rMed','la_xdisp_xy_sMed','la_xdisp_xy_peakMed','la_xdisp_xy_widthMed','la_xdisp_xy_troMed','la_xdisp_xy_rMin','la_xdisp_xy_rMax','la_xdisp_xy_rAmp','la_xdisp_xy_rStd','la_xdisp_xy_sStd','la_xdisp_xy_peakStd','la_xdisp_xy_widthStd','la_xdisp_xy_troStd','la_xdisp_xy_rVar','la_xdisp_xy_sVar','la_xdisp_xy_priFreq','la_xdisp_xy_meanFreq','la_xdisp_xy_Entrop','la_xdisp_xy_meanPower',
'ra_xdisp_xy_pMeanDist','ra_xdisp_xy_pMedDist','ra_xdisp_xy_pStdDist','ra_xdisp_xy_tMeanDist','ra_xdisp_xy_tMedDist','ra_xdisp_xy_tStdDist','ra_xdisp_xy_rMean','ra_xdisp_xy_sMean','ra_xdisp_xy_peakMean','ra_xdisp_xy_widthMean','ra_xdisp_xy_troMean','ra_xdisp_xy_rMed','ra_xdisp_xy_sMed','ra_xdisp_xy_peakMed','ra_xdisp_xy_widthMed','ra_xdisp_xy_troMed','ra_xdisp_xy_rMin','ra_xdisp_xy_rMax','ra_xdisp_xy_rAmp','ra_xdisp_xy_rStd','ra_xdisp_xy_sStd','ra_xdisp_xy_peakStd','ra_xdisp_xy_widthStd','ra_xdisp_xy_troStd','ra_xdisp_xy_rVar','ra_xdisp_xy_sVar','ra_xdisp_xy_priFreq','ra_xdisp_xy_meanFreq','ra_xdisp_xy_Entrop','ra_xdisp_xy_meanPower',
'lktla_height_xy_pMeanDist','lktla_height_xy_pMedDist','lktla_height_xy_pStdDist','lktla_height_xy_tMeanDist','lktla_height_xy_tMedDist','lktla_height_xy_tStdDist','lktla_height_xy_rMean','lktla_height_xy_sMean','lktla_height_xy_peakMean','lktla_height_xy_widthMean','lktla_height_xy_troMean','lktla_height_xy_rMed','lktla_height_xy_sMed','lktla_height_xy_peakMed','lktla_height_xy_widthMed','lktla_height_xy_troMed','lktla_height_xy_rMin','lktla_height_xy_rMax','lktla_height_xy_rAmp','lktla_height_xy_rStd','lktla_height_xy_sStd','lktla_height_xy_peakStd','lktla_height_xy_widthStd','lktla_height_xy_troStd','lktla_height_xy_rVar','lktla_height_xy_sVar','lktla_height_xy_priFreq','lktla_height_xy_meanFreq','lktla_height_xy_Entrop','lktla_height_xy_meanPower',
'rktra_height_xy_pMeanDist','rktra_height_xy_pMedDist','rktra_height_xy_pStdDist','rktra_height_xy_tMeanDist','rktra_height_xy_tMedDist','rktra_height_xy_tStdDist','rktra_height_xy_rMean','rktra_height_xy_sMean','rktra_height_xy_peakMean','rktra_height_xy_widthMean','rktra_height_xy_troMean','rktra_height_xy_rMed','rktra_height_xy_sMed','rktra_height_xy_peakMed','rktra_height_xy_widthMed','rktra_height_xy_troMed','rktra_height_xy_rMin','rktra_height_xy_rMax','rktra_height_xy_rAmp','rktra_height_xy_rStd','rktra_height_xy_sStd','rktra_height_xy_peakStd','rktra_height_xy_widthStd','rktra_height_xy_troStd','rktra_height_xy_rVar','rktra_height_xy_sVar','rktra_height_xy_priFreq','rktra_height_xy_meanFreq','rktra_height_xy_Entrop','rktra_height_xy_meanPower',
'hand_sym_fr_pMeanDist','hand_sym_fr_pMedDist','hand_sym_fr_pStdDist','hand_sym_fr_tMeanDist','hand_sym_fr_tMedDist','hand_sym_fr_tStdDist','hand_sym_fr_rMean','hand_sym_fr_sMean','hand_sym_fr_peakMean','hand_sym_fr_widthMean','hand_sym_fr_troMean','hand_sym_fr_rMed','hand_sym_fr_sMed','hand_sym_fr_peakMed','hand_sym_fr_widthMed','hand_sym_fr_troMed','hand_sym_fr_rMin','hand_sym_fr_rMax','hand_sym_fr_rAmp','hand_sym_fr_rStd','hand_sym_fr_sStd','hand_sym_fr_peakStd','hand_sym_fr_widthStd','hand_sym_fr_troStd','hand_sym_fr_rVar','hand_sym_fr_sVar','hand_sym_fr_priFreq','hand_sym_fr_meanFreq','hand_sym_fr_Entrop','hand_sym_fr_meanPower',
'letlw_height_xy_pMeanDist','letlw_height_xy_pMedDist','letlw_height_xy_pStdDist','letlw_height_xy_tMeanDist','letlw_height_xy_tMedDist','letlw_height_xy_tStdDist','letlw_height_xy_rMean','letlw_height_xy_sMean','letlw_height_xy_peakMean','letlw_height_xy_widthMean','letlw_height_xy_troMean','letlw_height_xy_rMed','letlw_height_xy_sMed','letlw_height_xy_peakMed','letlw_height_xy_widthMed','letlw_height_xy_troMed','letlw_height_xy_rMin','letlw_height_xy_rMax','letlw_height_xy_rAmp','letlw_height_xy_rStd','letlw_height_xy_sStd','letlw_height_xy_peakStd','letlw_height_xy_widthStd','letlw_height_xy_troStd','letlw_height_xy_rVar','letlw_height_xy_sVar','letlw_height_xy_priFreq','letlw_height_xy_meanFreq','letlw_height_xy_Entrop','letlw_height_xy_meanPower',
'retrw_height_xy_pMeanDist','retrw_height_xy_pMedDist','retrw_height_xy_pStdDist','retrw_height_xy_tMeanDist','retrw_height_xy_tMedDist','retrw_height_xy_tStdDist','retrw_height_xy_rMean','retrw_height_xy_sMean','retrw_height_xy_peakMean','retrw_height_xy_widthMean','retrw_height_xy_troMean','retrw_height_xy_rMed','retrw_height_xy_sMed','retrw_height_xy_peakMed','retrw_height_xy_widthMed','retrw_height_xy_troMed','retrw_height_xy_rMin','retrw_height_xy_rMax','retrw_height_xy_rAmp','retrw_height_xy_rStd','retrw_height_xy_sStd','retrw_height_xy_peakStd','retrw_height_xy_widthStd','retrw_height_xy_troStd','retrw_height_xy_rVar','retrw_height_xy_sVar','retrw_height_xy_priFreq','retrw_height_xy_meanFreq','retrw_height_xy_Entrop','retrw_height_xy_meanPower',
'le_angle_xy_pMeanDist','le_angle_xy_pMedDist','le_angle_xy_pStdDist','le_angle_xy_tMeanDist','le_angle_xy_tMedDist','le_angle_xy_tStdDist','le_angle_xy_rMean','le_angle_xy_sMean','le_angle_xy_peakMean','le_angle_xy_widthMean','le_angle_xy_troMean','le_angle_xy_rMed','le_angle_xy_sMed','le_angle_xy_peakMed','le_angle_xy_widthMed','le_angle_xy_troMed','le_angle_xy_rMin','le_angle_xy_rMax','le_angle_xy_rAmp','le_angle_xy_rStd','le_angle_xy_sStd','le_angle_xy_peakStd','le_angle_xy_widthStd','le_angle_xy_troStd','le_angle_xy_rVar','le_angle_xy_sVar','le_angle_xy_priFreq','le_angle_xy_meanFreq','le_angle_xy_Entrop','le_angle_xy_meanPower',
're_angle_xy_pMeanDist','re_angle_xy_pMedDist','re_angle_xy_pStdDist','re_angle_xy_tMeanDist','re_angle_xy_tMedDist','re_angle_xy_tStdDist','re_angle_xy_rMean','re_angle_xy_sMean','re_angle_xy_peakMean','re_angle_xy_widthMean','re_angle_xy_troMean','re_angle_xy_rMed','re_angle_xy_sMed','re_angle_xy_peakMed','re_angle_xy_widthMed','re_angle_xy_troMed','re_angle_xy_rMin','re_angle_xy_rMax','re_angle_xy_rAmp','re_angle_xy_rStd','re_angle_xy_sStd','re_angle_xy_peakStd','re_angle_xy_widthStd','re_angle_xy_troStd','re_angle_xy_rVar','re_angle_xy_sVar','re_angle_xy_priFreq','re_angle_xy_meanFreq','re_angle_xy_Entrop','re_angle_xy_meanPower',
'le_adisp_xy_pMeanDist','le_adisp_xy_pMedDist','le_adisp_xy_pStdDist','le_adisp_xy_tMeanDist','le_adisp_xy_tMedDist','le_adisp_xy_tStdDist','le_adisp_xy_rMean','le_adisp_xy_sMean','le_adisp_xy_peakMean','le_adisp_xy_widthMean','le_adisp_xy_troMean','le_adisp_xy_rMed','le_adisp_xy_sMed','le_adisp_xy_peakMed','le_adisp_xy_widthMed','le_adisp_xy_troMed','le_adisp_xy_rMin','le_adisp_xy_rMax','le_adisp_xy_rAmp','le_adisp_xy_rStd','le_adisp_xy_sStd','le_adisp_xy_peakStd','le_adisp_xy_widthStd','le_adisp_xy_troStd','le_adisp_xy_rVar','le_adisp_xy_sVar','le_adisp_xy_priFreq','le_adisp_xy_meanFreq','le_adisp_xy_Entrop','le_adisp_xy_meanPower',
're_adisp_xy_pMeanDist','re_adisp_xy_pMedDist','re_adisp_xy_pStdDist','re_adisp_xy_tMeanDist','re_adisp_xy_tMedDist','re_adisp_xy_tStdDist','re_adisp_xy_rMean','re_adisp_xy_sMean','re_adisp_xy_peakMean','re_adisp_xy_widthMean','re_adisp_xy_troMean','re_adisp_xy_rMed','re_adisp_xy_sMed','re_adisp_xy_peakMed','re_adisp_xy_widthMed','re_adisp_xy_troMed','re_adisp_xy_rMin','re_adisp_xy_rMax','re_adisp_xy_rAmp','re_adisp_xy_rStd','re_adisp_xy_sStd','re_adisp_xy_peakStd','re_adisp_xy_widthStd','re_adisp_xy_troStd','re_adisp_xy_rVar','re_adisp_xy_sVar','re_adisp_xy_priFreq','re_adisp_xy_meanFreq','re_adisp_xy_Entrop','re_adisp_xy_meanPower',
'lhtla_height_xy_pMeanDist','lhtla_height_xy_pMedDist','lhtla_height_xy_pStdDist','lhtla_height_xy_tMeanDist','lhtla_height_xy_tMedDist','lhtla_height_xy_tStdDist','lhtla_height_xy_rMean','lhtla_height_xy_sMean','lhtla_height_xy_peakMean','lhtla_height_xy_widthMean','lhtla_height_xy_troMean','lhtla_height_xy_rMed','lhtla_height_xy_sMed','lhtla_height_xy_peakMed','lhtla_height_xy_widthMed','lhtla_height_xy_troMed','lhtla_height_xy_rMin','lhtla_height_xy_rMax','lhtla_height_xy_rAmp','lhtla_height_xy_rStd','lhtla_height_xy_sStd','lhtla_height_xy_peakStd','lhtla_height_xy_widthStd','lhtla_height_xy_troStd','lhtla_height_xy_rVar','lhtla_height_xy_sVar','lhtla_height_xy_priFreq','lhtla_height_xy_meanFreq','lhtla_height_xy_Entrop','lhtla_height_xy_meanPower',
'rhtra_height_xy_pMeanDist','rhtra_height_xy_pMedDist','rhtra_height_xy_pStdDist','rhtra_height_xy_tMeanDist','rhtra_height_xy_tMedDist','rhtra_height_xy_tStdDist','rhtra_height_xy_rMean','rhtra_height_xy_sMean','rhtra_height_xy_peakMean','rhtra_height_xy_widthMean','rhtra_height_xy_troMean','rhtra_height_xy_rMed','rhtra_height_xy_sMed','rhtra_height_xy_peakMed','rhtra_height_xy_widthMed','rhtra_height_xy_troMed','rhtra_height_xy_rMin','rhtra_height_xy_rMax','rhtra_height_xy_rAmp','rhtra_height_xy_rStd','rhtra_height_xy_sStd','rhtra_height_xy_peakStd','rhtra_height_xy_widthStd','rhtra_height_xy_troStd','rhtra_height_xy_rVar','rhtra_height_xy_sVar','rhtra_height_xy_priFreq','rhtra_height_xy_meanFreq','rhtra_height_xy_Entrop','rhtra_height_xy_meanPower',
'a_dist_xy_pMeanDist','a_dist_xy_pMedDist','a_dist_xy_pStdDist','a_dist_xy_tMeanDist','a_dist_xy_tMedDist','a_dist_xy_tStdDist','a_dist_xy_rMean','a_dist_xy_sMean','a_dist_xy_peakMean','a_dist_xy_widthMean','a_dist_xy_troMean','a_dist_xy_rMed','a_dist_xy_sMed','a_dist_xy_peakMed','a_dist_xy_widthMed','a_dist_xy_troMed','a_dist_xy_rMin','a_dist_xy_rMax','a_dist_xy_rAmp','a_dist_xy_rStd','a_dist_xy_sStd','a_dist_xy_peakStd','a_dist_xy_widthStd','a_dist_xy_troStd','a_dist_xy_rVar','a_dist_xy_sVar','a_dist_xy_priFreq','a_dist_xy_meanFreq','a_dist_xy_Entrop','a_dist_xy_meanPower',
'la_ydisp_xy_pMeanDist','la_ydisp_xy_pMedDist','la_ydisp_xy_pStdDist','la_ydisp_xy_tMeanDist','la_ydisp_xy_tMedDist','la_ydisp_xy_tStdDist','la_ydisp_xy_rMean','la_ydisp_xy_sMean','la_ydisp_xy_peakMean','la_ydisp_xy_widthMean','la_ydisp_xy_troMean','la_ydisp_xy_rMed','la_ydisp_xy_sMed','la_ydisp_xy_peakMed','la_ydisp_xy_widthMed','la_ydisp_xy_troMed','la_ydisp_xy_rMin','la_ydisp_xy_rMax','la_ydisp_xy_rAmp','la_ydisp_xy_rStd','la_ydisp_xy_sStd','la_ydisp_xy_peakStd','la_ydisp_xy_widthStd','la_ydisp_xy_troStd','la_ydisp_xy_rVar','la_ydisp_xy_sVar','la_ydisp_xy_priFreq','la_ydisp_xy_meanFreq','la_ydisp_xy_Entrop','la_ydisp_xy_meanPower',
'ra_ydisp_xy_pMeanDist','ra_ydisp_xy_pMedDist','ra_ydisp_xy_pStdDist','ra_ydisp_xy_tMeanDist','ra_ydisp_xy_tMedDist','ra_ydisp_xy_tStdDist','ra_ydisp_xy_rMean','ra_ydisp_xy_sMean','ra_ydisp_xy_peakMean','ra_ydisp_xy_widthMean','ra_ydisp_xy_troMean','ra_ydisp_xy_rMed','ra_ydisp_xy_sMed','ra_ydisp_xy_peakMed','ra_ydisp_xy_widthMed','ra_ydisp_xy_troMed','ra_ydisp_xy_rMin','ra_ydisp_xy_rMax','ra_ydisp_xy_rAmp','ra_ydisp_xy_rStd','ra_ydisp_xy_sStd','ra_ydisp_xy_peakStd','ra_ydisp_xy_widthStd','ra_ydisp_xy_troStd','ra_ydisp_xy_rVar','ra_ydisp_xy_sVar','ra_ydisp_xy_priFreq','ra_ydisp_xy_meanFreq','ra_ydisp_xy_Entrop','ra_ydisp_xy_meanPower',
'atos_dist_xy_pMeanDist','atos_dist_xy_pMedDist','atos_dist_xy_pStdDist','atos_dist_xy_tMeanDist','atos_dist_xy_tMedDist','atos_dist_xy_tStdDist','atos_dist_xy_rMean','atos_dist_xy_sMean','atos_dist_xy_peakMean','atos_dist_xy_widthMean','atos_dist_xy_troMean','atos_dist_xy_rMed','atos_dist_xy_sMed','atos_dist_xy_peakMed','atos_dist_xy_widthMed','atos_dist_xy_troMed','atos_dist_xy_rMin','atos_dist_xy_rMax','atos_dist_xy_rAmp','atos_dist_xy_rStd','atos_dist_xy_sStd','atos_dist_xy_peakStd','atos_dist_xy_widthStd','atos_dist_xy_troStd','atos_dist_xy_rVar','atos_dist_xy_sVar','atos_dist_xy_priFreq','atos_dist_xy_meanFreq','atos_dist_xy_Entrop','atos_dist_xy_meanPower',
'lws_ydist_yz_pMeanDist','lws_ydist_yz_pMedDist','lws_ydist_yz_pStdDist','lws_ydist_yz_tMeanDist','lws_ydist_yz_tMedDist','lws_ydist_yz_tStdDist','lws_ydist_yz_rMean','lws_ydist_yz_sMean','lws_ydist_yz_peakMean','lws_ydist_yz_widthMean','lws_ydist_yz_troMean','lws_ydist_yz_rMed','lws_ydist_yz_sMed','lws_ydist_yz_peakMed','lws_ydist_yz_widthMed','lws_ydist_yz_troMed','lws_ydist_yz_rMin','lws_ydist_yz_rMax','lws_ydist_yz_rAmp','lws_ydist_yz_rStd','lws_ydist_yz_sStd','lws_ydist_yz_peakStd','lws_ydist_yz_widthStd','lws_ydist_yz_troStd','lws_ydist_yz_rVar','lws_ydist_yz_sVar','lws_ydist_yz_priFreq','lws_ydist_yz_meanFreq','lws_ydist_yz_Entrop','lws_ydist_yz_meanPower',
'rws_ydist_yz_pMeanDist','rws_ydist_yz_pMedDist','rws_ydist_yz_pStdDist','rws_ydist_yz_tMeanDist','rws_ydist_yz_tMedDist','rws_ydist_yz_tStdDist','rws_ydist_yz_rMean','rws_ydist_yz_sMean','rws_ydist_yz_peakMean','rws_ydist_yz_widthMean','rws_ydist_yz_troMean','rws_ydist_yz_rMed','rws_ydist_yz_sMed','rws_ydist_yz_peakMed','rws_ydist_yz_widthMed','rws_ydist_yz_troMed','rws_ydist_yz_rMin','rws_ydist_yz_rMax','rws_ydist_yz_rAmp','rws_ydist_yz_rStd','rws_ydist_yz_sStd','rws_ydist_yz_peakStd','rws_ydist_yz_widthStd','rws_ydist_yz_troStd','rws_ydist_yz_rVar','rws_ydist_yz_sVar','rws_ydist_yz_priFreq','rws_ydist_yz_meanFreq','rws_ydist_yz_Entrop','rws_ydist_yz_meanPower',
'lws_xdist_yz_pMeanDist','lws_xdist_yz_pMedDist','lws_xdist_yz_pStdDist','lws_xdist_yz_tMeanDist','lws_xdist_yz_tMedDist','lws_xdist_yz_tStdDist','lws_xdist_yz_rMean','lws_xdist_yz_sMean','lws_xdist_yz_peakMean','lws_xdist_yz_widthMean','lws_xdist_yz_troMean','lws_xdist_yz_rMed','lws_xdist_yz_sMed','lws_xdist_yz_peakMed','lws_xdist_yz_widthMed','lws_xdist_yz_troMed','lws_xdist_yz_rMin','lws_xdist_yz_rMax','lws_xdist_yz_rAmp','lws_xdist_yz_rStd','lws_xdist_yz_sStd','lws_xdist_yz_peakStd','lws_xdist_yz_widthStd','lws_xdist_yz_troStd','lws_xdist_yz_rVar','lws_xdist_yz_sVar','lws_xdist_yz_priFreq','lws_xdist_yz_meanFreq','lws_xdist_yz_Entrop','lws_xdist_yz_meanPower',
'rws_xdist_yz_pMeanDist','rws_xdist_yz_pMedDist','rws_xdist_yz_pStdDist','rws_xdist_yz_tMeanDist','rws_xdist_yz_tMedDist','rws_xdist_yz_tStdDist','rws_xdist_yz_rMean','rws_xdist_yz_sMean','rws_xdist_yz_peakMean','rws_xdist_yz_widthMean','rws_xdist_yz_troMean','rws_xdist_yz_rMed','rws_xdist_yz_sMed','rws_xdist_yz_peakMed','rws_xdist_yz_widthMed','rws_xdist_yz_troMed','rws_xdist_yz_rMin','rws_xdist_yz_rMax','rws_xdist_yz_rAmp','rws_xdist_yz_rStd','rws_xdist_yz_sStd','rws_xdist_yz_peakStd','rws_xdist_yz_widthStd','rws_xdist_yz_troStd','rws_xdist_yz_rVar','rws_xdist_yz_sVar','rws_xdist_yz_priFreq','rws_xdist_yz_meanFreq','rws_xdist_yz_Entrop','rws_xdist_yz_meanPower',
'n_angle_yz_pMeanDist','n_angle_yz_pMedDist','n_angle_yz_pStdDist','n_angle_yz_tMeanDist','n_angle_yz_tMedDist','n_angle_yz_tStdDist','n_angle_yz_rMean','n_angle_yz_sMean','n_angle_yz_peakMean','n_angle_yz_widthMean','n_angle_yz_troMean','n_angle_yz_rMed','n_angle_yz_sMed','n_angle_yz_peakMed','n_angle_yz_widthMed','n_angle_yz_troMed','n_angle_yz_rMin','n_angle_yz_rMax','n_angle_yz_rAmp','n_angle_yz_rStd','n_angle_yz_sStd','n_angle_yz_peakStd','n_angle_yz_widthStd','n_angle_yz_troStd','n_angle_yz_rVar','n_angle_yz_sVar','n_angle_yz_priFreq','n_angle_yz_meanFreq','n_angle_yz_Entrop','n_angle_yz_meanPower',
'lara_dist_yz_pMeanDist','lara_dist_yz_pMedDist','lara_dist_yz_pStdDist','lara_dist_yz_tMeanDist','lara_dist_yz_tMedDist','lara_dist_yz_tStdDist','lara_dist_yz_rMean','lara_dist_yz_sMean','lara_dist_yz_peakMean','lara_dist_yz_widthMean','lara_dist_yz_troMean','lara_dist_yz_rMed','lara_dist_yz_sMed','lara_dist_yz_peakMed','lara_dist_yz_widthMed','lara_dist_yz_troMed','lara_dist_yz_rMin','lara_dist_yz_rMax','lara_dist_yz_rAmp','lara_dist_yz_rStd','lara_dist_yz_sStd','lara_dist_yz_peakStd','lara_dist_yz_widthStd','lara_dist_yz_troStd','lara_dist_yz_rVar','lara_dist_yz_sVar','lara_dist_yz_priFreq','lara_dist_yz_meanFreq','lara_dist_yz_Entrop','lara_dist_yz_meanPower',
'la_angle_yz_pMeanDist','la_angle_yz_pMedDist','la_angle_yz_pStdDist','la_angle_yz_tMeanDist','la_angle_yz_tMedDist','la_angle_yz_tStdDist','la_angle_yz_rMean','la_angle_yz_sMean','la_angle_yz_peakMean','la_angle_yz_widthMean','la_angle_yz_troMean','la_angle_yz_rMed','la_angle_yz_sMed','la_angle_yz_peakMed','la_angle_yz_widthMed','la_angle_yz_troMed','la_angle_yz_rMin','la_angle_yz_rMax','la_angle_yz_rAmp','la_angle_yz_rStd','la_angle_yz_sStd','la_angle_yz_peakStd','la_angle_yz_widthStd','la_angle_yz_troStd','la_angle_yz_rVar','la_angle_yz_sVar','la_angle_yz_priFreq','la_angle_yz_meanFreq','la_angle_yz_Entrop','la_angle_yz_meanPower',
'ra_angle_yz_pMeanDist','ra_angle_yz_pMedDist','ra_angle_yz_pStdDist','ra_angle_yz_tMeanDist','ra_angle_yz_tMedDist','ra_angle_yz_tStdDist','ra_angle_yz_rMean','ra_angle_yz_sMean','ra_angle_yz_peakMean','ra_angle_yz_widthMean','ra_angle_yz_troMean','ra_angle_yz_rMed','ra_angle_yz_sMed','ra_angle_yz_peakMed','ra_angle_yz_widthMed','ra_angle_yz_troMed','ra_angle_yz_rMin','ra_angle_yz_rMax','ra_angle_yz_rAmp','ra_angle_yz_rStd','ra_angle_yz_sStd','ra_angle_yz_peakStd','ra_angle_yz_widthStd','ra_angle_yz_troStd','ra_angle_yz_rVar','ra_angle_yz_sVar','ra_angle_yz_priFreq','ra_angle_yz_meanFreq','ra_angle_yz_Entrop','ra_angle_yz_meanPower',
'lk_angle_yz_pMeanDist','lk_angle_yz_pMedDist','lk_angle_yz_pStdDist','lk_angle_yz_tMeanDist','lk_angle_yz_tMedDist','lk_angle_yz_tStdDist','lk_angle_yz_rMean','lk_angle_yz_sMean','lk_angle_yz_peakMean','lk_angle_yz_widthMean','lk_angle_yz_troMean','lk_angle_yz_rMed','lk_angle_yz_sMed','lk_angle_yz_peakMed','lk_angle_yz_widthMed','lk_angle_yz_troMed','lk_angle_yz_rMin','lk_angle_yz_rMax','lk_angle_yz_rAmp','lk_angle_yz_rStd','lk_angle_yz_sStd','lk_angle_yz_peakStd','lk_angle_yz_widthStd','lk_angle_yz_troStd','lk_angle_yz_rVar','lk_angle_yz_sVar','lk_angle_yz_priFreq','lk_angle_yz_meanFreq','lk_angle_yz_Entrop','lk_angle_yz_meanPower']

# generate the feats from a single data of input
def generateFeatsForOne(df, type): 
    # for each row, we are adding new columns, by fetching infor from columns
    data = []
    for i, row in df.iterrows(): 
        # joints 
        la = (row['marker_27_x'], row['marker_27_y'], row['marker_27_z']) # left ankle
        ra = (row['marker_28_x'], row['marker_28_y'], row['marker_28_z']) # right ankle

        lk = (row['marker_25_x'], row['marker_25_y'], row['marker_25_z']) # left knee
        rk = (row['marker_26_x'], row['marker_26_y'], row['marker_26_z']) # right knee

        lh = (row['marker_23_x'], row['marker_23_y'], row['marker_23_z']) # left hip
        rh = (row['marker_24_x'], row['marker_24_y'], row['marker_24_z']) # right hip

        ls = (row['marker_11_x'], row['marker_11_y'], row['marker_11_z']) # left shoulder
        rs = (row['marker_12_x'], row['marker_12_y'], row['marker_12_z']) # right shoulder

        le = (row['marker_13_x'], row['marker_13_y'], row['marker_13_z']) # left elbow
        re = (row['marker_14_x'], row['marker_14_y'], row['marker_14_z']) # right elbow

        lw = (row['marker_15_x'], row['marker_15_y'], row['marker_15_z']) # left wrist
        rw = (row['marker_16_x'], row['marker_16_y'], row['marker_16_z']) # right wrist

        h = (row['marker_0_x'], row['marker_0_y'], row['marker_0_z']) # head

        # features for row
        feats = []
        feats.append(row['subject']) # subject
        feats.append(row['move']) # move
        feats.append(row['timestamp']) # timestamp 
        
        ###########################
        ## FRONTAL (x-y plane) VIEW
        if type == 'videofront':
            # xz slopes 
            feats.append(fe.slope(lh[0], lh[2], rh[0], rh[2])) #hip slope
            feats.append(fe.slope(ls[0], ls[2], rs[0], rs[2])) #shoulder slope

            # ratios 
            feats.append(lw[0] / noDivideByZero(ls[0])) # left wrist to shoulder x position 
            feats.append(rw[0] / noDivideByZero(rs[0])) # right wrist to shoulder x position 

            # xy slopes & parallel
            feats.append(fe.slope(lh[0], lh[1], rh[0], rh[1]))
            feats.append(fe.slope(ls[0], ls[1], rs[0], rs[1]) / noDivideByZero(fe.slope(lh[0], lh[1], rh[0], rh[1]))) # shoulder to hip slopes
            feats.append(fe.slope(lk[0], lk[1], rk[0], rk[1]) / noDivideByZero(fe.slope(la[0], la[1], ra[0], ra[1]))) # knee to ankle slopes
            feats.append(fe.slope(lh[0], lh[1], rh[0], rh[1]) / noDivideByZero(fe.slope(lk[0], lk[1], rk[0], rk[1]))) # hip to knee slopes

            # heights 
            #df.at[i, 'lws_height_xy'] = 
            feats.append(lw[1] / noDivideByZero(ls[1])) # left wrist to shoulder height 
            #df.at[i, 'rws_height_xy'] = 
            feats.append(rw[1] / noDivideByZero(rs[1])) # right wrist to shoulder height 

            # abduction/adduction of shoulders
            # df.at[i, 'ls_angle_xy'] = 
            ls_angle_xy = fe.angleBetweenVectors(le[0], le[1], ls[0], ls[1], lh[0], lh[1]) 
            feats.append(ls_angle_xy) # left elbow-shoulder-hip angle
            #df.at[i, 'rs_angle_xy'] = 
            rs_angle_xy = fe.angleBetweenVectors(re[0], re[1], rs[0], rs[1], rh[0], rh[1])
            feats.append(rs_angle_xy) # right elbow-shoulder-hip angle

            # angular displacement of shoulder 
            # verify the same subject & move, and value exists at prev time stamp
            if i > 0 and df.at[i - 1, 'subject'] == df.at[i, 'subject'] and df.at[i - 1, 'move'] == df.at[i, 'move']:
                #df.at[i, 'ls_adisp_xy'] = df.at[i, 'ls_angle_xy'] - df.at[i - 1, 'ls_angle_xy'])
                feats.append(ls_angle_xy - data[i - 1][newcols.index('ls_angle_xy')])
                #df.at[i, 'rs_adisp_xy'] = df.at[i, 'rs_angle_xy'] - df.at[i - 1, 'rs_angle_xy']
                feats.append(rs_angle_xy - data[i - 1][newcols.index('rs_angle_xy')])
            else:
                feats.append(0.0)
                feats.append(0.0)

            # knee angles
            lk_angle_xy = fe.angleBetweenVectors(lh[0], lh[1], lk[0], lk[1], la[0], la[1])
            feats.append(lk_angle_xy) # left hip-knee-ankle angle
            rk_angle_xy = fe.angleBetweenVectors(rh[0], rh[1], rk[0], rk[1], ra[0], ra[1])
            feats.append(rk_angle_xy) # right hip-knee-ankle angle

            # angular displacement of knee
            # verify the same subject & move, and value exists at prev time stamp
            if i > 0 and df.at[i - 1, 'subject'] == df.at[i, 'subject'] and df.at[i - 1, 'move'] == df.at[i, 'move']:
                feats.append(lk_angle_xy - data[i - 1][newcols.index('lk_angle_xy')])
                feats.append(rk_angle_xy - data[i - 1][newcols.index('rk_angle_xy')])
            else:
                feats.append(0.0)
                feats.append(0.0)

            # horizontal displacement of ankle 
            # verify the same subject & move, and value exists at prev time stamp
            if i > 0 and df.at[i - 1, 'subject'] == df.at[i, 'subject'] and df.at[i - 1, 'move'] == df.at[i, 'move']:
                feats.append(la[0] - df.at[i - 1, 'marker_27_x' ])
                feats.append(ra[0] - df.at[i - 1, 'marker_28_x'])
            else:
                feats.append(0.0)
                feats.append(0.0)

            # knee height to ankle height ratio 
            feats.append(lk[1] / noDivideByZero(la[1]))
            feats.append(rk[1] / noDivideByZero(ra[1]))

            # hand1:hand2 
            feats.append(fe.distance1d(lw[0], h[0]) / noDivideByZero(fe.distance1d(rw[0], h[0]))) # distance of hand from center 
        
            # elbow height to wrist ratio 
            feats.append(le[1] / noDivideByZero(lw[1]))
            feats.append(re[1] / noDivideByZero(rw[1]))

            # elbow angles 
            le_angle_xy = fe.angleBetweenVectors(lw[0], lw[1], le[0], le[1], ls[0], ls[1])
            feats.append(le_angle_xy)
            re_angle_xy = fe.angleBetweenVectors(rw[0], rw[1], re[0], re[1], rs[0], rs[1])
            feats.append(re_angle_xy)

            # angular displacement of elbow angle 
            # verify the same subject & move, and value exists at prev time stamp
            if i > 0 and df.at[i - 1, 'subject'] == df.at[i, 'subject'] and df.at[i - 1, 'move'] == df.at[i, 'move']:
                feats.append(le_angle_xy - data[i - 1][newcols.index('le_angle_xy')])
                feats.append(re_angle_xy - data[i - 1][newcols.index('re_angle_xy')])
            else:
                feats.append(0.0)
                feats.append(0.0)

            # hip height to ankle height ratio 
            feats.append(lh[1] / noDivideByZero(la[1]))
            feats.append(rh[1] / noDivideByZero(ra[1]))

            # distances between la and ra
            feats.append(fe.distance2d(la[0], la[1], ra[0], ra[1]))

            # backwards displacement of ankle (Y)
            # verify the same subject & move, and value exists at prev time stamp
            if i > 0 and df.at[i - 1, 'subject'] == df.at[i, 'subject'] and df.at[i - 1, 'move'] == df.at[i, 'move']:
                feats.append(la[1] - df.at[i - 1, 'marker_27_y' ])
                feats.append(ra[1] - df.at[i - 1, 'marker_28_y'])
            else:
                feats.append(0.0)
                feats.append(0.0)

            # distance between ankles to distance between shoulders 
            feats.append(fe.distance2d(la[0], la[1], ra[0], ra[1]) / noDivideByZero(fe.distance2d(ls[0], ls[1], rs[0], rs[1])))

        ###########################
        ## LATERAL (y-z plane) VIEW
        ## need to edit this so that they are PROXY values 
        ## i.e. where relevant, if the lateral video exists, we use the lateral instead of the frontal - since we consider it more accurate
        ## for depth
        elif type == 'videolateral':
            # wirst to shoulder height ratio
            feats.append(lw[1] / noDivideByZero(ls[1])) # lws_dist_y
            feats.append(rw[1] / noDivideByZero(rs[1])) # rws_dist_y

            # wrist to shoulder distance ratio 
            feats.append(lw[0] / noDivideByZero(ls[0])) # lws_dist_x
            feats.append(rw[0] / noDivideByZero(rs[0])) # rws_dist_x

            # neck angle 
            feats.append(fe.angleReltoPerp(h[0], h[1], ls[0], ls[1])) # use left shoulder for convience

            # distance between la and ra
            feats.append(fe.distance1d(la[0], ra[0])) # ankle_depth_diff

            # Knee to ankle angle (perp. to floor)
            feats.append(fe.angleReltoPerp(lk[0], lk[1], la[0], la[1])) # la_center
            feats.append(fe.angleReltoPerp(rk[0], rk[1], ra[0], ra[1])) # ra_center

            # Knee Angle 
            feats.append(fe.angleBetweenVectors(lh[0], lh[1], lk[0], lk[1], la[0], la[1])) # lk_angle_yz
        data.append(feats)
    if type == 'videofront':
        return pd.DataFrame(data, columns = newcols)
    elif type == 'videolateral':
        return pd.DataFrame(data, columns = newcols_lat)

# for a smoothed time series, we now extract the stats 
# also verify that it is time stationary, if not make the correction
def extractStats(feats, df):
    data = []
    for f in feats: 
        # data of interest
        raw = df[f]
        y = raw.ewm(span=30).mean() # EMA of 30 window - NEED TO MAKE A CHANGE W/O - on other 
        yF = np.fft.rfft(df[f]) # FFT

        # get peaks & peak widths, throughs
        p, _ = find_peaks(y) 
        true_peaks = argrelextrema(y.values, np.greater)[0]
        true_trs = argrelextrema(y.values, np.less)[0]
        data_mean = np.nanmean(y)
        ploc, peaks = selectIfGreater(true_peaks, y[true_peaks].array, data_mean)
        tloc, troughs = selectIfLess(true_trs, y[true_trs].array, data_mean)
        width = peak_widths(y, p, rel_height=0.75)

        # average peak distance
        pdist = getDistance(ploc)
        tdist = getDistance(tloc)
        data.append(np.nanmean(pdist))
        # median peak distance
        data.append(np.nanmedian(pdist))
        # std peak distance
        data.append(np.nanstd(pdist))
        # average trough distance
        data.append(np.nanmean(tdist))
        # median trough distance
        data.append(np.nanmedian(tdist))
        # std trough distance
        data.append(np.nanstd(tdist))

        # mean
        data.append(np.nanmean(raw))
        data.append(data_mean)
        # average peak values 
        data.append(np.nanmean(peaks))
        # average peak width 
        data.append(np.nanmean(width))
        # average trough values 
        data.append(np.nanmean(troughs))
        
        # median
        data.append(np.nanmedian(raw))
        data.append(np.nanmedian(y))
        # meadian peak values 
        data.append(np.nanmedian(peaks))
        # median peak width 
        data.append(np.nanmedian(width))
        # median low values 
        data.append(np.nanmedian(troughs))

        # min
        minval = np.nanmin(raw)
        data.append(minval)
        # max
        maxval = np.nanmax(raw)
        data.append(maxval)
        # max peak amplitude 
        data.append(maxval - minval)
        
        # std 
        rStd = np.nanstd(raw)
        data.append(rStd)
        yStd = np.nanstd(y)
        data.append(yStd)
        # std of peak values 
        data.append(np.nanstd(peaks))
        # std peak width 
        data.append(np.nanstd(width))
        # std of through values
        data.append(np.nanstd(troughs))

        # variance
        data.append(rStd**2)
        data.append(yStd**2)

        # principal freq 
        N = len(yF)
        n = np.arange(N)
        samp = 1/0.06
        T = N/samp
        freq = n/T 
        ps = np.abs(yF)**2

        indx_max = np.argmax(np.abs(yF))
        if indx_max > 0 and indx_max <= len(freq):
            data.append(freq[indx_max])
        else:
            data.append(0) # default fail value 

        # mean frequency 
        meanFreq = np.nansum([freq[i] * ps[i]  for i in range(N)]) / N
        data.append(meanFreq)

        # entropy
        pp = ps / np.nansum(ps)
        S = (-1 / np.log2(N)) * np.nansum(pp * np.log2(pp))
        data.append(S)

        # mean power 
        data.append(np.nanmean(ps))
        
    return data


def main():
    df = []
    # repeat the process for every subject
    for s in range(len(subject)):
        #for i in range(len(videoFront)):
            # FRONT 
            path = root2 + str(s) + videoFront[8] + "_Mocap.csv"
            if os.path.exists(path) and os.path.isfile(path):
                data = pd.read_csv(path)
                stats = []
                if len(data) > 0:
                    data = generateFeatsForOne(data, 'videofront') # generates the features for one subject
                    stats = extractStats(newcols[3:], data)

                # LATERAL
                statsL = []
                path2 = root2 + str(s) + videoLateral[8] + "_Mocap.csv"

                if os.path.exists(path2) and os.path.isfile(path2):
                    dataL = pd.read_csv(path2)
                    dataL = generateFeatsForOne(dataL, 'videolateral') # generates the features for one subject
                    statsL = extractStats(newcols_lat[3:], dataL)
                else:
                    statsL = [0 for _ in range(len(newcols_lat[3:]))] # make empty otherwise

                # append our data
                if len(stats) != 0:
                    row = [subject[s], moves[8]]
                    row = row + stats + statsL
                    if len(row) == len(features_all):
                        df.append(row)

    dataFin = pd.DataFrame(df, columns=features_all)
    #export the data
    dataFin.to_csv(base + "RTP_Features.csv")

if __name__ == "__main__":
    main()