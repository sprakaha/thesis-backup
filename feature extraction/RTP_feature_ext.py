from blazepose_est import runModelWithVideo
from helpers import selectIfGreater, selectIfLess, getDistance, noDivideByZero
import pandas as pd
import feature_ext_array as fe
import numpy as np
import os
from scipy.signal import find_peaks, peak_widths,  argrelextrema, savgol_filter
from path_vars import root, root2, base 

# Script extracts pose markers at a high samplin rate ~15 Hz
# Upon extraction, the features are directly calculated
# Then, they are adjusted with AR / MA / FFT
# Finally, the peak (max) amplitude, trough (min) amplitude, frequency of wave, mean value, and domain frequency
# are extracted
# In the end for each inputted video, should return about 200 static features 

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

vF = ['Brush_knee_twist_step_left_ft_front_Trim',
 'Brush_knee_twist_step_right_ft_front_Trim',
 'Golden_rooster_left_ft_front_Trim',
 'Golden_rooster_right_ft_front_Trim',
 'Grasp_the_sparrow_tail_left_ft_front_Trim',
 'Grasp_the_sparrow_tail_right_ft_front_Trim',
 'Push_Left_ft_front_Trim',
 'Push_Right_ft_front_Trim',
 'Raising_the_power_front_Trim',
 'Wave_hands_like_clouds_both_hand_front_Trim',
 'Wave_hands_like_clouds_left_hand_front_Trim',
 'Wave_hands_like_clouds_right_hand_front_Trim']
 
vL = ['Brush_knee_twist_step_left_ft_lateral_Trim',
 'Brush_knee_twist_step_right_ft_lateral_Trim',
 'Golden_rooster_left_ft_lateral_Trim',
 'Golden_rooster_right_ft_lateral_Trim',
 'Grasp_the_sparrow_tail_left_ft_lateral_Trim',
 'Grasp_the_sparrow_tail_right_ft_lateral_Trim',
 'Push_Left_ft_lateral_Trim',
 'Push_Right_ft_lateral_Trim',
 'Raising_the_power_lateral_Trim',
 'Wave_hands_like_clouds_both_hand_lateral_Trim',
 'Wave_hands_like_clouds_left_hand_lateral_Trim',
 'Wave_hands_like_clouds_right_hand_lateral_Trim'] 

column_vals = ['subject', 'move', 'timestamp', 'marker_0_x', 'marker_0_y', 'marker_0_z', 'marker_11_x', 'marker_11_y', 'marker_11_z', 'marker_12_x', 'marker_12_y', 'marker_12_z', 'marker_13_x',  'marker_13_y',  'marker_13_z', 
    'marker_14_x', 'marker_14_y', 'marker_14_z','marker_15_x', 'marker_15_y', 'marker_15_z', 'marker_16_x', 'marker_16_y', 'marker_16_z', 'marker_23_x', 'marker_23_y', 'marker_23_z', 'marker_24_x', 'marker_24_y', 'marker_24_z', 
    'marker_25_x', 'marker_25_y', 'marker_25_z','marker_26_x', 'marker_26_y', 'marker_26_z', 'marker_27_x', 'marker_27_y', 'marker_27_z', 'marker_28_x', 'marker_28_y', 'marker_28_z']

# frontal & lateral
front = ['atos_dist_xy', 'wdist_sag_xy',  'lk_angle_xy',  'rk_angle_xy', 
'lktla_height_xy', 'rktra_height_xy', 'kdist_xy', 'kdist_sag_xy', 'lw_xy', 'rw_xy', 'h_slope_xy',
'htk_slope_xy', 'lh_y_xy', 'rh_y_xy', 'lws_xdiff_xy', 'rws_xdiff_xy','rw_ydisp_xy', 
'lw_ydisp_xy', 'lws_ydiff_xy', 'rws_ydiff_xy']
lateral = ['rwy_yz', 'rhy_yz', 'rk_angle_yz', 'rktorw_yz', 'rws_ydist_yz', 'rws_ydiff_yz', 'n_angle_yz']
newcols = ['subject', 'move', 'timestamp'] + front + lateral 
features_all = ['subject', 'move', 'atos_dist_xy_med', 'atos_dist_xy_var', 'atos_dist_xy_std',
'wdist_sag_xy_std', 'wdist_sag_xy_var',  'wdist_sag_xy_meanIV', 
'lk_angle_xy_med',  'lk_angle_xy_var',  'lk_angle_xy_std',
'rk_angle_xy_med',  'rk_angle_xy_var',  'rk_angle_xy_std',
'lktla_height_xy_medAtmax', 'lktla_height_xy_medAtmin',
'rktra_height_xy_medAtmax', 'rktra_height_xy_medAtmin',
'kdist_xy_std', 'kdist_xy_var', 'kdist_xy_range',
'kdist_sag_xy_std', 'kdist_sag_xy_var', 'kdist_sag_xy_range',
'lw_xy_vel', 'lw_xy_acc', 'lw_xy_jerk', 'lw_xy_std', 'lw_xy_var', 'lw_xy_mean', 'lw_xy_med',
'rw_xy_vel', 'rw_xy_acc', 'rw_xy_jerk', 'rw_xy_std', 'rw_xy_var', 'rw_xy_mean', 'rw_xy_med',
'h_slope_xy_std', 'h_slope_xy_var', 'h_slope_xy_mean', 'h_slope_xy_med',
'htk_slope_xy_std', 'htk_slope_xy_var', 'htk_slope_xy_mean', 'htk_slope_xy_med',
'lh_y_xy_corr', 'rh_y_xy_corr',
'lws_xdiff_xy_medIV', 'lws_xdiff_xy_medIVdisp',
'rws_xdiff_xy_medIV', 'rws_xdiff_xy_medIVdisp',
'rw_ydisp_xy_medIV', 'rw_ydisp_xy_medIVdisp',
'lw_ydisp_xy_medIV', 'lw_ydisp_xy_medIVdisp',
'lws_ydiff_xy_medIV', 'lws_ydiff_xy_medIVdisp',
'rws_ydiff_xy_medIV', 'rws_ydiff_xy_medIVdisp',   
'rwy_yz_vel', 'rwy_yz_acc', 'rwy_yz_jerk', 'rwy_yz_synch',
'rhy_yz_corr',
'rk_angle_yz_meanAmp', 'rk_angle_yz_meanPer',
'rktorw_yz_std', 'rktorw_yz_var', 'rktorw_yz_mean', 'rktorw_yz_med',
'rws_ydist_yz_medIV', 'rws_ydist_yz_medIVdisp', 'rws_ydist_yz_medIVw', 'rws_ydist_yz_medIVwdisp', 'rws_ydist_yz_medIVk', 'rws_ydist_yz_medIVkdisp',
'rws_ydiff_yz_medIV', 'rws_ydiff_yz_medIVdisp', 'rws_ydiff_yz_medIVw', 'rws_ydiff_yz_medIVwdisp', 'rws_ydiff_yz_medIVk', 'rws_ydiff_yz_medIVkdisp',
'n_angle_yz_std', 'n_angle_yz_var', 'n_angle_yz_mean', 'n_angle_yz_med'
]

def generateDataForOne(v, i, mov_ind): #pass videotype & subject
    data = []
    if v == "videolateral":
        view = vL
    elif v == "videofront": 
        view = vF

    # exceptions on 8, 9, 14, 26, 28, 29, 31, 32; should be caught by if
    path = root + subject[i] + '\\' + view[mov_ind] + '.mp4'
    if os.path.exists(path) and os.path.isfile(path):
        if len(data) == 0: 
            # for nyquist want, 0.066
            data = runModelWithVideo(subject[i], view[mov_ind], path, 0.066)
    data = pd.DataFrame(data, columns = column_vals)
    return data


# generate the feats from a single data of input
def generateFeatsForOne(df, df2, type): 
    # for each row, we are adding new columns, by fetching infor from columns
    data = []
    data.append(df['subject'].iloc[:].values) # subject
    data.append(df['move'].iloc[:].values) # move
    data.append(df['timestamp'].iloc[:].values) # timestamp 
    # data is smoothed with savitsky golay prior to data extraction 
    df = df.apply(lambda x: savgol_filter(x, 6, 3) if x.name in df.columns[4:] else x, axis = 1)
    entryNo = df.shape[0]
    ###########################
    ## FRONTAL (x-y plane) VIEW
    if type == 'videofront' or type == 'both':
        # joints 
        la = (df['marker_27_x'].iloc[:].values, df['marker_27_y'].iloc[:].values, df['marker_27_z'].iloc[:].values) # left ankle
        ra = (df['marker_28_x'].iloc[:].values, df['marker_28_y'].iloc[:].values, df['marker_28_z'].iloc[:].values) # right ankle

        lk = (df['marker_25_x'].iloc[:].values, df['marker_25_y'].iloc[:].values, df['marker_25_z'].iloc[:].values) # left knee
        rk = (df['marker_26_x'].iloc[:].values, df['marker_26_y'].iloc[:].values, df['marker_26_z'].iloc[:].values) # right knee

        lh = (df['marker_23_x'].iloc[:].values, df['marker_23_y'].iloc[:].values, df['marker_23_z'].iloc[:].values) # left hip
        rh = (df['marker_24_x'].iloc[:].values, df['marker_24_y'].iloc[:].values, df['marker_24_z'].iloc[:].values) # right hip

        ls = (df['marker_11_x'].iloc[:].values, df['marker_11_y'].iloc[:].values, df['marker_11_z'].iloc[:].values) # left shoulder
        rs = (df['marker_12_x'].iloc[:].values, df['marker_12_y'].iloc[:].values, df['marker_12_z'].iloc[:].values) # right shoulder

        lw = (df['marker_15_x'].iloc[:].values, df['marker_15_y'].iloc[:].values, df['marker_15_z'].iloc[:].values) # left wrist
        rw = (df['marker_16_x'].iloc[:].values, df['marker_16_y'].iloc[:].values, df['marker_16_z'].iloc[:].values) # right wrist

        h = (df['marker_0_x'].iloc[:].values, df['marker_0_y'].iloc[:].values, df['marker_0_z'].iloc[:].values) # head

        # saggital midpoint 
        mid = (ls[0] + rs[0])/2

        # distance between ankles to distance between shoulders 
        data.append(fe.ratio(fe.distance1d(la[0], ra[0]), fe.distance1d(ls[0], rs[0]))) # atos_dist_xy

        # distance of hand from center
        data.append(fe.ratio(fe.distance1d(lw[0], mid), fe.distance1d(rw[0], mid))) # wdist_sag_xy 

        # knee angles
        lk_angle_xy = fe.angleBetweenVectors(lh[0], lh[1], lk[0], lk[1], la[0], la[1]) # left hip-knee-ankle angle
        data.append(lk_angle_xy) # lk_angle_xy
        rk_angle_xy = fe.angleBetweenVectors(rh[0], rh[1], rk[0], rk[1], ra[0], ra[1]) # right hip-knee-ankle angle
        data.append(rk_angle_xy) # rk_angle_xy

        # knee height to ankle height 
        data.append(fe.ratio(lk[1], la[1])) # lktla_height_xy
        data.append(fe.ratio(rk[1], ra[1])) # rktra_height_xy

        # knee dist 
        # distances between lk and rk
        data.append(fe.distance1d(lk[0], rk[0])) # kdist_xy
        data.append(fe.ratio(fe.distance1d(lk[0], mid), fe.distance1d(rk[0], mid))) # kdist_sag_xy

        # wrist vertical movement 
        data.append(lw[1]) # lw_xy
        data.append(rw[1]) # rw_xy

        # xy slopes & parallel
        data.append(fe.slope(lh[0], lh[1], rh[0], rh[1])) # h_slope_xy
        data.append(fe.ratio(fe.slope(lh[0], lh[1], rh[0], rh[1]), fe.slope(lk[0], lk[1], rk[0], rk[1]))) # htk_slope_xy
        
        # pelvis veritcal movement 
        data.append(lh[1]) # lh_y_xy
        data.append(rh[1]) # rh_y_xy 

        # ratios 
        data.append(lw[0] - ls[0]) # lws_xdiff_xy 
        data.append(rw[0] - rs[0]) # rws_xdiff_xy 

        # heights 
        data.append(lw[1] - ls[1]) # lws_ydiff_xy
        data.append(rw[1] - rs[1]) # rws_ydiff_xy

        data.append(fe.ratio(lw[1], ls[1])) # lws_ydist_xy
        data.append(fe.ratio(rw[1], rs[1])) # rws_ydist_xy
    else:
        for _ in range(len(front)): # number of frontal features
            data.append([np.nan for _ in range(entryNo)])

    ###########################
    ## LATERAL (y-z plane) VIEWS
    ## i.e. where relevant, if the lateral video exists, we use otherwise fill NaN
    if type == 'videolateral' or type == 'both':
        # joints 
        ra = (df2['marker_28_x'].iloc[:].values, df2['marker_28_y'].iloc[:].values, df2['marker_28_z'].iloc[:].values) # right ankle
        rk = (df2['marker_26_x'].iloc[:].values, df2['marker_26_y'].iloc[:].values, df2['marker_26_z'].iloc[:].values) # right knee
        rh = (df2['marker_24_x'].iloc[:].values, df2['marker_24_y'].iloc[:].values, df2['marker_24_z'].iloc[:].values) # right hip
        rs = (df2['marker_12_x'].iloc[:].values, df2['marker_12_y'].iloc[:].values, df2['marker_12_z'].iloc[:].values) # right shoulder
        rw = (df2['marker_16_x'].iloc[:].values, df2['marker_16_y'].iloc[:].values, df2['marker_16_z'].iloc[:].values) # right wrist
        h = (df2['marker_0_x'].iloc[:].values, df2['marker_0_y'].iloc[:].values, df2['marker_0_z'].iloc[:].values) # head

        # Positions 
        data.append(rw[1]) # rwy_yz
        data.append(rh[1]) # rhy_yz

        # Knee Angle & Ratio
        data.append(fe.angleBetweenVectors(rh[0], rh[1], rk[0], rk[1], ra[0], ra[1])) # rk_angle_yz
        data.append(fe.ratio(rk[1], rw[1])) # rktorw_yz

        # wirst to shoulder height ratio
        data.append(fe.ratio(rw[1], rs[1])) # rws_ydist_yz

        # wrist to shoulder distance ratio 
        data.append(rw[1] - rs[1]) # rws_ydiff_yz

        # neck angle 
        data.append(fe.angleReltoPerp(h[0], h[1], rs[0], rs[1])) # n_angle_yz
    else:
        for _ in range(len(lateral)): # number of lateral features
            data.append([np.nan for _ in range(entryNo)])

    return data

# get max 
def getPeaks(y):
    # get peaks throughs
    if not np.nan in y:
        true_peaks = argrelextrema(y, np.greater)[0]
        data_mean = np.nanmean(y)
        ploc, peaks = selectIfGreater(true_peaks, y[true_peaks], data_mean)
        return ploc, peaks 
    else:
        return [np.nan], [np.nan] 

# get min 
def getTroughs(y):
    if not np.nan in y:
        true_trs = argrelextrema(y, np.less)[0]
        data_mean = np.nanmean(y)
        tloc, troughs = selectIfLess(true_trs, y[true_trs], data_mean)
        return tloc, troughs
    else:
        return [np.nan], [np.nan]

# get peak width
def getPeakWidth(y):
    if not np.nan in y:
        p, _ = find_peaks(y) 
        width = peak_widths(y, p, rel_height=0.75)
        return width
    else:
        return [np.nan]

# for a smoothed time series, we now extract the stats 
# also verify that it is time stationary, if not make the correction
def extractStats(feats, df):
    data = []
    max_knee_bend_pos, _ = getPeaks(df[feats.index('rk_angle_yz')][:])
    max_wrist_pos, _ = getPeaks(df[feats.index('rw_xy')][:])
    min_wrist_pos, _ = getTroughs(df[feats.index('rw_xy')][:])
    t = np.array(df[2][:])

    for f in feats[3:]: 
        # data of interest
        y = np.array(df[feats.index(f)][:])
        # feature extraction 
        if f == 'atos_dist_xy':
            data.append(np.nanmedian(y)) # median
            data.append(np.nanvar(y)) # variance
            data.append(np.nanstd(y)) # std
        elif f == 'wdist_sag_xy':
            data.append(np.nanstd(y)) # std
            data.append(np.nanvar(y)) # variance
            if np.nan not in y:
                _, vals = fe.getInflectionPoint(t, y) # Inflection points
                data.append(np.nanmean(vals)) # average y value of inflection points
            else:
                data.append(np.nan)
        elif f == 'lk_angle_xy' or f =='rk_angle_xy':
            data.append(np.nanmedian(y)) # median 
            data.append(np.nanvar(y)) # variance
            data.append(np.nanstd(y)) # std 
        elif f == 'lktla_height_xy' or f == 'rktra_height_xy':
            if np.nan not in max_wrist_pos: # measure at max hand height 
               data.append(np.nanmedian(y[max_wrist_pos]))
            else:
                data.append(np.nan)
            if np.nan not in min_wrist_pos: # measure at min hand height 
                data.append(np.nanmedian(y[min_wrist_pos]))
            else:
                data.append(np.nan)
        elif f == 'kdist_xy' or f == 'kdist_sag_xy':
            data.append(np.nanstd(y)) # std 
            data.append(np.nanvar(y)) # variance 
            data.append(np.nanmax(y) - np.nanmin(y)) # range
        elif f == 'lw_xy' or f == 'rw_xy':
            # kinematics over entire period
            if np.nan not in y:
                data.append(np.nanmean(fe.velocity(t, y))) # velocity 
                data.append(np.nanmean(fe.acceleration(t, y))) # acceleration
                data.append(np.nanmean(fe.jerk(t, y))) # jerk 
            else:
                data.append(np.nan)
                data.append(np.nan)
                data.append(np.nan)
            data.append(np.nanstd(y)) # std 
            data.append(np.nanvar(y)) # variance 
            data.append(np.nanmean(y)) # mean
            data.append(np.nanmedian(y)) # median
        elif f == 'h_slope_xy' or f == 'htk_slope_xy':
            data.append(np.nanstd(y)) # std 
            data.append(np.nanvar(y)) # variance 
            data.append(np.nanmean(y)) # mean
            data.append(np.nanmedian(y)) # median 
        elif f == 'lh_y_xy':
            # cross correlation
            if np.nan not in y and np.nan not in df[feats.index('lw_xy')][:]:  
                disph = fe.displacement(y)
                dispw = fe.displacement(df[feats.index('lw_xy')][:])
                data.append(np.correlate(disph, dispw)[0])
            else:
                data.append(np.nan)
        elif f == 'rh_y_xy': 
            # cross correlation
            if np.nan not in y and np.nan not in df[feats.index('rw_xy')][:]: 
                disph = fe.displacement(y)
                dispw = fe.displacement(df[feats.index('rw_xy')][:])   
                data.append(np.correlate(disph, dispw)[0])
            else:
                data.append(np.nan)
        elif f == 'lws_xdiff_xy' or f == 'rws_xdiff_xy' or f == 'rw_ydisp_xy' or f == 'lw_ydisp_xy' or f == 'lws_ydiff_xy' or f == 'rws_ydiff_xy':
            # measure at inflection points 
            if np.nan not in y:
                x, y = fe.getInflectionPoint(t, y)
                data.append(np.nanmedian(y))
                data.append(np.nanmedian(fe.displacement(x)))
            else:
                data.append(np.nan)
                data.append(np.nan)
        elif f == 'rwy_yz': 
            # kinematics over entire period
            t2 = np.array([i * 66 for i in range(len(y))])
            if np.nan not in y:
                data.append(np.nanmean(fe.velocity(t2, y))) # velocity 
                data.append(np.nanmean(fe.acceleration(t2, y))) # acceleration
                data.append(np.nanmean(fe.jerk(t2, y))) # jerk 
            else: 
                data.append(np.nan)
                data.append(np.nan)
                data.append(np.nan)
            # synchronicity factor - is max knee bend at highest wrist point
            if np.nan not in max_knee_bend_pos: # measure at max knee bend 
                data.append(np.nanmedian(y[max_knee_bend_pos]))
            else:
                data.append(np.nan)
        elif f == 'rhy_yz': 
            # cross correlation
            if np.nan not in y and np.nan not in df[feats.index('rwy_yz')][:]:
                disph = fe.displacement(y)
                dispw = fe.displacement(df[feats.index('rwy_yz')][:])   
                data.append(np.correlate(disph, dispw)[0])
            else:
                data.append(np.nan)
        elif f == 'rk_angle_yz':
            _, y0 = getPeaks(y) # knee bend at max
            _, y1 = getTroughs(y) # knee bend at min - relative
            data.append(np.nanmean(y0) - np.nanmean(y1)) # average amp
            data.append(np.nanmean(y1) / np.nanmean(y0)) # average percentage 
        elif f == 'rktorw_yz': 
            data.append(np.nanstd(y)) # std 
            data.append(np.nanvar(y)) # variance 
            data.append(np.nanmean(y)) # mean
            data.append(np.nanmedian(y)) # median 
        elif f == 'rws_ydist_yz' or f == 'rws_ydiff_yz':
            # measure at inflection points of y, hand movement, & knee movement 
            t2 = np.array([i * 66 for i in range(len(y))])
            if np.nan not in y:
                x, y = fe.getInflectionPoint(t2, y)
                data.append(np.nanmedian(y))
                data.append(np.nanmedian(fe.displacement(x)))
            else:
                data.append(np.nan)
                data.append(np.nan)
            if np.nan not in df[feats.index('rwy_yz')][:]:
                x, y = fe.getInflectionPoint(t2, df[feats.index('rwy_yz')][:])
                data.append(np.nanmedian(y))
                data.append(np.nanmedian(fe.displacement(x)))
            else:
                data.append(np.nan)
                data.append(np.nan)
            if np.nan not in df[feats.index('rk_angle_yz')][:]:
                x, y = fe.getInflectionPoint(t2, df[feats.index('rk_angle_yz')][:])
                data.append(np.nanmedian(y))
                data.append(np.nanmedian(fe.displacement(x)))
            else:
                data.append(np.nan)
                data.append(np.nan)
        elif f == 'n_angle_yz':
            data.append(np.nanstd(y)) # std 
            data.append(np.nanvar(y)) # variance 
            data.append(np.nanmean(y)) # mean
            data.append(np.nanmedian(y)) # median 

    return data


def main():
    df = []
    # repeat the process for every subject
    for s in range(len(subject)):
            path = root2 + str(s) + videoFront[8] + "_Mocap.csv"
            path2 = root2 + str(s) + videoLateral[8] + "_Mocap.csv"
            type = ''
            if os.path.exists(path) and os.path.isfile(path):
                type = 'videofront'
                data = pd.read_csv(path)
                if os.path.exists(path2) and os.path.isfile(path2):
                    type = 'both'
                    dataL = pd.read_csv(path2)
                else:
                    dataL = None
            elif os.path.exists(path2) and os.path.isfile(path2): 
                type = 'videolateral'
                data = None
                dataL = pd.read_csv(path2)
            else:
                type = 'error'
                print("No data for ", s)
            
            dataT = generateFeatsForOne(data, dataL, type) # generates the features for one subject
            stats = extractStats(newcols, dataT)

            # append our data
            if len(stats) != 0:
                row = [subject[s], moves[8]]
                row = row + stats
                df.append(row)
    ## Export the data
    dataFin = pd.DataFrame(df, columns=features_all)
    dataFin.to_csv(base + "RTP_FeaturesV3.csv", na_rep='NaN')

if __name__ == "__main__":
    main()