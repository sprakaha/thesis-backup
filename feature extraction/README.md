## Feature Extraction 
The following folder contains the functions and scripts that were used to convert the video data of the subjects to time series joint position data, which was finally converted into stationary features for each subject. 

### Testing 
- directory that contains files for testing functions during the creation process
- feature_ver.ipynb: tests the feature_ext.py and feature_ext_array.py scritps 
- test_smooth.ipynb: tests moving averages, FFT, and the SG filter

### Functions
- feature_ext.py: contains functions that extract kinematic features for a single data point
- feature_ext_array.py: cointains functions that extract kinematic features for data entered as an array
- helpers.py: contains helper functions that are called by other functions and scripts 
- blazepose_est.py: contains functions that runs the Media Pipe Pose software to extract the joints positions 

### Scripts 
- geneterate_ts.py: script that generates the time series joint positions for each subject
- general_feature_ext.py: script extracts generic features of interest 
- RTP_feature_ext.py: script that extracts Raising the Power specific features of interest
- merge_data.py: srcipt that merges the generated features with the scoring values for each subject

