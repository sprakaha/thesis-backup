# Overview 
The following repository contains the code for a senior thesis focused on the concept of using features related to the biomechanics 
of a Tai Chi movement to develop a transparent and effective ML classification model. The code walks through the process of pose marker extraction, feature development, feature selection, model development, and data visualization that was run on a series of 53 video recordings of the "Raising The Power" Tai Chi exercise.  


# \feature extraction
Contains the code to for the pipeline that ingests a video, leverages Media Pipe to extract pose markers, extracts the biomechanically relevant features, and finally returns a dataset with the feature matched to the appropriate scores.  

# \gui
Contains the code for rendering the data visualization of a subject's motion and providing feedback related to the decision making process of a particular model and the features it's using to make classifications. 

# \model development
Contains the Google Colab notebooks used to render visualizations of feature analysis and the development of the kNN and RF models with the wrapper functions designed to elucidate the math behind the specific implementation of that algorithm. 

# \preprocessing
Contains code for basic label analysis on the 5 scoring mechanisms used to label the initial dataset.

### Note: 
This code has been edited such that the local paths have been removed and replaced with refences to base, root, and root2 variables which were stored in a seperate Python file for security. A set of edited, anonymized mocap data that does not relate to any of the subject's from the study is in the gui\data folder, so one can spin up the visualization.  
