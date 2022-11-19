## GUI
The following folder contains the data and static files used to generate an interactive data visualization of a subject's movements. 

The visualization can be run by starting up a live server on a local port to run the raw JS/HTML/CSS without a need for Node, React, etc. The Live Server local development extension in Visual Studio Code was used in the development of this visualization.

### ./
- index.html: General main page formatting and structure.

### public\statuc
- \css
    - styles.css: Cascading style sheet for organizing the aesthetic of the page.
- \js 
    - lookups.js: Used to store the lookup jsons in Javascript for easy access.
    - visualize.js: Script that contains the visualization and parsing logic of creating the animation and displaying the model data.

### \data
- \definitions 
    - lookup_feature.json and lookup_joint.json are JSON versions of the look up tables preesnt in lookup.js. These contain the full explanations of each feature in plain English.
- knn_explanation.json: Output of the knn_wrapper function when run on the best performing kNN to test the classification of subject 1.
- rf_explanation.json:  Output of the rf_wrapper function when run on the best performing kNN to test the classification of subject 1.
- 0Raising_the_power_front_Mocap.csv: Anonymized pose marker time series for subject 1 from a frontal view. 
- 0Raising_the_power_lateral_Mocap.csv: Anonymized pose marker time series for subject 1 from a lateral view.