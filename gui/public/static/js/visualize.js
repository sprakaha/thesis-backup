// Global Variables
var dataFin = [];  // Storing all data for the movement
var dataExp;  // Stores the model explanation
var currentFrame = []; // Storing the current Frame
var nextFrame = []; // Storing the next Frame
var frame = 0; // Value for current frame
var stopped = 1; // 0/1 indicating playing or stopped animation
var end_frame = 0; // The final frame
var model = ""; // Type of model, either kNN or RF

// Constants
const delay_amount = 2000; // delay amount - ms
const headJoint = 0; // head joint
const figureScale = 500;  // The scaling factor for our visualizations
const h = 800;  // The height of the visualization
const w = 900;  // The width of the visualization
const jointDict = {0: "Head", 12: "Right Shoulder", 11: "Left Shoulder", 14: "Right Elbow", 13: "Left Elbow", 16: "Right Wrist",
15: "Left Wrist", 24: "Right Hip", 23: "Left Hip", 26: "Right Knee", 25: "Left Knee", 28: "Right Ankle", 27: "Left Ankle"}; // joint name dictionary
const joints = {0: 0, 1: 11, 2:12, 3:13, 4:14, 5:15, 6:16, 7:23, 8:24, 9:25, 10:26, 11:27, 12:28}; // joint mapping 
const skeleton = {1:3, 2:4, 3:5, 4:6, 7:9, 8:10, 9:11, 10:12}; // skeleton mapping
const skeleton2 = {1:2, 2:8, 7:1, 8:7}; // 2nd skeleton mapping 
const skeleton3 = {1:8, 2:7, 7:2, 8:1}; // alternate upper-body to skeleton 2

// Initial Set-Up 
var parent = d3.select("body").select("#viz-container");
var svg = parent.append("svg")
                .attr("width", w)
                .attr("height", h);
                
var details = d3.select("#textbox");

// Look Up Dictionaries
const jointLookUp = JSON.parse(joint_lookup);
const featLookUp = JSON.parse(feat_lookup)

// Format to 2 decimal placeds
function round2(num) {
  return (Math.round(num * 100) / 100).toFixed(2);
}

// Look up the relevant data for the marker in the timestamp
function lookUp(marker, exp) {
  // Get the features related to this pose marker 
  var all_features = jointLookUp[marker];

  // Format general information
  var data = "Predicted Score: " + exp["score"] + "\n";

  if (model == "kn") {
    // Set to hold features used by neighbors 
    var kFeats = new Set();
    // Iterate over all the neighbors
    var neighs = Object.keys(exp["results"]);
    for (let i = 0; i < neighs.length; i++) {
      var count = 0;
      var rel_feats = exp["results"][neighs[i]]["feat_rank"];
      var rel_dist = exp["results"][neighs[i]]["feat_list"];
      // Look if any of the relevant features are at this point
      for (let j = 0; j < all_features.length ; j++) {
        // Do not want to display too many at a joint
        if (count < 8) {
          var feat = all_features[j];
          if (rel_feats.includes(feat)) {
            count++;
            data += "Neighbor " + (i + 1) + ": "
            + feat + " at distance: " + round2(rel_dist[j]) + "\n" + "Assesses the " + featLookUp[feat]["description"] + "\n";
          }
        }
      }
    }
    // Display definitions of the relevant features
    var kFeats = Array.from(kFeats);
    for  (let k = 0; k < kFeats.length; k++) {
      data += "Feature: " + kFeats[k] + "\n" 
      + featLookUp[kFeats[k]]["description"] + "\n";
    }
  } else if (model = "rf") {
    // Look if any of the relevant features are at this point 
    for (let i = 0; i < all_features.length ; i++) {
      var feat = all_features[i];
      if (feat in exp) {
        data += "\nFeature: " + feat + "\n" 
        + featLookUp[feat]["description"] + "\n"
        + "Class 5 | Mean Value: " + round2(exp[feat]["5mean"]) + ", Std: " + round2(exp[feat]["5std"]) + "\n"
        + "Class 4 | Mean Value: " + round2(exp[feat]["4mean"]) + ", Std: " + round2(exp[feat]["4std"]) + "\n"
        + "Class 3 | Mean Value: " + round2(exp[feat]["3mean"]) + ", Std: " + round2(exp[feat]["3std"]) + "\n";
      }
    }
  }
  return data;
}

// Drawing function for animation
// Based of work found here: https://omid.al/posts/2016-08-23-MocapVis-D3/
function draw(p, index) {
  // Scale the data
  currentFrame = p.map(function(d) {
    return {
      m: d.m,
      x: d.x * figureScale,
      y: d.y * figureScale,
      z: d.z * figureScale
    };
  });

  // Clear befor drawing
  svg.selectAll("*").remove();

  // Joints
  svg.selectAll("circle.joints")
    .data(currentFrame)
    .enter()
    .append("circle")
    .attr("class", "update")
    .attr("cx", function(d) {
        return d.x;
    })
    .attr("cy", function(d) {
        return d.y;
    })
    .attr("r", function(d) {
        if (d.m == headJoint)
            return 8;
        else
            return 4;
    })
    .attr("fill", '#555555')
    .on("mouseover", function(d, i) { mousemove(i); })
    .on("mouseout", mousemoveout());

  // Body Segments 
  svg.selectAll("line.bones")
    .data(currentFrame)
    .enter()
    .append("line")
    .attr("stroke", "#555555")
    .attr("stroke-width", 2)
    .attr("x1", function(d, j) {
      if (j != 0 && j != 5 && j != 6 && j != 11 && j != 12) { return d.x;}
    })
    .attr("x2", function(d, j) {
      if (j != 0 && j != 5 && j != 6 && j != 11 && j != 12) { 
        return currentFrame[skeleton[j]].x; }
    })
    .attr("y1", function(d, j) {
      if (j != 0 && j != 5 && j != 6 && j != 11 && j != 12) { return d.y;}
    })
    .attr("y2", function(d, j) {
      if (j != 0 && j != 5 && j != 6 && j != 11 && j != 12) { 
        return currentFrame[skeleton[j]].y;
      }
    });

  // Additional Body Segments
  svg.selectAll("line.bones0")
    .data(currentFrame)
    .enter()
    .append("line")
    .attr("stroke", "#555555")
    .attr("stroke-width", 2)
    .attr("x1", function(d, j) {
      if (j == 1 || j == 2 || j == 7 || j == 8) { return d.x;}
    })
    .attr("x2", function(d, j) {
      if (j == 1 || j == 2 || j == 7 || j == 8) { 
        return currentFrame[skeleton2[j]].x; }
    })
    .attr("y1", function(d, j) {
      if (j == 1 || j == 2 || j == 7 || j == 8) { return d.y;}
    })
    .attr("y2", function(d, j) {
      if (j == 1 || j == 2 || j == 7 || j == 8) { 
        return currentFrame[skeleton2[j]].y;
      }
    });

  function mousemove(ind) {
   // Append text
   const results = lookUp(ind.m, dataExp);
   details.style("border", "1px black solid").text(results); 
  }

  function mousemoveout() {
    // Clear text and images
    details.selectAll("*").remove();
  }
}

// Event Handling
const myForm2 = document.getElementById("myForm2");
const csvFile = document.getElementById("csvFile");
const jsonFile = document.getElementById("jsonFile");
const mySlider = document.getElementById("slider-id"); 

myForm2.addEventListener("submit", function (e) {
  e.preventDefault();
  const input = csvFile.files[0];
  const reasons = jsonFile.files[0];
  
  const reader = new FileReader();
  const reader2 = new FileReader();

  reader.onload = function (e) {
    // Read in mocap data from csv as array of objects 
    // Objects is N x M, where N is # of frames, M # of joints
    const text = e.target.result;
    const data = d3.csvParse(text);

    // Iterate over the data entries
    for (let d=0; d < data.length; d++) {
      if (+data[d].iteration == 0) {
        const details = {"Subj": "", "Movement": ""};
        details["Subj"] = data[d].subject;
        details["Movement"] = data[d].move;
      }
      
      var positions = [];
      for (let i = 0; i < Object.keys(joints).length; i++){
        var joint = {m: 0, x: 0, y: 0, z: 0};
        joint.m = joints[i];
        joint.x = +data[d]["marker_" + String(joints[i]) +"_x"];
        joint.y = +data[d]["marker_" + String(joints[i]) +"_y"];
        joint.z = +data[d]["marker_" + String(joints[i]) +"_z"];
        positions.push(joint);
      }
      dataFin.push(positions);
    }
    end_frame = dataFin.length;
    mySlider.setAttribute("max", end_frame - 1);
    // Show the first frame
    draw(dataFin[0], 0);
  };
  reader2.onload = function (e) {
    // Read in explanations from json
    const text = e.target.result;
    dataExp = JSON.parse(text);
    model = reasons.name.substring(0, 2);
  }
  if (input != undefined) {
    reader.readAsText(input);
  } else {
    alert("Enter motion capture data");
  }
  if (reasons != undefined) {
    reader2.readAsText(reasons);
  }
});

mySlider.addEventListener("change", function(e) {
  var val = mySlider.value;
  if (end_frame == 0) {
    console.log("Data required for visualization.");
  } else if (val > 0 && val < end_frame) {
    draw(dataFin[val], val);
  } else {
    console.log("Invalid selection.")
  }
});