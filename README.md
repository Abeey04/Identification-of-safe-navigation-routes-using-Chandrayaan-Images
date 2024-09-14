# Identification of Safe Navigation routes using Chandrayaan 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Bharatiya%20Antariksh%20Hackathon-2024-yellowgreen.svg)](https://isro.hack2skill.com/2024/)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Scientific Stops](#scientific-stops)
- [Pathfinding](#pathfinding)
- [Visualization](#visualization)
- [Images](#images)
- [Contributors](#contributors)

---

## Introduction
Welcome to the **Radicals Lunar Exploration Project**, developed for the [Bharatiya Antariksh Hackathon 2024](https://isro.hack2skill.com/2024/). Our mission is to explore the lunar South Pole using advanced image processing, crater and boulder detection, and pathfinding algorithms. With the help of cutting-edge technologies such as **YOLOv5** and the **Ant Colony Optimization (ACO)** system, we aim to assist future moon missions in optimizing traversal routes while avoiding obstacles.

---

## Features
- 🌑 **Georeferenced OHRC Imagery**: Reprojects Orbiter High-Resolution Camera (OHRC) images for better precision.
- 🟡 **Crater & Boulder Detection**: Parallel processing through **YOLOv5** and traditional computer vision techniques like Hough Transforms.
- 🐜 **Ant Colony Optimization**: Innovative ACO-based pathfinding to traverse lunar surfaces efficiently, avoiding craters and boulders.
- 🛰️ **Simulated Rover Path Visualization**: Pathfinding and rover movement simulation using **Unity3D** on Digital Terrain Model (DTM) data.

---

## Technologies Used
- **Python**: Used for georeferencing, detection, and pathfinding.
- **GDAL**: For georeferencing OHRC images.
- **YOLOv5**: For crater and boulder detection.
- **Ant Colony Optimization (ACO)**: To optimize pathfinding.
- **Unity3D**: Simulating rover movement on the lunar surface.
- **QGIS**: For terrain data extraction.
  
---

## Usage

### Georeferencing & Image Calibration
1. OHRC images are **georeferenced** using **GDAL** and reprojected to the **Lunar South Polar Stereographic CRS**. The calibrated images are clipped to focus on landing sites.
2. After preprocessing (normalization, denoising), images are fed into **two parallel detection pipelines**: YOLOv5 and traditional computer vision methods.

### Crater & Boulder Detection
We implemented **two approaches** for detection:
- **Approach A**: Parallel processing using YOLOv5 and traditional methods (Hough Transform, Blob Detection).
- **Approach B**: Separate YOLOv5 and traditional CV models for flexibility between accuracy and speed.

---

## Scientific Stops
During the mission, we have a total of **10 scientific stops**, with significant analysis at each stop:
1. **Initial Landing Site Characterization** at coordinates 85.36161° S, 31.41594° E.
2. **Surface Composition**: 85.36162° S, 31.41252° E.
3. **Navigation & Mobility Test Site**.

---

## Pathfinding: Ant Colony Optimization
Our ACO algorithm optimizes the rover's path by avoiding obstacles like craters and boulders. The process involves pheromone-based pathfinding:
- Ants search for food and return to the colony, leaving pheromone trails that guide others. We simulate this behavior in **ACS** to identify optimal rover paths.

### Pathfinding Parameters:
- **Distance**: Changes based on elevation.
- **Speed**: Adjusted based on terrain steepness.
- **Heuristic Value**: Dynamically adjusted for optimal path calculation.

---

## Visualization
We used **Unity3D** to simulate the rover's movements based on the optimized traversal path. The rover starts at the **Landing Site** and navigates through the 10 scientific stops on the lunar surface.

#### Simulation Steps:
1. Extract terrain mesh using QGIS from DTM/DEM data.
2. Import the mesh into Unity.
3. Load the optimal path from the ACO results into the Unity script.
4. Simulate rover traversal across 10 scientific stops.

---

## Images

Here are some sample visualizations from the project:

### 1. **Landing Site Characterization**
   ![Landing Site](https://example.com/landing-site.png)  
   _Initial landing site for the rover at coordinates 85.36161° S._

### 2. **Crater & Boulder Detection Visualization**
   ![Crater Detection](https://example.com/crater-detection.png)  
   _Detected craters and boulders using YOLOv5 and traditional CV methods._

### 3. **Ant Colony Optimization Pathfinding**
   ![ACO Pathfinding](https://example.com/aco-pathfinding.png)  
   _ACO-based optimized path on lunar terrain, considering elevation and obstacles._

### 4. **Rover Simulation**
   ![Rover Simulation](https://example.com/rover-simulation.png)  
   _Unity3D simulation showing the rover traversing the optimal path._

---

## Contributors
- **Arpeet Chandane** - [GitHub](https://github.com/Abeey04)
- **Devesh Mhaske** - [GitHub](https://github.com/devesh)
- **Sahil Nichat** - [GitHub](https://github.com/sahil)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

