# Neuroevolutionary Training of an Autonomous Racecar

Self-driving cars are a major area of research from both universities and the private sector nowadays. They will revolutionize transportation in a similar way that the first Ford cars did in the 19th century. Some of the important areas where they will have an impact are road safety, where driver behaviour or error are a factor in 94 percent of crashes, higher independence for disabled people, reduced congestion and environmental benefits [[1]](#1).

Two methods of achieving self-driving cars are backpropagation-based reinforcement learning algorithms (RL) and genetic algorithms (GA). This repository will focus on GAs in combination with curriculum learning. The basis for this project was provided Deloitte Denmark with whom we have collaborated throughout the project.



## Table of Contents
* [General Info](#general-information)
* [Setup](#setup)
* [Required Packages](#required-packages)
* [Features](#features)
* [Results and Plots](#results-and-plots)



## General Information

Darwin inspired Neuroevolutionary methods are implemented to accelerate the training of an autonomous racecar. Curriculum learning is applied iteratively to improve the ability of the agent to navigate more difficult environments. It was found that the performance of the agents was highly sensitive to the applied penalty functions, which are responsible for the fitness determination of the subsequent generations. Furthermore, it was found that the training improved when put in context of previous states which were provided by a LTSM network.  

## Setup
The virtual environment is registered using the setup.py file, the required file structure is specified in gym-Racecar. The virtual racetrack is generated using  training process is done using Colab's GPU which is found in the main notebook. 

### Required Packages

OpenAI Gym - pip install gym

Pytorch - pip install pytorch

Shapely - pip install shapely


## Features
The performance of different model architectures are investigated. The models are named accordingly and are found in the python files folder. The weights resulting from the training process are stored as dictionaries for their respective map orientation in the weights folder.


## Results and Plots
![3 Turn Map Example](https://github.com/PNikoui/Racecar/blob/Fall_2020/Images/Maps/Map3.JPG | width=300)

![Fitness Plot for 3 Turn Map with Vanilla Network](https://github.com/PNikoui/Racecar/blob/Fall_2020/Images/51%20Generations%20Vanilla/Fitness3.JPG)
![Goals Plot for 3 Turn Map with Vanilla Network](https://github.com/PNikoui/Racecar/blob/Fall_2020/Images/51%20Generations%20Vanilla/Goal3.JPG)



## References
<a id="1">[1]</a> 
Santokh Singh (2015). 
Critical Reasons for Crashes Investigated in the National Motor Vehicle Crash Causation Survey
