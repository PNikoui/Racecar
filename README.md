# Neuroevolutionary Training of an Autonomous Racecar

Self-driving cars are a major area of research from both universities and the private sector nowadays. They will revolutionize transportation in a similar way that the first Ford cars did in the $19^{th}$ century. Some of the important areas where they will have an impact are road safety, where driver behaviour or error are a factor in 94 percent of crashes, higher independence for disabled people, reduced congestion and environmental benefits \cite{Singh2015CriticalRF}.

## Table of Contents
* [General Info](#general-information)
* [Setup](#setup)
* [Required Packages](#required-packages)
* [Features](#features)
* [Screenshots](#screenshots)



## General Information
Two methods of achieving self-driving cars are backpropagation-based reinforcement learning algorithms (RL) and genetic algorithms (GA). This repository will focus on GAs in combination with curriculum learning. The basis for this project was provided Deloitte Denmark with whom we have collaborated throughout the project.


## Setup
The training process is done using Colab's GPU which is found in the main notebook. 
### Required Packages
OpenAI Gym - pip install gym
Pytorch - pip install pytorch
Shapely - pip install shapely


## Features
The performance of different model architectures are investigated. The models are called upon from the models folder and the resulting weights are stored as dictionaries for their respective map orientation in the weights folder.


## Screenshots
![Example screenshot](./img/screenshot.png)
<!-- If you have screenshots you'd like to share, include them here. -->

