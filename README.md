## Pathfinder Artificial Intelligence
### Introduction
Programming AI to find an optimal path is a classic problem in AI. 
What seems intuitive to humans is less intuitive to an agent that relies on its vision 
and a rule-base to find the way through the maze. 

To solve this assignment, I programmed a lot of overhead around the agent. 
I made 4 classes:
1. Coord2D
2. WorldMap
3. Robot
4. Simulation

### Features

**The program**: Coord2D only exists to help the 2-dimensional properties of the agent and the world. 
Worldmap generates the world, while Robot ensures I can run a robot-agent. 
Robot has most methods regarding moving and the rule base and helper methods, perceive the environment etc. 
Simulation is the class to run a simulation and test the agent. 
It has the fundamental method “runSim” that runs a simulation. 
Class Simulation is the only class the user needs to edit to run different experiments.

**Requirements**: The only requirement to run the simulation is to have an instance of Python installed. 
The program has only been tested on Python 3.8.3, so please try it on this version. 
It may also work on other versions of Python.

**Running a simulation**: The simulation can be initiated by running the main file. 
The simulation logs the movement, the position, and the orientation of the agent. 
The user will also see what the agent can see from these updates. 
The logs show the order of each action. It records both rotations and moves as actions. 

The simulation will provide two maps for the user to monitor during the simulation. 
The map to the left will show the true map, while the map of the right
shows what the robot has perceived about the map.
The robot only sees straight forward, and 45 degrees to its left and right. 

**Changing simulation parameters**: If you want to change the parameters of the simulation you should consider 
the following parameters: 
- _Standard_Init_Row_ and/or _Standard_Init_Column_ to change the starting position of the agent.

- _Standard_Goal_Row_ and/or _Standard_Goal_Column_ to change the position of the goal. 
Remember to keep it inside the bounds of the map.

- _Standard_Map_Row_ and/or _Standard_Map_Column_ to change the length or width of the map.

- _Obstacle_percentage_ takes values between 0 and 1 and determines the ratio of obstacles tiles in the map.

**Note:** It is not strictly necessary to edit these values - they default to 
starting position (6,6), goal position (32,37) and map size 35x40 with 20% obstacles. 
The simulation also generates a random map for each simulation. 

### Files in this directory:
The current files should exist in the directory:
- assignment1.py
- README.md
- GradingSheet1.docx
- Assingment1_report.pdf
- output.pkl (only after 1 simulation has been run)

### Contact info:
Author: Lars Engesæth

email: lengesaeth3430@sdsu.edu / laen@nmbu.no

