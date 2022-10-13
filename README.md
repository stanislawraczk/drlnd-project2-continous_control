# DRLND-project2-continous-control

In this project agent is trained to move an arm to follow a green ball in an unity enviroment

## Enviroment

State space in this enviroment has 33 dimentions corresponding to position, rotation, velocity, and angular velocities of the arm. Given this information agent has to move the arm to target location.

Action space is continous and consists of a 4 number vector corresponding to torque applicable to two joints. 

The task is solved when the average score of the agent in 100 consecutive episodes is 30 or higher.

## Setup

To run the code, either to train or check trained agent, clone the repository and install requirements using:

```bash
pip install -r requirements.txt
```

This task was solved using Python 3.9 in order to use newer version of pytorch that would cooperate with CUDA 116

Then download the unity enviorment from the link below and unzip it in into the cloned repository. Links provided below are for the 20 agent version of the enviroment.

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)

[Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)

[Windows(32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)

[Windows(64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Run

To train agent run cells from Continous_Control.ipynb notebook
