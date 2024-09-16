#!/bin/python3

# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to generate a cardioid trajectory


# ---   IMPORTS   --- #
# ⨪------------------ #
import numpy as np
import sys


# ---   CONSTANTS   --- #
# --------------------- #
ANGULAR_STEPS = 200
TIME_PER_STEP = 0.03  # In seconds
RADIUS = 20.0  # In meters
HEIGHT = 100.0  # In meters (z coordinate)
Z_STDEV = 0.5  # In meters


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    # Compute parametric spiral on the plane at given height
    theta = np.linspace(0, 2*np.pi, ANGULAR_STEPS)
    t = np.linspace(0, 1, ANGULAR_STEPS)  # Param
    x = 2.0*RADIUS*(1.0-np.cos(theta))*np.cos(theta)
    y = 2.0*RADIUS*(1.0-np.cos(theta))*np.sin(theta)
    z = np.ones_like(theta)*HEIGHT + np.random.normal(0, Z_STDEV, theta.shape)
    # Generate trajectory
    time = t * TIME_PER_STEP*(ANGULAR_STEPS-1)
    roll_and_pitch = np.zeros_like(theta)  # Roll and pitch, both are zero
    yaw = -theta
    traj = np.vstack([time, x, y, z, roll_and_pitch, roll_and_pitch, yaw]).T
    # Print trajectory file through stdout
    np.savetxt(sys.stdout, traj, fmt='%.4f', delimiter=',')
