#!/bin/python3

# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to generate a spiral trajectory


# ---   IMPORTS   --- #
# ⨪------------------ #
import numpy as np
import sys


# ---   CONSTANTS   --- #
# --------------------- #
ANGULAR_STEPS = 200
REVOLUTIONS = 3  # Number of spiral revolutions
TIME_PER_STEP = 0.03  # In seconds
MIN_RADIUS = 15.0  # In meters
MAX_RADIUS = 60.0  # In meters
HEIGHT = 100.0  # In meters (z coordinate)


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    # Compute parametric spiral on the plane at given height
    theta = np.linspace(-REVOLUTIONS*np.pi, REVOLUTIONS*np.pi, ANGULAR_STEPS)
    radius_slope = MAX_RADIUS-MIN_RADIUS
    t = np.linspace(0, 1, ANGULAR_STEPS)  # Param
    r = radius_slope*t+MIN_RADIUS  # Radii
    x = r*np.sin(theta)
    y = r*np.cos(theta)
    z = np.ones_like(theta)*HEIGHT
    # Generate trajectory
    time = t * TIME_PER_STEP*(ANGULAR_STEPS-1)
    roll_and_pitch = np.zeros_like(theta)  # Roll and pitch, both are zero
    yaw = -theta
    traj = np.vstack([time, x, y, z, roll_and_pitch, roll_and_pitch, yaw]).T
    # Print trajectory file through stdout
    np.savetxt(sys.stdout, traj, fmt='%.4f', delimiter=',')
