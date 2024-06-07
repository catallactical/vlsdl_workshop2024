#!/bin/python3

# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to generate a circle-like trajectory


# ---   IMPORTS   --- #
# ⨪------------------ #
import numpy as np
import sys

# ---   CONSTANTS   --- #
# --------------------- #
ANGULAR_STEPS = 300
TIME_PER_STEP = 0.05  # In seconds
RADIUS = 15  # In meters
HEIGHT = 0  # In meters (z coordinate)

# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
    # Compute parametric circle on the plane at given height
    theta = np.linspace(-np.pi, np.pi, ANGULAR_STEPS)
    x = RADIUS*np.sin(theta)
    y = RADIUS*np.cos(theta)
    z = np.ones_like(theta)*HEIGHT
    # Generate trajectory
    t = np.linspace(0, TIME_PER_STEP*(ANGULAR_STEPS-1), ANGULAR_STEPS)  # Time
    roll_and_pitch = np.zeros_like(theta)  # Roll and pitch, both are zero
    yaw = -theta
    traj = np.vstack([t, x, y, z, roll_and_pitch, roll_and_pitch, yaw]).T
    # Print trajectory file through stdout
    np.savetxt(sys.stdout, traj, fmt='%.4f', delimiter=',')
