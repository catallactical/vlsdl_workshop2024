#!/bin/python3

# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Script to generate a circle-like trajectory with acceleration
#			and deceleration stages.


# ---   IMPORTS   --- #
# ⨪------------------ #
import matplotlib.pyplot as plt
import numpy as np
import sys

# ---   CONSTANTS   --- #
# --------------------- #
TIME_PER_ITER = 0.05 # In seconds (time step of the trajectory simulation)
ACCEL_TIME = 7.5  # How many seconds does the acceleration take
TRAVEL_TIME = 25.0  # How many seconds must be traveled at constant speed
DECEL_CONST = -0.75  # The constant deceleration in meter/second^2
MAX_SPEED = 2.8  # In meter/second
RADIUS = 15  # In meters
HEIGHT = 0  # In meters (z coordinate)


# ---   M A I N   --- #
# ------------------- #
if __name__ == '__main__':
	# ---  TRAJECTORY SIMULATION  --- #
	# ------------------------------- #
    # Compute parametric circle on the plane at given height
	vi, ai = 0, 0 # Speed and acceleration at i-th iteration
	theta_a, theta_b = -np.pi, np.pi  # Initial and final angle
	ti = 0  # Time at current iteration
	thetai = theta_a  # Angle at i-th iteration
	x, y = [], []  # Trajectory on the plane
	theta = []  # Angles
	v, vr, a = [], [0], []
	t = []  # Time points
	max_accel = 3*MAX_SPEED/(2*ACCEL_TIME)  # Max acceleration (peak)
	k = -6*MAX_SPEED/ACCEL_TIME**3  # Acceleration coefficient
	decel_time = -MAX_SPEED/DECEL_CONST  # Time from max speed to zero
	t_decel = 1e15  # Time point at which deceleration must start (to be computed)
	while vi!=0 or t_decel==1e15:  # Iterate until the circumference has been completed		
		# Store time, angle, and position on the plane for current iteration
		t.append(ti)
		theta.append(thetai)
		x.append(RADIUS*np.sin(thetai))
		y.append(RADIUS*np.cos(thetai))
		# Track speed
		v.append(vi)		
		# Acceleration phase
		if ti < ACCEL_TIME:			
			# Update speed for next iteration
			vi = min(MAX_SPEED, max_accel*ti+k*(ti**3/3 - ACCEL_TIME/2*ti*ti + (ACCEL_TIME/2)**2*ti))
			# Update and track acceleration
			ai = max_accel + k*((ti)-ACCEL_TIME/2)**2
			a.append(ai)
		# Deceleration phase
		elif ti >= t_decel:
			# Update speed for next iteration
			vi = max(0, DECEL_CONST*(ti-t_decel)+MAX_SPEED)
			# Update and track acceleration
			ai = DECEL_CONST
			a.append(ai)
		# Travel phase
		else:
			if t_decel == 1e15:  # Find deceleration values
				omega = MAX_SPEED*TIME_PER_ITER/RADIUS
				# Legacy t_decel (not simplified)
				#t_decel = -(  # Compute the time point to start deceleration
				#	TIME_PER_ITER*decel_time**2*DECEL_CONST
				#	- 2*np.pi*TIME_PER_ITER*RADIUS
				#	+ 2*TIME_PER_ITER*decel_time*MAX_SPEED
				#	- 2*omega*RADIUS*ACCEL_TIME
				#	+ 2*TIME_PER_ITER*RADIUS*theta[-1]
				#)/(
				#	2*omega*RADIUS
				#)
				# Simplified t_decel
				t_decel = -(  # Compute the time point to start deceleration
					decel_time**2*DECEL_CONST
					- 2*np.pi*RADIUS
					+ 2*decel_time*MAX_SPEED
					+ 2*RADIUS*theta[-1]
					- 2*MAX_SPEED*ACCEL_TIME
				)/(
					2*MAX_SPEED
				)
			# Update and track acceleration
			ai = 0
			a.append(ai)
		# Update angle for next iteration
		thetai = theta[-1] + vi*TIME_PER_ITER/RADIUS
		# Track real speed
		if len(theta) > 1:
			vr.append((RADIUS*(theta[-1]-theta[-2]))/TIME_PER_ITER)		
		# Update time for next iteration
		ti += TIME_PER_ITER


	# ---  EXPORT TRAJECTORY  --- #
	# --------------------------- #
	# Generate trajectory
	theta = np.array(theta)
	x, y = np.array(x), np.array(y)
	z = np.ones_like(theta)*HEIGHT
	t = np.array(t)  # Time
	roll_and_pitch = np.zeros_like(theta)  # Roll and pitch, both are zero
	yaw = -theta
	traj = np.vstack([t, x, y, z, roll_and_pitch, roll_and_pitch, yaw]).T
	# Print trajectory file through stdout
	np.savetxt(sys.stdout, traj, fmt='%.4f', delimiter=',')



	# ---  DEBUG FIGURE  --- #
	# ---------------------- #
	# Plot trajectory
	fig = plt.figure(figsize=(10, 7))

	# Position subplot
	ax = fig.add_subplot(2, 2, 1)
	ax.scatter(x, y, lw=2, c=t, cmap='viridis', s=16)
	ax.set_xlabel('$x$')
	ax.set_ylabel('$y$')
	ax.axis('equal')
	ax.set_title('Trajectory on the plane')

	# Speed and acceleration subplot
	ax = fig.add_subplot(2, 2, 2)
	plt_speed = ax.plot(t, v, lw=2, color='tab:blue', label=r'Speed $[m/s]$')
	ax2 = ax.twinx()
	plt_accel = ax2.plot(t, a, lw=2, color='tab:red', label=r'Acceleration $[m/s^2]$')
	ax.set_zorder(ax2.get_zorder()+1)
	ax.set_frame_on(False)
	plts = plt_speed+plt_accel
	ax.set_xlabel(r'$t$ [$\mathrm{s}$]')
	ax.set_ylabel(r'$v(t)$ [$\mathrm{m}/\mathrm{s}$]')
	ax2.set_ylabel(r'a(t) [$\mathrm{m}/\mathrm{s}^2$]')
	ax.axis('equal')
	ax.legend(plts, [plot.get_label() for plot in plts], loc='upper right')	
	ax.grid('both')
	ax.set_axisbelow(True)
	ax.set_title('Speed and acceleration')

	# Speed validation subplot
	ax = fig.add_subplot(2, 2, 3)
	ax.plot(t, v, lw=5, color='black', label=r'Speed (expected) $[m/s]$', zorder=4)
	ax.plot(t, vr, lw=2, color='tab:green', label=r'Speed (model) $[m/s]$', zorder=5)
	ax.set_xlabel(r'$t$ [$\mathrm{s}$]')
	ax.set_ylabel(r'$v(t)$ [$\mathrm{m}/\mathrm{s}$]')
	ax.axis('equal')
	ax.grid('both')
	ax.set_axisbelow(True)
	ax.legend(loc='best')
	ax.set_title('Model and expected speed')

	# Angle subplot
	ax = fig.add_subplot(2, 2, 4)
	ax.plot(t, theta*180/np.pi, lw=2, color='purple', label=r'$\theta$ [deg]')
	ax.set_xlabel(r'$t$ [s]')
	ax.set_ylabel(r'$\theta(t)$ [deg]')
	ax.grid('both')
	ax.set_axisbelow(True)
	ax.legend(loc='best')
	ax.set_title('Angle over time')

	# Show plot
	fig.tight_layout()
	plt.show()
