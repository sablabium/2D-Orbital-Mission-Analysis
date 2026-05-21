"""
Created on : Sun May 10 4:04 PM 2026

@author: Saba
"""

import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt

plt.close('all')
### DEFINE CONSTANT PARAMETERS

G = 6.6742 * 10**-11 # Gravitational constant (SI unit)

### ROCKET
mass = 640.0/1000.0 # kg

### PLANET
# Earth
Rplanet = 6.378e6 # meters
mplanet = 5.972e24 # kg


### Gravitational Acceleration
def gravity(x,z):
    global Rplanet, mplanet
    r = np.sqrt(x**2 + z**2)
    if r < Rplanet:
        accelx = 0.0
        accelz = 0.0
    else:
        accelx = G*mplanet/(r**3)*x
        accelz = G*mplanet/(r**3)*z

    return np.asarray([accelx, accelz])
### Equation of Motion
## F = m*a = m*zddot
## z is the altitude from the center of the planet along the North Pole
## x is the altitude from the center of the planet along equator
## zdot is the velocity along z
## zddot is acceleration along x
## Second Order Differential Equation
## Passing state vector and time
## for now state vector is two variables
def Derivatives(state, t):
    ### Globals
    global mass
    # state vector
    x = state[0]
    z = state[1]
    velx = state[2]
    velz = state[3]

    # Compute z dot - kinematic relationship
    xdot = velx
    zdot = velz

    ### Compute The Total Forces
    ### Gravity - z is positive UP so gravity needs to be negative
    gravityF = -gravity(x,z)*mass
    ### Aerodynamics
    aeroF = np.asarray([0.0,0.0])
    ### Thrust
    thrustF = np.asarray([0.0,0.0])
    ### Total
    Forces = gravityF + aeroF + thrustF

    # Compute Acceleration
    ddot = Forces/mass
    # compute the state dot
    statedot = np.asarray([xdot,zdot, ddot[0],ddot[1]])

    return statedot


################## MAIN SCRIPT #################

### Initial Conditions
x0 = Rplanet # m
z0 = 0.0 # m
r0 = np.sqrt(x0**2 + z0**2)
velz0 = np.sqrt(G*mplanet/r0) # m/s
velx0 = 0.0 # m/s
stateinitial = np.asarray([x0, z0, velx0, velz0])
print(np.sqrt(G*mplanet/r0))

### Time Window
period = 2*np.pi/np.sqrt(G*mplanet)*r0**(3.0/2.0) # Calculate an orbital period
tout = np.linspace(0, period, 1000)
### Numerical Integration Call
stateout = sci.odeint(Derivatives, stateinitial, tout)
### Rename Variables
xout = stateout[:,0]
zout = stateout[:,1]
altitude = np.sqrt(xout**2 + zout**2) - Rplanet # Distance from the surface
velxout = stateout[:,2]
velzout = stateout[:,3]
velout = np.sqrt(velxout**2 + velzout**2)
### Plot

### Altitude
plt.figure(1)
plt.plot(tout, altitude)
plt.title('Height vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Height (m)')
plt.grid()
#plt.show()

### Velocity
plt.figure(2)
plt.plot(tout, velout)
plt.title('Velocity vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Total Speed (m)')
plt.grid()
#plt.show()

### 2D Orbit
theta = np.linspace(0, 2*np.pi, 100)
xplanet = Rplanet*np.sin(theta)
yplanet = Rplanet*np.cos(theta)
plt.figure(3)
plt.plot(xout, zout, 'r-', label='Orbit')
plt.title('2D Orbit')
plt.plot(xplanet, yplanet, 'b-', label='Planet')
plt.grid()
plt.legend()

plt.axis('equal')
plt.show()