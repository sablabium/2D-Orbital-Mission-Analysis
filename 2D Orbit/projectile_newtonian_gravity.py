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
### Earth
Rplanet = 6.378e6 # meters
Mplanet = 5.972e24 # kg

### Gravitational Acceleration
def gravity(z):
    global Rplanet, Mplanet
    r = np.sqrt(z**2)
    if r < Rplanet:
        accel = 0.0
    else:
        accel = G*Mplanet/(z**3)*r

    return accel
### Equation of Motion
## F = m*a = m*zddot
## z is the altitude of the surface (m)
## zdot is the velocity
## zddot is acceleration
## Second Order Differential Equation
## Passing state vector and time
## for now state vector is two variables
def Derivatives(state, t):
    ### Globals
    global mass
    # state vector
    z = state[0]
    velz = state[1]

    ### Compute The Total Forces
    ### Gravity - z is positive UP so gravity needs to be negative
    gravityF = -gravity(z)*mass
    ### Aerodynamics
    aeroF = 0.0
    ### Thrust
    thrustF = 0.0
    ### Total
    Forces = gravityF + aeroF + thrustF

    # Compute z dot - kinematic relationship
    zdot = velz
    # Compute Acceleration
    zddot = Forces/mass
    # compute the state dot
    statedot = np.asarray([zdot, zddot])

    return statedot


################## MAIN SCRIPT #################

### Test Surface Gravity
print('surface gravity(m/s) = ', gravity(Rplanet))

### Initial Conditions
z0 = Rplanet # altitude
velz0 = 1064.0 #m/s
stateinitial = np.asarray([z0, velz0])

### Time Window
tout = np.linspace(0, 350, 100)
### Numerical Integration Call
stateout = sci.odeint(Derivatives, stateinitial, tout)
### Rename Variables
zout = stateout[:,0]
altitude = zout - Rplanet # Distance from the surface
velzout = stateout[:,1]

### Plot

### Altitude
plt.plot(tout, altitude)
plt.title('Height vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Height (m)')
plt.grid()
plt.show()

### Velocity
plt.plot(tout, velzout)
plt.title('Velocity vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Normal Speed (m)')
plt.grid()
plt.show()