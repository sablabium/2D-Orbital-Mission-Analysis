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
### PLANET
# Earth
Rplanet = 6.378e6 # meters
mplanet = 5.972e24 # kg

### ROCKET
### Initial Conditions -- for single stage rocket ###
x0 = Rplanet # m
z0 = 0.0 # m
velz0 = 0.0 # m/s
velx0 = 0.0 # m/s
period = 550
mass0 = 2900000.0 # kg Initial mass - All of the rocket with full fuel
max_thrust = 33000000.0 # Newtons
Isp = 250.0 # Specific Impulse - thrust per unit of fuel in seconds = F / mdot * 9.81
tMECO = 150.0 # time Main Engine Cutoff

### Gravitational Acceleration
def gravity(x,z):
    global Rplanet, mplanet
    r = np.sqrt(x**2 + z**2)
    if r < Rplanet:
        accelx = 0
        accelz = 0
    else:
        accelx = G*mplanet/(r**3)*x
        accelz = G*mplanet/(r**3)*z

    return np.asarray([accelx, accelz])

def propulsion(t):
    global max_thrust, Isp, tMECO, ve
    if t < tMECO:
        thrustF = max_thrust
    else:
        thrustF = 0.0
    ## Angle of the Thrust
    theta = 0.1
    thrustx = thrustF*np.cos(theta)
    thrustz = thrustF*np.sin(theta)

    mdot = -thrustF/ve
    return np.asarray([thrustx, thrustz]), mdot
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

    # state vector
    x = state[0]
    z = state[1]
    velx = state[2]
    velz = state[3]
    mass = state[4]

    # Compute z dot - kinematic relationship
    xdot = velx
    zdot = velz

    ### Compute The Total Forces
    ### Gravity - z is positive UP so gravity needs to be negative
    gravityF = -gravity(x,z)*mass
    print(mass)
    ### Aerodynamics
    aeroF = np.asarray([0.0,0.0])
    ### Thrust
    thrustF, mdot = propulsion(t)
    ### Total
    Forces = gravityF + aeroF + thrustF

    # Compute Acceleration
    if mass > 0:
        ddot = Forces/mass
    else:
        ddot = 0.0
        mdot = 0.0
    # compute the state dot
    statedot = np.asarray([xdot,zdot, ddot[0],ddot[1], mdot])

    return statedot


################## MAIN SCRIPT #################


"""
### Initial Conditions -- for orbit ###
x0 = Rplanet # m
z0 = 0.0 # m
r0 = np.sqrt(x0**2 + z0**2)
velz0 = np.sqrt(G*mplanet/r0) # m/s
velx0 = 0.0 # m/s
stateinitial = np.asarray([x0, z0, velx0, velz0])
period = 2*np.pi/np.sqrt(G*mplanet)*r0**(3.0/2.0) # Calculate an orbital period
"""

### Compute Exit Velocity
ve = Isp * 9.81  # m/s
### Populate Initial Condition Vector
stateinitial = np.asarray([x0, z0, velx0, velz0, mass0])
### Time Window
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
massout = stateout[:,4]

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

### Mass
plt.figure(3)
plt.plot(tout, massout)
plt.title('Mass vs Time')
plt.xlabel('Time (sec)')
plt.ylabel('Total Mass (kg)')
plt.grid()

### 2D Orbit
theta = np.linspace(0, 2*np.pi, 1000)
xplanet = Rplanet*np.sin(theta)
yplanet = Rplanet*np.cos(theta)
plt.figure(4)
plt.plot(xout, zout, 'r-', label='Orbit')
plt.title('2D Orbit')
plt.plot(xplanet, yplanet, 'b-', label='Planet')
plt.grid()
plt.legend()

plt.axis('equal')
plt.show()