import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
plt.close('all')
### DEFINE CONSTANT PARAMETERS

### ROCKET
mass = 640.0/1000.0 # kg


### PLANET

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
    gravity = -9.81*mass
    ### Aerodynamics
    aero = 0.0
    ### Thrust
    thrust = 0.0
    ### Total
    Forces = gravity + aero + thrust

    # Compute z dot - kinematic relationship
    zdot = velz
    # Compute Acceleration
    zddot = Forces/mass
    # compute the state dot
    statedot = np.asarray([zdot, zddot])

    return statedot


################## MAIN SCRIPT #################

### Initial Conditions
z0 = 0.0 # altitude
velz0 = 164.0 #m/s
stateinitial = np.asarray([z0, velz0])

### Time Window
tout = np.linspace(0, 35, 100)
### Numerical Integration Call
stateout = sci.odeint(Derivatives, stateinitial, tout)
print(stateout)
### Rename Variables
zout = stateout[:,0]
velzout = stateout[:,1]

### Plot

### Altitude
plt.plot(tout, zout)
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