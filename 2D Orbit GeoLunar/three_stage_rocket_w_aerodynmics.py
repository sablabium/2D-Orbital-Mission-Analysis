"""
Created on : Sun May 10 4:04 PM 2026

@author: Saba Ghudushauri
"""

import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt

plt.close('all')
np.seterr(all='warn')
### DEFINE CONSTANT PARAMETERS

G = 6.6742 * 10**-11 # Gravitational constant (SI unit)
### PLANET
celestial_name = np.asarray(['Earth', 'Moon'])
celestial_radius = np.asarray([6.378e6, 1737400])
celestial_mass = np.asarray([972e24, 7348e22])

CelestialBodies = {
    'name': celestial_name,
    'radius': celestial_radius,
    'mass': celestial_mass,
}
# Earth
Rplanet = 6.378e6 # meters
mplanet = 5.972e24 # kg
name = 'Earth'
# Moon
Rmplanet = 1737400 # meters
mmplanet = 7348e+22

### ROCKET
### Initial Conditions -- for single stage rocket ###
x0 = Rplanet # m
z0 = 0 # m
velz0 = 0.0 # m/s
velx0 = 0.0 # m/s
minutes = 10
period = minutes #*15
mass0 = 2900000.0 # kg Initial mass - All of the rocket - WET
max_thrust = 33000000.0 # Newtons
Isp1 = 250.0 # Specific Impulse - thrust per unit of fuel in seconds = F / mdot * 9.81
Isp2 = 420.0
Isp3 = 421.0
tMECO = 160.0 # time Main Engine Cutoff
angle1 = 65 # degrees of lean in the first stage launch

mass1 = 137000.0 # kg of the first stage - DRY
tsep1 = 1.0 # Length of time to remove the first stage
t2start = 170.0 # (s) When to Start Second Stage Burn
t2end = t2start + 360
max_thrust2 = 5400000 # Newtons Second stage Thrust in
angle2 = 95 # Degrees Of Lean in The Second Stage Launch

mass2 = 15200.0 # kg of the first stage - DRY
tsep2 = 1.0 # Length of time to remove the first stage
t3start = 5500.0 # (s) When to Start Second Stage Burn
t3end = t3start + 270
max_thrust3 = 1000000 # Newtons Second stage Thrust in
angle3 = 90

S = 10.1 # Cross Sectional Area of the Rocket # TODO: different stages different S
CD = 0.4 # Drag Coefficient

### Aerodynamics Class
class Aerodynamics():
    def __init__(self, name):
        self.name = name
        if name == 'Earth':
            ### import aeromodel for earth
            data = np.loadtxt('earth_athmosphere_density.txt')
            self.altitude = data[:,0]
            self.density = data[:,1]
            self.rhos = self.density[0] #1.225  # kg/m**3
            self.beta = 0.1354 / 1000.0  # density constant for earth
        else:
            print('No Such Body Found to set it\'s aero')
    def getDensity(self, altitude):
        if self.name == 'Earth':
            ### Special Equation Using beta, if we do this, I wont have to import earth atmosphere density
            #rho = self.rhos*np.exp(-self.beta*altitude)

            ### Simplified Equation
            rho = np.interp(altitude, self.altitude, self.density)
            #print(rho)
        else:
            #np.interp(altitude, self.altitude, self.density)
            print('No such Planet to set rho')
        return rho

# Create the aeroModel variable which is instance
# of the class Aerodynamics, putting it here so it's global
aeroModel = Aerodynamics(name)

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

    return np.asarray([accelx, accelz]), r

def propulsion(t):
    global max_thrust, Isp, tMECO, ve, theta
    if t < tMECO:
        #print(np.linspace(0, angle1 * np.pi / 180.0, int(tMECO))[int(t)])
        # TODO: this is messy!
        theta = np.linspace(0, angle1, int(tMECO))[int(t)] * np.pi / 180.0
        #theta = angle1 * np.pi / 180.0
        ve = Isp1 * 9.81  # m/s
        thrustF = max_thrust
        mdot = -thrustF/ve
    # TODO: make it instant, mass separation should happen after sep1, and separate instantly not gradually
    if t > tMECO and t < (tMECO + tsep1):
        thrustF = 0.0
        # Mass lost from first stage separation
        mdot = -mass1/tsep1
    if t > (tMECO + tsep1):
        thrustF = 0.0
        mdot = 0.0
    # Second Stage Thrust
    if t > t2start and t < (t2end):
        # TODO: this is messy!
        theta = np.linspace(angle1, angle2, int(t2end-t2start))[int(t-t2start)] * np.pi / 180.0
        #theta = angle2 * np.pi / 180.0
        ve = Isp2 * 9.81  # m/s
        thrustF = max_thrust2
        mdot = -thrustF / ve
    if t > t2end and t < (t2end + tsep2):
        thrustF = 0.0
        # Mass lost from first stage separation
        mdot = -mass2/tsep2
    if t > (t2end + tsep2):
        thrustF = 0.0
        mdot = 0.0
    if t > t3start and t < (t3end):
        # TODO: this is messy!
        theta = angle3 * np.pi / 180.0
        ve = Isp3 * 9.81  # m/s
        thrustF = max_thrust3
        mdot = -thrustF / ve
    if t > t3end:
        thrustF = 0.0
        mdot = 0.0

    ## Angle of the Thrust
    thrustx = thrustF*np.cos(theta)
    thrustz = thrustF*np.sin(theta)
    # thrustx = 0.0
    # thrustz = 0.0
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
    global aeroModel, Rplanet
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
    accel, r = gravity(x,z)
    gravityF = -accel*mass

    ### Aerodynamics
    altitude = r - Rplanet # Distance from the surface
    rho = aeroModel.getDensity(altitude) # Density
    V = np.sqrt(velx**2 + velz**2)
    qinf = 0.5*rho*S*np.abs(V)
    aeroF = -qinf* CD * np.asarray([velx, velz])

    ### Thrust
    thrustF, mdot = propulsion(t)
    #thrustF, mdot = [0.0,1000.0]

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

### Test Surface Gravity
#print('Surface gravity is: ', gravity(0,Rplanet))

### plot Air Density as a Function of Above Ground Model
test_altitude = np.linspace(0, 100000,100)
test_rho = aeroModel.getDensity(test_altitude)
plt.figure(10)
plt.plot(test_altitude, test_rho, 'b-')
plt.xlabel('altitude (m)')
plt.ylabel('air density (kg/m**3)')
plt.grid()
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


### Populate Initial Condition Vector
stateinitial = np.asarray([x0, z0, velx0, velz0, mass0])
### Time Window
tout = np.linspace(0, period, 1000*15)
### Numerical Integration Call
stateout, info = sci.odeint(Derivatives,stateinitial,tout, full_output=True)
# print(info['message'])  # Explains the specific failure
# print(info['tcur'])
#print(info['message'])

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

xmoon = Rmplanet*np.sin(theta) - 384400000
ymoon = Rmplanet*np.cos(theta)
plt.figure(4)
plt.plot(xout, zout, 'r-', label='Orbit')
plt.title('2D Orbit')
### Earth
plt.plot(xplanet, yplanet, 'b-', label='Planet')
### Moon
#plt.plot(xmoon, ymoon, 'g-', label='Moon')
plt.grid()
plt.legend()

plt.axis('equal')
plt.show()