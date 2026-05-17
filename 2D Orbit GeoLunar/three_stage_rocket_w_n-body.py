"""
Created on : Sun May 10 4:04 PM 2026

@author: Saba Ghudushauri
"""

import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
#import plotly.express as px

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
mmplanet = 7.348e22
ecc = 0.0549 # eccentricity
per = 363300000 # (m) perigee
apo = 405500000 # (m) apogee
incl = 0.0 # degrees in inclination
orb_pd = 27.322 * 24 * 60 * 60 # (sec) Orbital Period
a = (per + apo) / 2
init_degrees = 110 # Starting degrees of the moon in relation to Earth

### ROCKET
### Initial Conditions -- for single stage rocket ###
seconds = 0.0
minutes = 0.0
hours = 0.0
days = 0.0
period = 10000 * 250 #*100
t_span = (0, period)
t_eval = np.linspace(0, period, int(period/75))

### Ascent Guidance ###
# target_peri = 220 * 1000 # (m) target periapsis
# target_apo = 220 * 1000 # (m) target apoapsis
# booster_pitch_start =
# last_stage =

x0 = Rplanet # m
z0 = 0 # m
velz0 = 0.0 # m/s
velx0 = 0.0 # m/s
mass0 = 2900000.0 # kg Initial mass - All of the rocket - WET



Isp1 = 250.0 # Specific Impulse - thrust per unit of fuel in seconds = F / mdot * 9.81
max_thrust = 33000000.0 # Newtons
tMECO = 160.0 # time Main Engine Cutoff
angle1 = 65 # degrees of lean in the first stage launch

mass1 = 137000.0 # kg of the first stage - DRY
tsep1 = 1.0 # Length of time to remove the first stage
t2start = 170.0 # (s) When to Start Second Stage Burn
t2end = t2start + 360
Isp2 = 420.0
max_thrust2 = 5400000 # Newtons Second stage Thrust in
angle2 = 95 # Degrees Of Lean in The Second Stage Launch

mass2 = 15200.0 # kg of the first stage - DRY
tsep2 = 1.0 # Length of time to remove the first stage
t3start = 5550.0 # (s) When to Start Second Stage Burn
t3end = t3start + 269.0
Isp3 = 421.0
max_thrust3 = 1000000 # Newtons Second stage Thrust in
angle3 = 94

t4start = 60000.0
t4end = t4start + 1000.0

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

    ### Right now r is from the 0,0. but we should iterate through each planet
    ### and offset each r in relation to the planet center
    ### foreach planet calculate r, and add to accel

    ### Distance from the 0,0, Need to offset it by earth location
    r = np.sqrt(x**2 + z**2)
    if r < Rplanet:
        return np.asarray([0.0,0.0]), (0.0,0.0)

    accel_mag = G * mplanet / (r ** 2) ## Magnitude of Earth Gravity

    accelx = accel_mag * x/r
    accelz = accel_mag * z/r

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
    # Coast
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
    # First Stage Burn 1
    if t > t3start and t < (t3end):
        # TODO: this is messy!
        theta = angle3 * np.pi / 180.0
        ve = Isp3 * 9.81  # m/s
        thrustF = max_thrust3
        mdot = -thrustF / ve
    # Coast
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
def derivatives(t, state):
    ### Globals
    global aeroModel, Rplanet
    # state vector
    x,z,velx,velz,mass = state

    mx, mz = moonOrbit(t)

    accel_earth, r = gravity(x,z)

    # Distance from Rocket to Moon
    dx = x - mx
    dz = z - mz
    r_moon = np.sqrt(dx**2 + dz**2)

    # Standard G*M/r^2 formula
    # We add a small '1e3' to prevent infinite gravity if you crash into the moon
    accel_moon_mag = G * mmplanet / (r_moon**2 + 1e3)
    accel_moon_x = -accel_moon_mag * (dx / r_moon)
    accel_moon_z = -accel_moon_mag * (dz / r_moon)

    # Compute z dot - kinematic relationship
    xdot = velx
    zdot = velz

    ### Compute The Total Forces
    ### Gravity - z is positive UP so gravity needs to be negative
    #accel, r = gravity(x,z)
    #gravityF = [ax, az]

    ### Aerodynamics
    # altitude = r - Rplanet # Distance from the surface
    # rho = aeroModel.getDensity(altitude) # Density
    # V = np.sqrt(velx**2 + velz**2)
    # qinf = 0.5*rho*S*np.abs(V)
    # aeroF = -qinf* CD * np.asarray([velx, velz])

    ### Thrust
    thrustF, mdot = propulsion(t)
    #thrustF, mdot = [0.0,1000.0]

    ax = -accel_earth[0] + accel_moon_x + thrustF[0]/mass
    az = -accel_earth[1] + accel_moon_z + thrustF[1]/mass
    ### Total
    Forces = [ax,az] #+ aeroF

    # Compute Acceleration
    if mass > 0:
        ddot = Forces#/mass
    else:
        ddot = 0.0
        mdot = 0.0

    # compute the state dot
    statedot = np.asarray([xdot,zdot, ddot[0],ddot[1], mdot])


    return statedot


### Moon Orbit
def moonOrbit(t):
    ### Constant
    mu = G * (mplanet * mmplanet)

    ### Mean Anomaly M if the orbit was the perfect circle
    M = (2 * np.pi * t / orb_pd) + (init_degrees * np.pi / 180.0)

    ### Solve Keplers Equation M = E - e*sin(E)
    ### We need to find E - Eccentric Anomaly
    ### 5 Is enough for moons Eccentricity,
    E = M
    for _ in range(5):
        E = E - (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))

    ### Calculate True Anomaly Theta, This is where velocity change happens, depending
    ### On The Moons Place in the Orbit
    th = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2))

    ### Using Kepler's First Law for distance
    r = a * (1 - ecc**2)/(1 + ecc * np.cos(th))

    ### Velocity Magnitude (speed)
    speed = np.sqrt(mu * (2/r - 1/a))

    ### Calculating Moon Velocity
    vx = -np.sqrt(mu / (a * (1 - ecc**2))) * np.sin(th)
    vz =  np.sqrt(mu / (a * (1 - ecc**2))) * (ecc + np.cos(th))

    # Convert to Cartesian coordinates
    x = r * np.cos(th) # X coordinate
    z = r * np.sin(th) # z coordinate

    return np.asarray([x,z])


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
#tout = np.linspace(0, period, period) # 1000*15
### Numerical Integration Call ## Using high precision model
sol = sci.solve_ivp(derivatives, t_span, stateinitial, t_eval = t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
tout = sol.t
stateout = sol.y.T
# print(info['message'])  # Explains the specific failure
# print(info['tcur'])
#print(info['message'])

### Calculate Moon Position
moon_positions = np.array([moonOrbit(t) for t in tout])
moon_x = moon_positions[:, 0]
moon_z = moon_positions[:, 1]
#moon_x = stateout[:5]
#moon_z = stateout[:6]

### Rename Variables
xout = stateout[:,0]
zout = stateout[:,1]
altitude = np.sqrt(xout**2 + zout**2) - Rplanet # Distance from the surface
velxout = stateout[:,2]
velzout = stateout[:,3]
velout = np.sqrt(velxout**2 + velzout**2)
massout = stateout[:,4]

### Plot

### MATPLOTLYB PLOTS ###
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
zplanet = Rplanet*np.cos(theta)

xmoon = Rmplanet*np.sin(theta) + moon_x[-1]
zmoon = Rmplanet*np.cos(theta) + moon_z[-1]
plt.figure(4)
plt.title('2D Orbit')
### Earth
plt.plot(xplanet, zplanet, 'b-', label='Planet')
### ANIMATION
### Rocket
plt.plot(xout, zout, 'r-', label='Orbit')
### Moon
plt.plot(xmoon, zmoon, 'g-', label='Moon')
plt.plot(moon_x, moon_z, color='grey', linewidth=1, label='Moon Orbit')
plt.grid()
plt.legend()

plt.axis('equal')
plt.show()

import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_xlim(-a * 1.2, a * 1.2)  # Scale based on Moon's semi-major axis
ax.set_ylim(-a * 1.2, a * 1.2)

# Create the visual elements
earth_plot = ax.plot(xplanet, zplanet, 'bo', markersize=1, label='Earth')[0]
moon_plot, = ax.plot([], [], 'ko', markersize=5, label='Moon')
moon_trail, = ax.plot([], [], 'k-', ls=':', markersize=1) # Moon Path
rocket_plot, = ax.plot([], [], 'r-', label='Rocket Path')
rocket_dot, = ax.plot([], [], 'ro', markersize=3)  # Current rocket position


step = 50
def update(i):
    # 'step' makes the animation run faster by skipping data points
    idx = i * step

    if idx >= len(tout):
        idx = len(tout) - 1

    # Update Moon dot
    moon_plot.set_data([moon_x[idx]], [moon_z[idx]])
    # Update Rocket dot and trailing line
    rocket_dot.set_data([xout[idx]], [zout[idx]])
    rocket_plot.set_data(xout[:idx], zout[:idx])
    # Update Moon Trail
    moon_trail.set_data(moon_x[:idx], moon_z[:idx])

    return moon_plot, rocket_dot, rocket_plot, moon_trail


# Set 'frames' based on your step size
ani = animation.FuncAnimation(fig, update, frames=len(tout) // step, interval=20, blit=True, repeat=True)
plt.grid()
plt.legend()
plt.show()
### MATPLOTLYB PLOTS ###

'''### PLOTLY PLOTS ###
# Altitude
fig1 = px.line(x=tout, y=altitude, title='Height vs Time',
               labels={'x': 'Time (sec)', 'y': 'Height (m)'})
fig1.show()

# Velocity
fig2 = px.line(x=tout, y=velout, title='Velocity vs Time',
               labels={'x': 'Time (sec)', 'y': 'Total Speed (m/s)'})
fig2.show()

# Mass
fig3 = px.line(x=tout, y=massout, title='Mass vs Time',
               labels={'x': 'Time (sec)', 'y': 'Total Mass (kg)'})
fig3.show()

import plotly.graph_objects as go

### 2D Orbit
theta = np.linspace(0, 2*np.pi, 1000)
xplanet = Rplanet*np.sin(theta)
zplanet = Rplanet*np.cos(theta)

xmoon = Rmplanet*np.sin(theta) + moon_x[-1]
zmoon = Rmplanet*np.cos(theta) + moon_z[-1]

fig4 = go.Figure()

# Orbit path
fig4.add_trace(go.Scatter(x=xout, y=zout, mode='lines', name='Orbit', line=dict(color='red')))

# Planet
fig4.add_trace(go.Scatter(x=xplanet, y=zplanet, mode='lines', name='Planet', line=dict(color='blue')))

# Moon
fig4.add_trace(go.Scatter(x=xmoon, y=zmoon, mode='lines', name='Moon', line=dict(color='green')))

# Moon Orbit Path
fig4.add_trace(go.Scatter(x=moon_x, y=moon_z, mode='lines', name='Moon Orbit', line=dict(color='grey', width=1)))

fig4.update_layout(
    title='2D Orbit',
    margin=dict(l=0, r=0, t=40, b=0), # 't' is 40 to keep room for the title
    paper_bgcolor="white",           # The outer background
    plot_bgcolor="white",            # The inner coordinate space
    xaxis=dict(
        showgrid=True,               # Keep grid for coordinate reference
        zeroline=True,               # Keep the x=0 line
        showline=True,              # Hide the outer "box" border
        showticklabels=True          # Keep numbers for context
    ),
    yaxis=dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        scaleanchor="x", scaleratio=1 # Keep circles circular
    )
)
fig4.show()
### PLOTLY PLOTS ###'''