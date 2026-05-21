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
Rplanet = 6.378e6*1 # meters
mplanet = 5.972e24*1 # kg
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
init_degrees = 62 # Starting degrees of the moon in relation to Earth

### Initial Conditions -- for single stage rocket ###
seconds = 0.0
minutes = 0.0
hours = 0.0
days = 0.0
period = 1000*90 #*100
t_span = (0, period)
t_eval = np.linspace(0, period, int(period/100)) #int(period/100)

### ROCKET ###
x0 = Rplanet# m
z0 = 0 # m
velz0 = 0.0 # m/s
velx0 = 0.0 # m/s

### TODO: right now thrust of booster is at sealevel and others at vacuum, have to add the formula. Dynamic Pressure Q
### Mass Wet is the full mass of the stage only, Not the full top part. Burn Time is Max Burn Time
### Static ###
STAGES = [
    {'name': 'Stage 1', 'm_wet': 2214e3,'m_dry': 13e3,  'thrust': 33e6, 'isp': 250},
    {'name': 'Stage 2', 'm_wet': 470e3, 'm_dry': 43e3,  'thrust': 44e5, 'isp': 420},
    #{'name': 'Stage 3', 'm_wet': 1205e2,'m_dry': 152e2, 'thrust': 1e6,  'isp': 421},
]
# Calculate burn times based on parameters
stage_count =   len(STAGES)
stage_thrusts = np.asarray([s['thrust'] for s in STAGES])
stage_isps =    np.asarray([s['isp'] for s in STAGES])
stage_drys =    np.asarray([s['m_dry'] for s in STAGES])
stage_fuels =   np.asarray([s['m_wet'] for s in STAGES]) - stage_drys
### Calculating Max Burn Time For Each Stage
burn_times =    (stage_fuels * stage_isps * 9.81 / stage_thrusts) + 0.75 # Burn Time Formula (fuel(kg)/(thrust(N)/(isp(s)*G))
#mmm = stage_fuels/(stage_thrusts/stage_isps*9.81)
#burn_times = mmm
#print('Each Stage Burn Times Are: ', burn_times)

# Calculate cumulative mass drop times
burn_times_cumull = np.cumsum(burn_times)
#print(burn_times_cumull)
# Calculate cumulative mass wet
full_weights = np.cumsum(np.asarray([s['m_wet'] for s in STAGES])[::-1])[::-1]
mass0 = np.sum(s['m_wet'] for s in STAGES) # kg Initial mass - All the Rocket - WET

print("Delta-V budget per stage:")
for i, s in enumerate(STAGES):
    dv = stage_isps[i] * 9.81 * np.log(full_weights[i] / (full_weights[i] - stage_fuels[i]))
    print(f"Stage {i}: {dv:.0f} m/s")
TWR = stage_thrusts[0] / (mass0 * 9.81)
print(f"TWR: {TWR:.2f}")

# TODO: different stages different S
S = 10.1 # Cross Sectional Area of the Rocket
CD = 0.4 # Drag Coefficient

### Ascent Guidance ###
## Set Parameters
GUIDANCE_ENABLED = True
target_peri = 220 * 1000 # (m) target periapsis
target_apo = 220 * 1000 # (m) target apoapsis

booster_pitch_start = 40 # (sec) when does the booster start pitching

pitch_rate = 1.0 # degree/sec

coast_stage: int = 2 # Coast Before Burning Stage At Given Number {0:0,1:10,2:30}
refresh_rate = 2 # In how many seconds should we be refreshing calculations
## Closed Loop Calculations, calculates trajectory at the beginning

## Open Loop - Calculates Every Set Step

### Aerodynamics Class
class Aerodynamics():
    def __init__(self, name):
        self.name = name
        if name == 'Earth':
            ### import aeromodel for earth
            data = np.loadtxt('../earth_athmosphere_density.txt')
            self.altitude = data[:,0]
            self.density = data[:,1]
            self.rhos = self.density[0] #1.225  # kg/m**3
            self.beta = 0.1354 / 1000.0  # density constant for earth
        else:
            print('No Such Body Found to set it\'s aero')
    def getDensity(self, altitude):
        if self.name == 'Earth':
            ### Special Equation Using beta, if we do this, I won't have to import earth atmosphere density
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

    ### Distance from the 0,0, Need to offset it by earth location if I add its movement
    r = np.sqrt(x**2 + z**2)
    #if r <= Rplanet:
        #print('STOP THE FALL')
        #return np.asarray([0.0, 0.0]), r

    accel_mag = G * mplanet / (r ** 2) ## Magnitude of Earth Gravity

    accelx = -accel_mag * x/r
    accelz = -accel_mag * z/r

    return np.asarray([accelx, accelz]), r

### Ascent Guidance, Funnel through which we decide how to call propulsion
## We can split it into closed(store calculations in an array) and open loops.
## Key Formulas:
## dVneeded = dVorbit + dVgravity + dVdrag + dVsteering. How much delta V we need to get into desired orbit
## dVactual = dVideal - dVorbit - dVgravity - dVdrag - dVsteering. How much delta V we really have
## dy/dt = (g/V - V/r) * cosy. y is the target angle. Most Optimal Flight Path.
## dv = Isp * g0 * ln(m0/mf)

### dv/dt = T(t)/m(t) + g(r) + D(r,v)/m(t). Fundamental Vector Differential Equation
### dv/dt =
## v is velocity vector of rocket,
## T(t) is engine Thrust vector
## m(t) - instantaneous vehicle mass
## g(r) - local gravity vector
## D(r,v) - aerodynamic drag force vector
liftoff = True
gravity_assist = False
orbital_insertion = False
leo_coast = False
translunar_injection = False
lunar_orbit_insertion = False
transearth_injection = False

def flight_phase():
    ## Phase 1: Liftoff         - clearing tower
    ## Phase 2: Gravity turn    - following velocity vector through atmosphere
    ## Phase 3: PEG stage 1     - orbit injection
    ## Phase 4: Coast LEO       - no thrust, verify orbit
    ## Phase 5: PEG stage 2     - circularize if needed
    ## Phase 6: Coast LEO       - wait for TLI window
    ## Phase 7: PEG TLI         - Burn For Lunar Intercept
    ## Phase 8: Coast to moon   - Wait till we get to moon
    ## Phase 9: LOI             - Lunar Orbit Injection
    ...

### Powered Explicit Guidance - PEG
guidance_log = []
def peg_guidance(t, state, isp, dry_mass) -> tuple[float, float]:
    ### Returns Angle and Throttle ( 0 - 100 )
    global target_apo, target_peri, booster_pitch_start, \
        liftoff, gravity_assist, orbital_insertion, leo_coast, translunar_injection, lunar_orbit_insertion, transearth_injection

    x,z,velx,velz,mass = state
    ## Updating Values
    r = np.sqrt(x**2+z**2)
    altitude = r - Rplanet

    angle = 0.0
    throt: float = 0.0
    coast_start_t = 0.0
    ### liftoff
    if liftoff:
        angle = 0.0
        throt = 1.0
        if altitude > 120:
            angle = 2.0
        if altitude > 350:
            print("stop liftoff")
            gravity_assist = True
            liftoff = False

    ### pitching
    elif gravity_assist:
        pitch_fraction = np.clip(altitude / (target_peri/2), 0, 1)
        print(f'-------{pitch_fraction}---------')
        angle = pitch_fraction * 90.0
        throt = 1.0

        r_hat = np.array([x, z]) / r
        t_hat = np.array([-z, x]) / r
        v_rad = np.dot([velx, velz], r_hat)
        v_tan = np.dot([velx, velz], t_hat)
        v_circ = np.sqrt(G * mplanet / r)

        # Hand off to PEG when high enough and moving mostly horizontal
        if altitude > 50000 and v_tan > 3000:
            orbital_insertion = True
            gravity_assist = False
            print(f"Handing to PEG: alt={altitude / 1000:.0f}km v_tan={v_tan:.0f} v_rad={v_rad:.0f}")

    ### Get on LEO
    elif orbital_insertion:
        throt = 1.0
        r_target = Rplanet + target_peri
        v_target = np.sqrt(G * mplanet / r_target)

        r_hat = np.array([x, z]) / r
        t_hat = np.array([-z, x]) / r
        v_rad = np.dot([velx, velz], r_hat)
        v_tan = np.dot([velx, velz], t_hat)

        dv_rad = -v_rad
        dv_tan = v_target - v_tan

        dv_vec = dv_rad * r_hat + dv_tan * t_hat
        angle = np.degrees(np.arctan2(dv_vec[1], dv_vec[0]))
        print(f'PEG angle={angle:.1f} v_tan={v_tan:.0f} v_target={v_target:.0f} dv_tan={dv_tan:.0f}')

        dv_needed = np.sqrt(dv_rad ** 2 + dv_tan ** 2)
        if dv_needed < 50:
            throt = 0.0
            orbital_insertion = False
            leo_coast = True
            print("ORBIT ACHIEVED")
    ### Wait for the perfect moment
    elif leo_coast:
        throt = 0.0


    ### HOHMANN TRANSFER
    ### trans-lunar injection
    elif translunar_injection:
        pass
    ### lunar orbit insertion
    elif lunar_orbit_insertion:
        pass
    ### trans-earth injection
    elif transearth_injection:
        pass

    guidance_log.append([t, angle, throt])
    return angle, throt

### Thrust Ramps Up, Then Ramps down
def get_thrust_coeff(t_stage, burn_time: float):
    """
    Returns a multiplier (0 to 1) based on time into the burn.
    t_stage: time since this stage started
    burn_time: total planned duration
    """
    # 1. Start-up ramp (e.g., 0.5 seconds to reach full thrust)
    if t_stage < 0.5:
        return t_stage / 0.5

    # 2. Tail-off (e.g., 1.0 second decay at the end)
    if t_stage > burn_time - 1.0:
        # Linear decay for simplicity, or use a cosine/exponential for realism
        time_left = burn_time - t_stage
        return max(0, time_left / 1.0)

    # 3. Steady State
    return 1.0
def propulsion(t, state):
    global stage_thrusts, stage_isps, stage_drys, stage_fuels, stage_count,burn_times,burn_times_cumull

    ### Current stage Index
    idx = np.searchsorted(burn_times_cumull, t)

    ### If no more stages left
    if idx >= stage_count:
        return np.asarray([0.0, 0.0]), 0.0

    angle, throt = peg_guidance(t,state, stage_isps[idx], stage_drys[idx])
    if throt>0:
        # Calculate time relative to the start of THIS stage
        t_prev_stages = burn_times_cumull[idx-1] if idx > 0 else 0
        t_stage = t - t_prev_stages

        coeff = get_thrust_coeff(t_stage, burn_times[idx])

        ## Calculate Thrust
        thrust_mag = stage_thrusts[idx] * coeff
        ve = stage_isps[idx] * 9.81
        mdot = -thrust_mag / ve

        theta = np.deg2rad(angle)
        thrustx = thrust_mag*np.cos(theta)
        thrustz = thrust_mag*np.sin(theta)

        return np.asarray([thrustx, thrustz]), mdot
    else:
        return np.asarray([0.0, 0.0]), 0

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
    print(f"t={t:.3f} x={x:.0f} z={z:.0f} vx={velx:.1f} vz={velz:.1f} m={mass:.0f}")
    mx, mz = moonOrbit(t)

    accel_earth, r = gravity(x,z)

    # 3. Gravity: Moon on Rocket
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
    altitude = r - Rplanet
    rho = aeroModel.getDensity(altitude)
    V = np.sqrt(velx ** 2 + velz ** 2)

    if V > 0:
        drag_mag = 0.5 * rho * V ** 2 * CD * S  # correct dynamic pressure * CD * S
        aeroF = -drag_mag * np.asarray([velx, velz]) / V  # unit vector for direction
    else:
        aeroF = np.array([0.0, 0.0])
    # Drag is an acceleration not a force
    D = aeroF/mass

    ### Thrust
    thrustF, mdot = propulsion(t, state)

    #thrustF, mdot = [0.0,1000.0]
    ax = accel_earth[0] + accel_moon_x + thrustF[0]/mass
    az = accel_earth[1] + accel_moon_z + thrustF[1]/mass
    ### Total
    Forces = [ax,az] + D

    # Compute Acceleration
    if mass > 0:
        ddot = Forces#/mass
    else:
        ddot = [0.0, 0.0]
        mdot = 0.0

    # compute the state dot
    #print('Mass Of the Rocket: ', mass)
    #print(xdot,zdot, ddot[0],ddot[1], mdot)
    statedot = np.asarray([xdot,zdot, ddot[0],ddot[1], mdot])


    return statedot


### Moon Orbit
def moonOrbit(t):
    ### Constant
    mu = G * mplanet

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

### Time Window
#tout = np.linspace(0, period, period) # 1000*15
### Numerical Integration Call ## Using high precision model
def hit_ground(t, state):
    x, z, velx, velz, mass = state
    r = np.sqrt(x**2 + z**2)
    return r - Rplanet* 0.999  # goes to zero when rocket hits surface

hit_ground.terminal = True   # stops the integration
hit_ground.direction = -1


def run_simulation():
    state = np.asarray([x0, z0, velx0, velz0, mass0])

    all_t = []
    all_y = []

    # Time windows: [0 -> end of stage1, stage1 -> end of stage2, ...]
    stage_times = [0.0] + [bt + 2.0 for bt in burn_times_cumull] + [period]

    for i in range(len(stage_times) - 1):
        t_start = stage_times[i]
        t_end = stage_times[i + 1]

        seg = sci.solve_ivp(
            derivatives,
            t_span=(t_start, t_end),
            y0=state,
            # Higher the division less accurate the Simulation
            # Set To None if you don't want Animation
            t_eval=np.linspace(t_start, t_end, max(100, int((t_end - t_start)/20))),
            method='LSODA',
            rtol=1e-7,
            atol=1e-7,
            events=hit_ground
        )

        all_t.append(seg.t)
        all_y.append(seg.y.T)

        # Check if rocket hit the ground
        if seg.status == 1:
            print(f"Hit ground at t={seg.t[-1]:.1f}s")
            break

        # Drop dry mass of the stage that just finished
        # (don't drop the last stage's dry mass)
        state = seg.y[:, -1].copy()
        print(f"Segment {i} end — mass: {state[4]:.1f} kg, t: {seg.t[-1]:.1f}s")
        stage_idx = i  # stage i just finished
        if stage_idx < stage_count - 1:
            state[4] -= stage_drys[stage_idx]
            print(f"  → After sep drop ({stage_drys[stage_idx]:.0f} kg): {state[4]:.1f} kg")
            #print(
                #f"Stage {stage_idx + 1} separated, dropped {stage_drys[stage_idx]:.0f} kg, mass now {state[4]:.0f} kg")

    # Concatenate all segments
    t_out = np.concatenate(all_t)
    y_out = np.concatenate(all_y, axis=0)
    return t_out, y_out


tout, stateout = run_simulation()

# sol = sci.solve_ivp(derivatives, t_span=t_span, y0=stateinitial, t_eval = None, method='LSODA', rtol=1e-3, atol=1e-6, events=hit_ground)
# tout = sol.t
# stateout = sol.y.T
# print('---------------------------')
# print("Status:", sol.status)
# print("Message:", sol.message)
# print("Number of points:", len(sol.t))
# print("Time range:", sol.t[0] if len(sol.t) > 0 else "EMPTY", "to", sol.t[-1] if len(sol.t) > 0 else "EMPTY")
# print("t_events:", sol.t_events)
# print("y shape:", sol.y.shape)

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

print('---------------------------')
print("Final mass:", massout[-1])
print("Final altitude:", altitude[-1])
print("Final time:", tout[-1])
print("Total burn time:", burn_times_cumull[-1])

### Plot

theta = np.linspace(0, 2 * np.pi, 1000)
xplanet = Rplanet * np.sin(theta)
zplanet = Rplanet * np.cos(theta)
### MATPLOTLYB PLOTS ###
### Altitude
def plt_altitude():
    global tout, altitude
    plt.figure(1)
    plt.plot(tout, altitude)
    plt.title('Height vs Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Height (m)')
    plt.grid()
### Velocity
def plt_speed():
    global tout, velout
    plt.figure(2)
    plt.plot(tout, velout)
    plt.title('Speed vs Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Total Speed (m)')
    plt.grid()
### Mass
def plt_mass():
    global tout, massout
    plt.figure(3)
    plt.plot(tout, massout)
    plt.title('Mass vs Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Total Mass (kg)')
    plt.grid()
### 2D Orbit
def plt_orbit2D():
    global zplanet, moon_x, moon_z, theta, xplanet, zplanet

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
### Angle
def plt_angle_throttle():
    global guidance_log
    g_log = np.array(guidance_log)
    log_t     = g_log[:, 0]
    log_angle = g_log[:, 1]
    log_throt = g_log[:, 2]

    plt.figure(5)
    plt.plot(log_t, log_angle)
    plt.title('Guidance Angle vs Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Angle (deg)')
    plt.grid()

    plt.figure(6)
    plt.plot(log_t, log_throt)
    plt.title('Throttle vs Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Throttle')
    plt.grid()
### plot Air Density as a Function of Above Ground Model
def plt_air_density():
    test_altitude = np.linspace(0, 100000,100)
    test_rho = aeroModel.getDensity(test_altitude)
    plt.figure(7)
    plt.plot(test_altitude, test_rho, 'b-')
    plt.xlabel('altitude (m)')
    plt.ylabel('air density (kg/m**3)')
    plt.grid()
### Animation
def plt_orbit2D_animation():
    global xplanet, zplanet, tout, xplanet,zplanet
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

plt_altitude()
plt_speed()
# plt_mass()
plt_orbit2D()
plt_angle_throttle()
# plt_air_density()
# plt_orbit2D_animation()

plt.show()

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