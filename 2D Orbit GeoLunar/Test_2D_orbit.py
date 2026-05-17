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
            ### Special Equation Using beta, if we do this, I won't have to import earth atmosphere density
            #rho = self.rhos*np.exp(-self.beta*altitude)

            ### Simplified Equation
            rho = np.interp(altitude, self.altitude, self.density)
            #print(rho)
        else:
            #np.interp(altitude, self.altitude, self.density)
            print('No such Planet to set rho')
        return rho
aeroModel = Aerodynamics(name)

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
###############################################
### WORLD — gravity, atmosphere, celestial bodies
###############################################
class World:
    def __init__(self):
        self.G = 6.6742e-11
        self.bodies = [
            {'name': 'Earth', 'mass': 5.972e24, 'radius': 6.378e6},
            {'name': 'Moon', 'mass': 7.348e22, 'radius': 1.737e6},
        ]

    def gravity(self, x, z, t):
        ax, az = 0.0, 0.0
        for body in self.bodies:
            bx, bz = self.get_body_position(body, t)
            dx, dz = x - bx, z - bz
            r = np.sqrt(dx ** 2 + dz ** 2)
            if r < body['radius']:
                continue
            a = self.G * body['mass'] / r ** 2
            ax -= a * dx / r
            az -= a * dz / r
        return ax, az

    def get_body_position(self, body, t):
        if body['name'] == 'Earth':
            return 0.0, 0.0
        elif body['name'] == 'Moon':
            return moonOrbit(t)  # your existing function

    def atmosphere(self, x, z):
        r = np.sqrt(x ** 2 + z ** 2)
        alt = r - self.bodies[0]['radius']
        return aeroModel.getDensity(alt)


###############################################
### VEHICLE — stages, mass, thrust
###############################################
class Vehicle:
    def __init__(self, stages, payload):
        self.stages = stages
        self.payload = payload
        self.current_stage = 0

        # Precompute
        self.stage_fuels = np.array([s['m_wet'] - s['m_dry'] for s in stages])
        self.isps = np.array([s['isp'] for s in stages])
        self.thrusts = np.array([s['thrust'] for s in stages])
        self.dry_masses = np.array([s['m_dry'] for s in stages])

    def total_mass(self):
        # Sum of current + upper stages + payload
        total = self.payload
        for i in range(self.current_stage, len(self.stages)):
            total += self.stages[i]['m_wet']
        return total

    def separate_stage(self):
        print(f"Stage {self.current_stage + 1} separated")
        self.current_stage += 1

    def get_thrust(self, throttle, angle_deg):
        if self.current_stage >= len(self.stages):
            return 0.0, 0.0, 0.0
        thrust = self.thrusts[self.current_stage] * throttle
        ve = self.isps[self.current_stage] * 9.81
        mdot = -thrust / ve
        angle = np.radians(angle_deg)
        return thrust * np.cos(angle), thrust * np.sin(angle), mdot


###############################################
### MISSION — phases, guidance, events
###############################################
class Mission:
    def __init__(self, vehicle, world, target_peri, target_apo):
        self.vehicle = vehicle
        self.world = world
        self.target_peri = target_peri
        self.target_apo = target_apo

        # Mission phase sequence
        self.phases = [
            {'name': 'Liftoff', 'mode': 'burn', 'stage': 0, 'end': 'altitude', 'value': 500},
            {'name': 'Ascent', 'mode': 'burn', 'stage': 0, 'end': 'burnout', 'value': None},
            {'name': 'Coast 1', 'mode': 'coast', 'stage': 0, 'end': 'duration', 'value': 10},
            {'name': 'Stage 2', 'mode': 'burn', 'stage': 1, 'end': 'burnout', 'value': None},
            {'name': 'Coast 2', 'mode': 'coast', 'stage': 1, 'end': 'duration', 'value': 30},
            {'name': 'Stage 3 LEO', 'mode': 'burn', 'stage': 2, 'end': 'orbit', 'value': None},
            {'name': 'LEO Coast', 'mode': 'coast', 'stage': 2, 'end': 'duration', 'value': 9000},
            {'name': 'TLI', 'mode': 'burn', 'stage': 2, 'end': 'tli', 'value': None},
        ]
        self.phase_idx = 0

    @property
    def phase(self):
        return self.phases[self.phase_idx]

    def next_phase(self, state, t):
        name = self.phase['name']
        print(f"Phase complete: {name} at t={t:.0f}s")

        # Drop stage if needed
        if self.phase['mode'] == 'burn' and self.phase['end'] == 'burnout':
            self.vehicle.separate_stage()

        self.phase_idx += 1
        if self.phase_idx >= len(self.phases):
            print("Mission complete!")
            return False

        print(f"Starting phase: {self.phase['name']}")
        return True

    def guidance(self, t, state):
        x, z, velx, velz, mass = state
        r = np.sqrt(x ** 2 + z ** 2)
        alt = r - self.world.bodies[0]['radius']
        mode = self.phase['mode']
        name = self.phase['name']

        if mode == 'coast':
            return 0.0, 0.0  # angle irrelevant, throttle 0

        # Guidance by phase name
        if name == 'Liftoff':
            return 0.0, 1.0

        elif name == 'Ascent':
            pitch = np.clip(alt / 150000, 0, 1) * 89.0
            return pitch, 1.0

        elif name in ('Stage 2', 'Stage 3 LEO', 'TLI'):
            return self._peg(state, r, alt)

        return 0.0, 0.0

    def _peg(self, state, r, alt):
        x, z, velx, velz, mass = state
        Rplanet = self.world.bodies[0]['radius']

        r_hat = np.array([x, z]) / r
        t_hat = np.array([-z, x]) / r
        v_rad = np.dot([velx, velz], r_hat)
        v_tan = np.dot([velx, velz], t_hat)

        if self.phase['name'] == 'TLI':
            r_moon = 384400000
            r_target = Rplanet + self.target_peri
            v_target = np.sqrt(self.world.G * 5.972e24 / r_target) * np.sqrt(2 * r_moon / (r_target + r_moon))
        else:
            r_target = Rplanet + self.target_peri
            v_target = np.sqrt(self.world.G * 5.972e24 / r_target)

        dv_rad = -v_rad
        dv_tan = v_target - v_tan
        dv_vec = dv_rad * r_hat + dv_tan * t_hat
        angle = np.degrees(np.arctan2(dv_vec[1], dv_vec[0]))

        dv_needed = np.sqrt(dv_rad ** 2 + dv_tan ** 2)
        throt = 0.0 if dv_needed < 50 else 1.0

        return angle, throt

    def check_phase_end(self, t, state, t_start):
        x, z, velx, velz, mass = state
        r = np.sqrt(x ** 2 + z ** 2)
        alt = r - self.world.bodies[0]['radius']
        end = self.phase['end']
        value = self.phase['value']

        if end == 'altitude':  return alt >= value
        if end == 'duration':  return (t - t_start) >= value
        if end == 'burnout':
            stage = self.phase['stage']
            t_burn = self.vehicle.stage_fuels[stage] * \
                     self.vehicle.isps[stage] * 9.81 / \
                     self.vehicle.thrusts[stage]
            return (t - t_start) >= t_burn
        if end in ('orbit', 'tli'):
            r_hat = np.array([x, z]) / r
            t_hat = np.array([-z, x]) / r
            v_rad = np.dot([velx, velz], r_hat)
            v_tan = np.dot([velx, velz], t_hat)
            v_circ = np.sqrt(self.world.G * 5.972e24 / r)
            return v_tan >= v_circ * 0.99 and abs(v_rad) < 200
        return False


###############################################
### INTEGRATOR — runs the physics
###############################################
class Simulation:
    def __init__(self, mission, vehicle, world):
        self.mission = mission
        self.vehicle = vehicle
        self.world = world
        self.S = 10.1
        self.CD = 0.4

    def derivatives(self, t, state):
        x, z, velx, velz, mass = state
        r = np.sqrt(x ** 2 + z ** 2)

        # Gravity
        ax, az = self.world.gravity(x, z, t)

        # Guidance
        angle, throt = self.mission.guidance(t, state)

        # Thrust
        fx, fz, mdot = self.vehicle.get_thrust(throt, angle)
        ax += fx / mass
        az += fz / mass

        # Drag
        rho = self.world.atmosphere(x, z)
        V = np.sqrt(velx ** 2 + velz ** 2)
        if V > 0:
            drag = 0.5 * rho * V ** 2 * self.CD * self.S
            ax -= drag * velx / (V * mass)
            az -= drag * velz / (V * mass)

        return [velx, velz, ax, az, mdot]

    def hit_ground(self, t, state):
        x, z = state[0], state[1]
        return np.sqrt(x ** 2 + z ** 2) - self.world.bodies[0]['radius'] * 0.999

    hit_ground = property(lambda self: self._make_hit_ground())

    def _make_hit_ground(self):
        def event(t, state):
            x, z = state[0], state[1]
            return np.sqrt(x ** 2 + z ** 2) - self.world.bodies[0]['radius'] * 0.999

        event.terminal = True
        event.direction = -1
        return event

    def run(self, initial_state, max_time=1e6):
        state = np.array(initial_state)
        t_current = 0.0
        all_t, all_y = [], []
        t_phase_start = 0.0

        while self.mission.phase_idx < len(self.mission.phases):
            # Run until phase end condition or max time
            seg = sci.solve_ivp(
                self.derivatives,
                t_span=(t_current, t_current + 99999),
                y0=state,
                t_eval=np.linspace(t_current, t_current + 99999, 5000),
                method='LSODA',
                rtol=1e-6, atol=1e-9,
                events=self._make_hit_ground(),
                max_step=10.0  # prevent solver skipping phase transitions
            )

            # Find where phase ended
            phase_end_idx = len(seg.t) - 1
            for i, (t, s) in enumerate(zip(seg.t, seg.y.T)):
                if self.mission.check_phase_end(t, s, t_phase_start):
                    phase_end_idx = i
                    break

            all_t.append(seg.t[:phase_end_idx + 1])
            all_y.append(seg.y.T[:phase_end_idx + 1])

            if seg.status == 1:
                print("Hit ground!")
                break

            state = seg.y.T[phase_end_idx].copy()
            t_current = seg.t[phase_end_idx]

            if not self.mission.next_phase(state, t_current):
                break

            t_phase_start = t_current

        return np.concatenate(all_t), np.concatenate(all_y, axis=0)


world   = World()
vehicle = Vehicle(STAGES, payload=0)
mission = Mission(vehicle, world, target_peri=220000, target_apo=220000)
sim     = Simulation(mission, vehicle, world)

initial_state = [Rplanet * 1.001, 0, 0, 0, vehicle.total_mass()]
tout, stateout = sim.run(initial_state)

moon_positions = np.array([moonOrbit(t) for t in tout])
moon_x = moon_positions[:, 0]
moon_z = moon_positions[:, 1]

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
### Animation
#'''
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

