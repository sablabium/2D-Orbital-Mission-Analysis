"""
Created on : Sun May 17 12:22 PM 2026

@author: Saba Ghudushauri
"""

import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt

plt.close('all')
np.seterr(all='warn')

### DEFINE CONSTANT PARAMETERS
G = 6.6742e-11 # Gravitational constant (SI unit)
### PLANET

# Earth
radius_Earth = 6.378e6 # meters
mass_Earth = 5.972e24  # kg
mu_Earth = G * mass_Earth
# Moon
radius_Luna = 1737400 # meters
mass_Luna = 7.348e22 # kg
ecc = 0.0549 # eccentricity
luna_per = 363300000  # (m) perigee
luna_apo = 405500000  # (m) apogee
incl = 0.0  # degrees in inclination
orb_pd = 27.322 * 24 * 60 * 60  # (sec) Orbital Period
a = (luna_per + luna_apo) / 2
init_degrees = -65  # Starting degrees of the moon in relation to Earth

### Initial Conditions -- for single stage rocket ###
seconds = 0.0
minutes = 0.0
hours = 0.0
days = 0.0
period = 99999  # *100
t_span = (0, period)

### ROCKET ###
x0 = 0  # m
z0 = radius_Earth + 1  # m
velx0 = 465.1  # m/s Earth linear velocity is around 465.1 at equator
velz0 = 0.0  # m/s

STAGES = [
    {'name': 'Stage 1', 'm_wet': 2214e3, 'm_dry': 13e3, 'thrust': 33e6, 'isp': 250},
    {'name': 'Stage 2', 'm_wet': 470e3, 'm_dry': 43e3, 'thrust': 44e5, 'isp': 420},
    {'name': 'Stage 3', 'm_wet': 1205e2, 'm_dry': 152e2, 'thrust': 1e6, 'isp': 421},
]

stage_count = len(STAGES)
stage_thrusts = np.asarray([s['thrust'] for s in STAGES])
stage_isps = np.asarray([s['isp'] for s in STAGES])
stage_drys = np.asarray([s['m_dry'] for s in STAGES])
stage_fuels = np.asarray([s['m_wet'] for s in STAGES]) - stage_drys

S = 10.1  # Cross Sectional Area of the Rocket
CD = 0.4  # Drag Coefficient

### Ascent Guidance ###
GUIDANCE_ENABLED = True
target_peri = 166 * 1000 # (m) target periapsis
target_apo = 166 * 1000 # (m) target apoapsis
booster_pitch_start = 530 # (m) when does the booster start pitching
final_path_angle = 75.0 # (deg) 75.0
turn_shape = 0.30 # 1 = linear, 2 = steeper towards the end

### TLI Guidance ###
## tli_overshoot = 100000 # (m) difference in target height and luna orbit ( > 0 - overshoot, < 0 undershoot )

## Open loop
ascent_profile = np.linspace(0, target_peri, 1000)
ratio = np.clip(ascent_profile / target_peri, 0, 1)
ascent_path = 90.0 - (np.power(ratio, turn_shape) * final_path_angle)
ascent_path[ascent_profile < booster_pitch_start] = 90.0 # go straight up before we pitch

mission_phases = [
    {'name': 'Liftoff', 'mode': 'burn', 'stage': 0, 'end': 'altitude', 'value': booster_pitch_start},
    {'name': 'Ascent', 'mode': 'burn', 'stage': 0, 'end': 'burnout', 'value': None},
    {'name': 'Stage 2', 'mode': 'burn', 'stage': 1, 'end': 'burnout', 'value': None},
    {'name': 'Stage 3 LEO', 'mode': 'burn', 'stage': 2, 'end': 'orbit', 'value': None},
    ## Coast in LEO until the moon is in the correct angle
    {'name': 'LEO Coast', 'mode': 'coast', 'stage': 2, 'end': 'luna_in_place', 'value': None},
    ## Coast in LEO until the perfect alignment window strikes
    {'name': 'LEO Coast Window', 'mode': 'coast', 'stage': 2, 'end': 'tli_window', 'value': None},
    ## One continuous TLI burn that stops ONLY when target apoapsis matches lunar distance
    {'name': 'TLI Burn', 'mode': 'burn', 'stage': 2, 'end': 'tli', 'value': None},
    {'name': 'TLI Coast', 'mode': 'coast', 'stage': 2, 'end': 'altitude', 'value': luna_per},
    {'name': 'Luna Orbit Coast', 'mode': 'coast', 'stage': 2, 'end': 'duration', 'value': 10000},
]

class Aerodynamics():
    def __init__(self, name):
        self.name = name
        if name == 'Earth':
            data = np.loadtxt('../earth_athmosphere_density.txt')
            self.altitude = data[:, 0]
            self.density = data[:, 1]
            self.rhos = self.density[0]
            self.beta = 0.1354 / 1000.0
        else:
            print('No Such Body Found to set it\'s aero')
    def getDensity(self, altitude):
        if self.name == 'Earth':
            rho = np.interp(altitude, self.altitude, self.density)
        else:
            rho = 0.0
            print('No such Planet to set rho')
        return rho
aeroModel = Aerodynamics('Earth')

def lunaOrbit(t, peri_multiplier=1):
    M = (2 * np.pi * t / orb_pd) + (init_degrees * np.pi / 180.0)
    E = M
    for _ in range(5):
        E = E - (E - ecc * np.sin(E) - M) / (1 - ecc * np.cos(E))

    th = 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2)) * peri_multiplier
    r = a * (1 - ecc ** 2) / (1 + ecc * np.cos(th))

    vx = -np.sqrt(mu_Earth / (a * (1 - ecc ** 2))) * np.sin(th)
    vz = np.sqrt(mu_Earth / (a * (1 - ecc ** 2))) * (ecc + np.cos(th))

    x = r * np.cos(th)
    z = r * np.sin(th)

    return np.asarray([x, z])
class World:
    def __init__(self, ):
        self.bodies = [
            {'name': 'Earth', 'mass': mass_Earth, 'radius': radius_Earth},
            {'name': 'Luna', 'mass': mass_Luna, 'radius': radius_Luna},
        ]
    def gravity(self, x, z, t):
        ax, az = 0.0, 0.0
        for body in self.bodies:
            bx, bz = self.get_body_position(body, t)
            dx, dz = x - bx, z - bz
            r = np.sqrt(dx ** 2 + dz ** 2)
            if r < body['radius']:
                continue
            a = G * body['mass'] / (r ** 2)
            ax += -a * dx / r
            az += -a * dz / r
        return np.asarray([ax, az])
    def get_body_position(self, body, t):
        if body['name'] == 'Earth':
            return 0.0, 0.0
        if body['name'] == 'Luna':
            return lunaOrbit(t)
        return 0.0, 0.0
    def atmosphere(self, x, z):
        r = np.sqrt(x ** 2 + z ** 2)
        alt = r - self.bodies[0]['radius']
        return aeroModel.getDensity(alt)
class Vehicle:
    def __init__(self, stages, payload):
        self.stages = stages
        self.payload = payload
        self.current_stage = 0

        self.fuels = np.array([s['m_wet'] - s['m_dry'] for s in stages])
        self.isps = np.array([s['isp'] for s in stages])
        self.thrusts = np.array([s['thrust'] for s in stages])
        self.dry_masses = np.array([s['m_dry'] for s in stages])

    def total_mass(self):
        total = self.payload
        for i in range(self.current_stage, len(self.stages)):
            total += self.stages[i]['m_wet']
        return total
    def separate_stage(self):
        print(f'Stage {self.current_stage + 1} Separated')
        self.current_stage += 1
    def get_thrust(self, throttle, angle_deg, current_mass):
        if self.current_stage >= len(self.stages):
            return 0.0, 0.0, 0.0

        min_mass = self.payload + self.dry_masses[self.current_stage]
        for i in range(self.current_stage + 1, len(self.stages)):
            min_mass += self.stages[i]['m_wet']

        if current_mass <= min_mass:
            return 0.0, 0.0, 0.0

        thrust = self.thrusts[self.current_stage] * throttle
        ve = self.isps[self.current_stage] * 9.81
        mdot = -thrust / ve
        angle = np.radians(angle_deg)
        return thrust * np.cos(angle), thrust * np.sin(angle), mdot
class Mission:
    def __init__(self, vehicle, world, target_peri, target_apo,
                 arrival_offset_deg=0.0, window_angle_deg=2.5, luna_interception_angle_deg=0.0):
        self.vehicle = vehicle
        self.world = world
        self.target_peri = target_peri
        self.target_apo = target_apo
        ## Dynamic parameter extensions
        self.arrival_offset = np.radians(arrival_offset_deg)
        self.window_angle = np.radians(window_angle_deg)
        self.luna_interception_angle = np.radians(luna_interception_angle_deg)

        self.phases = mission_phases
        self.phase_idx = 0

    @property
    def phase(self):
        return self.phases[self.phase_idx]
    def next_phase(self, state, t):
        name = self.phase['name']
        print(f'Phase complete: {name} at t={t:.0f}s')

        if self.phase['mode'] == 'burn' and self.phase['end'] == 'burnout':
            print('Time of Separation: ', t)
            self.vehicle.separate_stage()

        self.phase_idx += 1
        if self.phase_idx >= len(self.phases):
            print('Mission complete')
            return False

        print(f'Starting phase: {self.phase["name"]}')
        return True
    def _get_tli_angle_error(self, t, state):
        x, z, velx, velz, mass = state

        ## Coordinate tracking vectors
        phi_rocket = np.arctan2(z, x)
        moon_pos = lunaOrbit(t)
        phi_moon = np.arctan2(moon_pos[1], moon_pos[0])

        ## Angular relative geometry sweeping counter-clockwise
        current_phase = (phi_moon - phi_rocket) % (2 * np.pi)

        ## Flight dynamics time prediction to lunar crossing distance
        r = np.sqrt(x ** 2 + z ** 2)
        a_trans = (r + luna_per) / 2.0
        t_trans = np.pi * np.sqrt(a_trans ** 3 / mu_Earth)

        ## Integrate prediction sweeps with custom arrival offset targets
        omega_moon = 2 * np.pi / orb_pd
        delta_theta_moon = omega_moon * t_trans
        target_phase = (np.pi + self.arrival_offset - delta_theta_moon) % (2 * np.pi)

        ## Clean normalization bounds
        error = (current_phase - target_phase + np.pi) % (2 * np.pi) - np.pi
        return error
    def guidance(self, t, state):
        if 'Coast' in self.phase['name']:
            return 0.0, 0.0

        x, z, velx, velz, mass = state
        r = np.sqrt(x ** 2 + z ** 2)
        alt = r - radius_Earth

        if self.phase['name'] in ['Liftoff', 'Ascent']:
            pitch = np.interp(alt, ascent_profile + booster_pitch_start, ascent_path)
            return pitch, 1.0
        ## 'Stage 2' so it actually uses the PEG guidance to get to orbit!
        elif self.phase['name'] in ['Stage 2', 'Stage 3 LEO']:
            return self._peg(state, r, alt)
        elif self.phase['name'] == 'TLI Burn':
            ## a singular uninterrupted burn
            pitch_angle = np.degrees(np.arctan2(velz, velx))
            return pitch_angle, 1.0

        return 0.0, 0.0
    def _peg(self, state, r, alt):
        x, z, velx, velz, mass = state
        r_hat = np.asarray([x, z]) / r
        t_hat = np.asarray([z, -x]) / r

        v_rad = np.dot([velx, velz], r_hat)
        v_tan = np.dot([velx, velz], t_hat)

        r_target = radius_Earth + self.target_peri
        v_target = np.sqrt(G * mass_Earth / r_target)
        v_circ = np.sqrt(G * mass_Earth / r)

        v_deficit = v_target - v_tan
        throttle = np.clip(v_deficit / 50.0, 0.0, 1.0)
        target_v_rad = np.clip((r_target - r) * 0.01, -50, 150)

        base_pitch = np.radians(45.0) * (1.0 - np.clip(v_tan / v_circ, 0, 1))
        pitch_correction = (target_v_rad - v_rad) / 800.0
        pitch_angle = base_pitch + pitch_correction

        min_pitch = np.radians(0.0) if alt < self.target_peri else np.radians(-15.0)
        max_pitch = np.radians(65.0)
        pitch_angle = np.clip(pitch_angle, min_pitch, max_pitch)

        dv_vec_global = np.sin(pitch_angle) * r_hat + np.cos(pitch_angle) * t_hat
        angle = np.degrees(np.arctan2(dv_vec_global[1], dv_vec_global[0]))
        return angle, throttle
    ## MTM
    def _compute_tli_time(self, state):
        ## Hohmann transfer half-period from current orbital radius to luna_per.
        ## This is the expected flight time from TLI ignition to lunar perigee.
        ## Using the vis-viva semi-major axis of the transfer ellipse
        # a_trans = (r_now + luna_per) / 2
        # t_trans = pi * sqrt(a_trans**3 / mu)
        x, z = state[0], state[1]
        r = np.sqrt(x ** 2 + z ** 2)
        a_t = (r + luna_per) / 2.0
        return np.pi * np.sqrt(a_t ** 3 / mu_Earth)
    def _luna_in_place_error(self, t, state):
        ## Signed angular error (rad) between the Moon's predicted angle
        ## at TLI arrival time and the desired luna_interception_angle
        ## Returns 0 when the wait is over (fire TLI now -> Moon at interception)
        t_trans = self._compute_tli_time(state)  # dynamic flight time
        moon_pos = lunaOrbit(t + t_trans)  # Moon pos at arrival
        arrival_angle = np.arctan2(moon_pos[1], moon_pos[0])
        target = self.luna_interception_angle
        ## Normalised to (−pi, pi) so solve_ivp sees a clean zero crossing
        return (arrival_angle - target + np.pi) % (2.0 * np.pi) - np.pi
    ## Called if gatekeep
    def check_phase_end(self, t, state, t_start):
        x, z, velx, velz, mass = state
        r = np.sqrt(x ** 2 + z ** 2)
        alt = r - radius_Earth
        end = self.phase['end']
        value = self.phase['value']

        if end == 'altitude':
            return alt >= value - 1.0  # 1 meter tolerance
        if end == 'duration':
            return (t - t_start) >= value - 1e-2
        if end == 'burnout':
            stage = self.phase['stage']
            t_burn = self.vehicle.fuels[stage] * self.vehicle.isps[stage] * 9.81 / self.vehicle.thrusts[stage]
            return (t - t_start) >= t_burn - 1e-2
        if end == 'orbit':
            t_hat = np.asarray([z, -x]) / r
            v_tan = np.dot([velx, velz], t_hat)
            v_circ = np.sqrt(G * mass_Earth / r)
            return v_tan >= v_circ - 0.1  # 0.1 m/s tolerance
        if end == 'tli_window':
            error = self._get_tli_angle_error(t, state)
            ## Add a microscopic margin to prevent float rejection
            return abs(error) <= self.window_angle + 1e-4
        if end == 'tli':
            V2 = velx ** 2 + velz ** 2
            energy = (V2 / 2.0) - (G * mass_Earth / r)
            if energy >= 0:
                return True
            a_orbit = - (G * mass_Earth) / (2.0 * energy)
            r_apo = (2.0 * a_orbit) - r
            return r_apo >= luna_per + 0  # tolerance
        if end == 'luna_in_place':
            ## continuous angular error with dynamic TLI time
            error = self._luna_in_place_error(t, state)
            return abs(error) < np.radians(2.0)   # +-2ddegrees to coarse tolerance
        return False


guidance_log = []


class Simulation:
    def __init__(self, mission, vehicle, world):
        self.mission = mission
        self.vehicle = vehicle
        self.world = world
        self.t_phase_start = 0.0
    def derivatives(self, t, state):
        x, z, velx, velz, mass = state

        if mass <= 1.0:
            return np.asarray([velx, velz, 0.0, 0.0, 0.0])

        ax, az = self.world.gravity(x, z, t)
        angle, throt = self.mission.guidance(t, state)
        fx, fz, mdot = self.vehicle.get_thrust(throt, angle, mass)

        ax += fx / mass
        az += fz / mass

        rho = self.world.atmosphere(x, z)
        V = np.sqrt(velx ** 2 + velz ** 2)
        if V > 0:
            drag = 0.5 * rho * V ** 2 * CD * S
            ax -= drag * velx / (V * mass)
            az -= drag * velz / (V * mass)

        guidance_log.append([t, angle, throt])
        return np.asarray([velx, velz, ax, az, mdot])
    def _make_hit_ground(self):
        def event(t, state):
            x, z = state[0], state[1]
            r = np.sqrt(x ** 2 + z ** 2)
            return r - radius_Earth

        event.terminal = True
        event.direction = -1
        return event
    def _make_phase_end_event(self):
        end = self.mission.phase['end']

        def event(t, state):
            x, z, velx, velz, mass = state
            r = np.sqrt(x ** 2 + z ** 2)
            alt = r - radius_Earth
            value = self.mission.phase['value']

            if end == 'altitude':
                return alt - value
            if end == 'duration':
                return (t - self.t_phase_start) - value
            if end == 'burnout':
                stage = self.mission.phase['stage']
                t_burn = self.vehicle.fuels[stage] * self.vehicle.isps[stage] * 9.81 / self.vehicle.thrusts[stage]
                return (t - self.t_phase_start) - t_burn
            if end == 'orbit':
                t_hat = np.asarray([z, -x]) / r
                v_tan = np.dot([velx, velz], t_hat)
                v_circ = np.sqrt(G * mass_Earth / r)
                return v_tan - v_circ
            if end == 'tli':
                V2 = velx ** 2 + velz ** 2
                energy = (V2 / 2.0) - (G * mass_Earth / r)
                if energy >= 0:
                    return 0.0
                a_orbit = - (G * mass_Earth) / (2.0 * energy)
                r_apo = (2.0 * a_orbit) - r
                return r_apo - luna_per
            if end == 'tli_window':
                error = self.mission._get_tli_angle_error(t, state)
                return self.mission.window_angle - abs(error)
            if end == 'luna_in_place':
                return self.mission._luna_in_place_error(t, state)
            return 1.0

        event.terminal = True
        event.direction = 1
        return event
    def run(self, initial_state, max_time=1e8):
        state = np.asarray(initial_state)
        t_current = 0.0
        all_t, all_y = [], []
        self.t_phase_start = 0.0

        while self.mission.phase_idx < len(self.mission.phases):
            print(f"\n--- Phase Evaluation: {self.mission.phase['name']} ---")

            if self.mission.check_phase_end(t_current, state, self.t_phase_start):
                print(f"[Gatekeeper] Phase target already met at t={t_current:.1f}s. Skipping solver step.")
                old_stage = self.vehicle.current_stage
                if not self.mission.next_phase(state, t_current):
                    break
                if self.vehicle.current_stage > old_stage:
                    state[4] -= self.vehicle.dry_masses[old_stage]
                self.t_phase_start = t_current
                continue

            sol = sci.solve_ivp(
                self.derivatives,
                t_span=(t_current, t_current + period),
                y0=state,
                t_eval=None,
                method="LSODA",
                rtol=1e-6, atol=1e-9,
                events=[self._make_hit_ground(), self._make_phase_end_event()],
                max_step=5.0,
            )

            all_t.append(sol.t)
            all_y.append(sol.y.T)

            if sol.t_events[0].size > 0:
                print("Hit Ground! Terminating Simulation.")
                break

            state = sol.y.T[-1].copy()
            t_current = sol.t[-1]

            if self.mission.check_phase_end(t_current, state, self.t_phase_start):
                old_stage = self.vehicle.current_stage
                if not self.mission.next_phase(state, t_current):
                    break
                if self.vehicle.current_stage > old_stage:
                    state[4] -= self.vehicle.dry_masses[old_stage]
                self.t_phase_start = t_current
            else:
                print(f"  [Notice] solver reached period limit at t={t_current:.0f}s. Continuing same phase...")

        return np.concatenate(all_t), np.concatenate(all_y, axis=0)


################## MAIN #################

def main():
    world = World()
    vehicle = Vehicle(STAGES, payload=0)

    # Customized targeting configuration passed directly right here:
    mission = Mission(
        vehicle,
        world,
        target_peri=target_peri,
        target_apo=target_apo,
        arrival_offset_deg=-0.5,  # Adjust offset interception point angle relative to Moon position
        window_angle_deg=1.0,  # Total tolerance window margin around the target ignition geometry
        luna_interception_angle_deg=15.0,
    )

    sim = Simulation(mission, vehicle, world)

    initial_state = [x0, z0, velx0, velz0, vehicle.total_mass()]
    tout, stateout = sim.run(initial_state)
    moon_positions = np.array([lunaOrbit(t) for t in tout])
    moon_x = moon_positions[:, 0]
    moon_z = moon_positions[:, 1]

    xout = stateout[:, 0]
    zout = stateout[:, 1]
    altitude = np.sqrt(xout ** 2 + zout ** 2) - radius_Earth
    velxout = stateout[:, 2]
    velzout = stateout[:, 3]
    velout = np.sqrt(velxout ** 2 + velzout ** 2)
    massout = stateout[:, 4]

    print('---------------------------')
    print("Final mass:", massout[-1])
    print("Final altitude:", altitude[-1])
    print("Final time:", tout[-1])

    def output_plots():
        theta = np.linspace(0, 2 * np.pi, 1000)
        xplanet = radius_Earth * np.sin(theta)
        zplanet = radius_Earth * np.cos(theta)
        g_log = np.asarray(guidance_log)
        log_t = g_log[:, 0]
        log_angle = g_log[:, 1]
        log_throt = g_log[:, 2]

        ### MATPLOTLYB PLOTS ###
        def plt_altitude():
            #global tout, altitude
            plt.figure(1)
            plt.plot(tout, altitude)
            plt.title('Height vs Time')
            plt.xlabel('Time (sec)')
            plt.ylabel('Height (m)')
            plt.grid()
        def plt_speed():
            plt.figure(2)
            plt.plot(tout, velout)
            plt.title('Speed vs Time')
            plt.xlabel('Time (sec)')
            plt.ylabel('Total Speed (m)')
            plt.grid()
        def plt_mass():
            plt.figure(3)
            plt.plot(tout, massout)
            plt.title('Mass vs Time')
            plt.xlabel('Time (sec)')
            plt.ylabel('Total Mass (kg)')
            plt.grid()
        def plt_orbit2D():
            xmoon = radius_Luna * np.sin(theta) + moon_x[-1]
            zmoon = radius_Luna * np.cos(theta) + moon_z[-1]
            plt.figure(4)
            plt.title('2D Orbit')
            ### Earth
            plt.plot(xplanet, zplanet, 'bo', markersize=1, label='Planet')
            ### ANIMATION
            ### Rocket
            plt.plot(xout, zout, 'r-', label='Orbit')
            ### Moon
            plt.plot(xmoon, zmoon, 'ko', markersize=1, label='Moon')
            plt.plot(moon_x, moon_z, 'k-', linewidth=1, label='Moon Orbit')
            plt.grid()
            plt.legend()
            plt.axis('equal')
        def plt_angle():
            plt.figure(5)
            plt.plot(log_t, log_angle)
            plt.title('Angle vs Time')
            plt.xlabel('Time (sec)')
            plt.ylabel('Angle (deg)')
            plt.grid()
        def plt_throttle():
            plt.figure(6)
            plt.plot(log_t, log_throt)
            plt.title('Throttle vs Time')
            plt.xlabel('Time (sec)')
            plt.ylabel('Throttle')
            plt.grid()
        def plt_air_density():
            test_altitude = np.linspace(0, 100000, 100)
            test_rho = aeroModel.getDensity(test_altitude)
            plt.figure(7)
            plt.plot(test_altitude, test_rho, 'b-')
            plt.xlabel('altitude (m)')
            plt.ylabel('air density (kg/m**3)')
            plt.grid()
        def plt_orbit2D_animation():
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
            ax.set_xlim(-a * 1.2, a * 1.2)  # Scale based on Moon's semi-major axis
            ax.set_ylim(-a * 1.2, a * 1.2)

            # Create the visual elements
            earth_plot = ax.plot(xplanet, zplanet, 'bo', markersize=1, label='Earth')[0]
            moon_plot, = ax.plot([], [], 'ko', markersize=5, label='Moon')
            moon_trail, = ax.plot([], [], 'ko', markersize=1)  # Moon Path
            rocket_plot, = ax.plot([], [], 'r-', label='Rocket Path')
            rocket_dot, = ax.plot([], [], 'ro', markersize=3)  # Current rocket position

            step = 300

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
            return ani

        # plt_air_density()
        # plt_altitude()
        # plt_speed()
        # plt_mass()
        # plt_angle()
        # plt_throttle()
        plt_orbit2D()
        plt_orbit2D_animation()

        plt.show()
    output_plots()

if __name__ == '__main__':
    main()