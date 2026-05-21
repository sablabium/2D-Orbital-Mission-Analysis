# Lunar Transfer Orbital Simulation
A 2D Earth–Moon orbital **flight simulation** written in Python using numerical integration and event-driven mission phases.

This project simulates a multi-stage launch vehicle from liftoff, orbital insertion, translunar injection (TLI), and lunar transfer coast using custom physics, guidance, and mission sequencing systems.

<img width="1536" height="754" alt="Figure_5" src="https://github.com/user-attachments/assets/68e6a08a-bb64-48b8-a704-30d2e235ce09" />

# Overview
This simulator was built to explore _orbital mechanics, launch guidance, and translunar mission planning_ from the ground up without relying on external aerospace simulation engines.

**The simulation includes:**

- Multi-stage launch vehicle dynamics
- Earth and Moon gravity
- Atmospheric drag
- Open-loop ascent guidance
- PEG-inspired orbital insertion guidance
- Translunar Injection (TLI)
- Dynamic launch window targeting
- Event-driven mission phases
- Numerical orbit propagation using SciPy

The project started as a launch/orbit sandbox and gradually evolved into a larger _orbital mission framework._

# Main Features

<img width="800" height="800" alt="animation" src="https://github.com/user-attachments/assets/c9a3e91f-a3d6-4d15-9376-087a72c924f4" />

## Vehicle Simulation

- Multi-stage rocket support
- Wet/dry mass tracking
- Stage separation
- Variable thrust and ISP per stage
- Mass flow simulation

```text
STAGES = [
    {'name': 'Stage 1', 'm_wet': 2214e3, 'm_dry': 13e3, 'thrust': 33e6, 'isp': 250},
    {'name': 'Stage 2', 'm_wet': 470e3, 'm_dry': 43e3, 'thrust': 44e5, 'isp': 420},
    {'name': 'Stage 3', 'm_wet': 1205e2, 'm_dry': 152e2, 'thrust': 1e6, 'isp': 421},
]
```

<img width="1536" height="754" alt="ascent_1" src="https://github.com/user-attachments/assets/882f6bea-1fd3-48de-827a-bbd6a82b915e" />

## Orbital Mechanics

- Newtonian gravity
- Two-body and Earth–Moon interactions
- Elliptical Moon orbit
- Orbital energy calculations
- Apoapsis/periapsis estimation

<img width="100%" alt="Orbit" src="https://github.com/user-attachments/assets/605a19fc-ba16-4e05-8119-aea7e48140af" />

<img width="100%" alt="luna_orbit" src="https://github.com/user-attachments/assets/d0c55801-4f03-4150-bc9e-bffe0cc47951" />



## Atmospheric Model

- Atmospheric density interpolation from real data
- Aerodynamic drag calculations
- Dynamic velocity losses during ascent

<img width="640" height="480" alt="Figure_7" src="https://github.com/user-attachments/assets/dabbaa2b-d355-4167-af7e-a0f488247a72" />

# Guidance System

## Ascent Guidance

The launch vehicle performs a gravity turn using configurable pitch shaping. With parameters for _pitch start, ascent path, final booster angle and such._

## Orbital Insertion
A **PEG**-inspired guidance mode attempts to circularize into Low Earth Orbit.

## Translunar Injection (TLI)
The simulation computes launch timing windows and performs a continuous `prograde burn` until the transfer trajectory reaches lunar distance.

<img width="1918" height="1017" alt="ascent_h_s_m" src="https://github.com/user-attachments/assets/5233d4d0-04dd-46c8-aadd-235186c962c2" />

# Development Notes

Early mission planning, transfer geometry calculations, and guidance experimentation were initially prototyped on whitebaord/paper before implementation.

<table width="100%">
  <tr>
    <td width="50%" align="left">
      <img width="100%" alt="Whiteboard2" src="https://github.com/user-attachments/assets/3a2d8f03-d541-45fa-a2ee-c0aa8e57d31d" />
  </td>
    <td width="50%" align="right">
      <img width="100%" alt="Whiteboard1" src="https://github.com/user-attachments/assets/c761d536-d02b-43c9-a629-addf26e3ac8e" />
    </td>
  </tr>
</table>

## Mission Sequencing
The mission is organized into _flexible event-driven_ phases, which can be freely modified:
- Liftoff
- Ascent
- Stage Separation
- Orbital Insertion
- Parking Orbit Coast
- TLI Window Alignment
- Translunar Injection Burn
- Lunar Transfer Coast

```text
mission_phases = [
    {'name': 'Liftoff', 'mode': 'burn', 'stage': 0, 'end': 'altitude', 'value': booster_pitch_start},
    {'name': 'Ascent', 'mode': 'burn', 'stage': 0, 'end': 'burnout', 'value': None},
    {'name': 'Stage 2', 'mode': 'burn', 'stage': 1, 'end': 'burnout', 'value': None},
    {'name': 'Stage 3 LEO', 'mode': 'burn', 'stage': 2, 'end': 'orbit', 'value': None},
    {'name': 'Coast', 'mode': 'coast', 'stage': 2, 'end': 'duration', 'value': 4000},
    {'name': 'LEO Coast', 'mode': 'coast', 'stage': 2, 'end': 'luna_in_place', 'value': None},
    {'name': 'LEO Coast Window', 'mode': 'coast', 'stage': 2, 'end': 'tli_window', 'value': None},
    {'name': 'TLI Burn', 'mode': 'burn', 'stage': 2, 'end': 'tli', 'value': None},
    {'name': 'TLI Coast', 'mode': 'coast', 'stage': 2, 'end': 'altitude', 'value': luna_per},
    {'name': 'Luna Orbit Coast', 'mode': 'coast', 'stage': 2, 'end': 'duration', 'value': 40000},
]
```

## Physics Model
### Main Equations used
The simulator numerically integrates motion using:\
`F = m * a`

and Newtonian gravity:\
`F = G * ( m1 * m2 / r ^ 2 )`

Orbital Velocity / Vis-Viva Equation"\
`v = sqrt( mu * (2/r - 1/a))`

Kepler’s Third Law:\
`T = 2pi * sqrt( a^3/mu )`

Rocket Equation (Tsiolkovsky):\
`dv = Isp * g0 * ln( m0/mt )`

The equations of motion are propagated using:\
`scipy.integrate.solve_ivp`\
`LSODA adaptive solver`

## Technologies Used
- Python
- NumPy
- SciPy
- Matplotlib

# Example Outputs
## 2D Earth–Moon Transfer
<img width="1536" height="754" alt="1" src="https://github.com/user-attachments/assets/2005fbcd-18e3-4ed0-9f9b-b1c90c2958f8" />

## Altitude Profile
<img width="1536" height="754" alt="4" src="https://github.com/user-attachments/assets/57490995-5390-4bd1-9f09-b5bac76e4a06" />
<img width="1536" height="754" alt="2" src="https://github.com/user-attachments/assets/b80963d0-3d68-429b-b732-21cf3a30e4bc" />

## Speed Profile
<img width="1536" height="754" alt="5" src="https://github.com/user-attachments/assets/3b198e75-d025-4a25-9c78-f90750a300cc" />

## Mass Profile
<td><img width="1536" height="754" alt="7" src="https://github.com/user-attachments/assets/296adce4-7b23-4fda-b157-a24ee7e6560a" />
<td><img width="1536" height="754" alt="6" src="https://github.com/user-attachments/assets/473e106a-edf5-4dae-83c9-7361450a8032" />

## Guidance Angles
<img width="1536" height="754" alt="8" src="https://github.com/user-attachments/assets/3819f7cb-25c7-478e-99ee-c9eb583b599c" />
<img width="1536" height="754" alt="9" src="https://github.com/user-attachments/assets/d6e5d01c-fc7a-400e-baba-cef4de077a00" />

## Project Structure
```text
orbital_simulation/
├── main.py
├── earth_atmosphere_density.txt
└── README.md
```

Future versions will split the project into:
- physics modules
- guidance systems
- targeting utilities
- visualization tools
- mission configuration files

# Current Limitations
This project is still under active development and currently uses several approximations:
- Simplified 2D orbital plane
- No true patched-conic sphere-of-influence transitions
- Simplified Moon transfer targeting
- No full n-body propagation
- Guidance algorithms are still experimental
- Lunar orbit insertion is not yet implemented

Despite these simplifications, the simulation produces stable launch, orbit insertion, and translunar transfer behavior suitable for experimentation and visualization.

# Future Improvements
Planned upgrades include:
- Patched conics
- Lunar sphere-of-influence transitions
- Real lunar capture/orbit insertion
- Improved PEG guidance
- Orbital element utilities
- Config system and GUI launcher
- Better telemetry and plotting
- Real-time visualization
- 3D simulation support
- Numerical optimization for transfer targeting

# Running the Simulation
## Install dependencies
`pip install numpy scipy matplotlib`
## Run
`python Main.py`

# Why I Built This
This project was created as a personal engineering and physics challenge to better understand:
- orbital mechanics
- launch dynamics
- numerical integration
- spacecraft guidance
- mission sequencing
- aerospace simulation architecture

Most of the systems were implemented manually from first principles rather than using existing aerospace frameworks.

# References / Resources
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
- https://icas.org/icas_archive/ICAS2012/PAPERS/611.PDF
- https://ntrs.nasa.gov/api/citations/20250008278/downloads/PEG_ASC25_Mahajan.pdf
- https://www.mdpi.com/2226-4310/12/1/61
- https://www.youtube.com/watch?v=HE3a_bmEHyo
- https://www.youtube.com/watch?v=9mmmHaOfyco
