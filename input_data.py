# Simulation mode
# sim_mode = 1 => Constant density in atmosphere (convection controlled by ballistic path. Velocity = 0 when the ground is reached)
# sim_mode = 2 => Variable density in atmosphere (convection controlled by ballistic path. Velocity = 0 when the ground is reached)
# sim_mode = 3 => Variable density in atmosphere (convection controlled by ballistic path. Velocity = 0 when the ground is reached). Set of simulations.
sim_mode = 2

# Simulation topography (if it is absent, sim_top = 1).
# sim_top = 1 => Considering Hi and Hf.
# sim_top = 2 => Considering an input topography (input_topography.py, see example. It must start with 0).
sim_top = 2

# Radius of the particle (m). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
r = 0.10

# Initial temperature of the particle (K). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
T0 = 1073.15

# Initial velocity of the particle (m/s). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
u0 = 100

# Initial angle of the particle (deg). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
ang0 = 40

# Glass transition temperature (K) (if it is absent, Tt = 1073.15). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
Tt = 873.15

# Initial height (m). Required if sim_top = 1.
Hi = 0.0

# Final height (m) (if it is absent, Hf = 0). Required for sim_top = 1.
Hf = 0.0

# Initial volume fraction of bubbles. Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
alpha_g_0 = 0.01

# Initial mass fraction of water (exsolved and dissolved). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
x_dg = 0.01

# Initial volume fraction of crystals in phase 1. Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
alpha_c_0 = 0.10

# Initial pressure of the particle (Pa). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
p0 = 101325.0

# Reference temperature for melt density (K).
T_r2 = 1273.15

# Reference melt density (kg/m3) (at temperature T_r2).
rho_r = 2200.0

# Coefficient of thermal expansion of melt (1/K).
beta = 1e-4

# Density of crystals (kg/m3),
rho_c = 2400.0

# Maximum vesicularity allowed for resulting bomb (if it is absent, phi_m = 0.7)
phi_m = 0.7

# Strength of the ballistic projectile (MPa) (if it is absent, p_failure = 10.0).  Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
p_failure = 10.0

# Thermal diffusivity of the ballistic projectile (m2/s) (if they are absent, k = 3.0e-7).
k = 3.0e-7

# Thermal conductivity of the ballistic projectile (W/mK) (if they are absent, kb = 0.10).
kb = 0.10

# Black body emmisivity (W/m2K4) (if it is absent, gamma = 5.669e-8).
gamma = 5.669e-8

# Atmosphere density (kg/m3), if it is constant (if it is absent, rho_atm = 1.293). Required for sim_mode = 1.
rho_atm = 1.293

# Sea-level atmosphere density (kg/m3) (if it is absent, rho_atm = 1.293). Required for sim_mode = 2 and 3.
rho_0 = 1.293

# Atmospheric scale-height (m) (if it is absent, H_atm = 8200). Required for sim_mode = 2 and 3.
H_atm = 8200

# Heat capacity of air (J/kgK) (if it is absent, Cp = 1005.0).
Cp = 1005.0

# Viscosity of air when T = 273.15 K (kg/ms) (if it is absent, mu = 1.7e-5 ).
mu_r = 1.7e-5

# Expected Prandtl number (if it is absent, Pr = 0.71).
Pr = 0.71

# Conductivity of air (W/mK) (if it is absent, ka = 0.024).
ka = 0.024

# Air temperature (K) (if it is absent, Tinf = 273.15).
Tinf = 273.15

# Relaxation parameter for exsolution (s). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
relax_exsolution = 1e2

# Relaxation parameter for gas expansion (s). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
relax_pressure = 1e0

# Relaxation parameter for crystallization (s). Fixed for sim_mode = 1 and 2. Mean value for sim_mode = 3.
relax_crystallization = 1e3

# Spacial steps in the particle (if it is absent, N = 20).
N = 100

# Plots trajectory (1: Yes, 0: No).  Required if sim_mode = 1 and 2.
plot_trajectory = 1

# Plots temperature evolution (1: Yes, 0: No). Required if sim_mode = 1 and 2.
plot_temperature = 1

# Plot heat flux (1: Yes, 0: No).  Required if sim_mode = 1 and 2.
plot_heat = 1

# Plot rind growth (1: Yes, 0: No).  Required if sim_mode = 1 and 2.
plot_rind = 1

# Plot bubble and crystal dynamics (1: Yes, 0: No).  Required if sim_mode = 1 and 2.
plot_bubbles = 1

# Variables for sim_mode = 3
# Number of simulations
Nsim = 3

# Variability of radius of the particle (m).
r_var = 0.02

# Variability of initial temperature of the particle (K).
T0_var = 0.0

# Variability of initial velocity of the particle (m/s).
u0_var = 0.0

# Variability unitial pressure of the particle (Pa).
p0_var = 0.0

# Variability of initial angle of the particle (deg).
ang0_var = 0.0

# Variability of glass transition temperature (K).
Tt_var = 0.0

# Variability of initial volume fraction of bubbles.
alpha_g_0_var = 0.0

# Variability of initial mass fraction of water.
x_dg_var = 0.0

# Variability of initial volume fraction of crystals in phase 1.
alpha_c_0_var = 0.0

# Variability of strength of the ballistic projectile (MPa).
p_failure_var = 0.0

# Relaxation parameter for exsolution (s).
relax_exsolution_var = 0.0

# Relaxation parameter for gas expansion (s).
relax_pressure_var = 0.0

# Relaxation parameter for crystallization (s).
relax_crystallization_var = 0.0
