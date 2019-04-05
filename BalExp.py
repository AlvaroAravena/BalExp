from fipy import *
import numpy as np
from math import exp, sqrt, cos, sin, atan, log
import matplotlib.pyplot as plt
import sys

##########################################################################################################################
################################################### MAIN PROGRAM #########################################################
##########################################################################################################################

# INPUT PARAMETERS

file_txt = open('input_data.py')

line = file_txt.readlines()

file_txt.close()

[sim_mode, sim_top, r, T0, u0, ang0, p0] = [-1, -1, -1, -1, -1, -1, -1]

[Tt, Hi, Hf, alpha_g_0, x_dg, alpha_c_0, T_r2] = [1073.15, -1, 0.0, -1, -1, -1, -1]

[rho_r, beta, rho_c, phi_m] = [ -1, -1, -1, 0.7]

[p_failure, k, kb, gamma] = [10.0, 3.0e-7, 0.10, 5.669e-8]

[rho_atm, rho_0, H_atm, Cp, mu_r, Pr, ka, Tinf] = [1.293, 1.293, 8200.0, 1005.0, 1.7e-5, 0.71, 0.024, 273.15]

[relax_exsolution, relax_pressure, relax_crystallization, N] = [1, 1, 1, 20]

[plot_trajectory, plot_temperature, plot_heat, plot_rind, plot_bubbles] = [1, 1, 1, 1, 1]

[Nsim, r_var, T0_var, u0_var, ang0_var, Tt_var, alpha_g_0_var, p0_var] = [10, 0, 0, 0, 0, 0, 0, 0]

[x_dg_var, alpha_c_0_var, p_failure_var, relax_exsolution_var] = [0, 0, 0, 0]

[relax_pressure_var, relax_crystallization_var] = [0, 0]

for i in range(0,len(line)):

	line[i] = line[i].replace('=',' ')

	aux = line[i].split()

	if(len(aux) > 0):
		if( aux[0][0] != '#'):
			if( aux[0] == 'sim_mode'):
				sim_mode = int(aux[1])
			if( aux[0] == 'sim_top'):
				sim_top = int(aux[1])
			if( aux[0] == 'r'):
				r = float(aux[1])
			if( aux[0] == 'T0'):
				T0 = float(aux[1])
			if( aux[0] == 'u0'):
				u0 = float(aux[1])
			if( aux[0] == 'p0'):
				p0 = float(aux[1])
			if( aux[0] == 'ang0'):
				ang0 = float(aux[1])
			if( aux[0] == 'Tt'):
				Tt = float(aux[1])
			if( aux[0] == 'Hi'):
				Hi = float(aux[1])
			if( aux[0] == 'Hf'):
				Hf = float(aux[1])
			if( aux[0] == 'alpha_g_0'):
				alpha_g_0 = float(aux[1])
			if( aux[0] == 'x_dg'):
				x_dg = float(aux[1])
			if( aux[0] == 'alpha_c_0'):
				alpha_c_0 = float(aux[1])
			if( aux[0] == 'T_r2'):
				T_r2 = float(aux[1])
			if( aux[0] == 'rho_r'):
				rho_r = float(aux[1])
			if( aux[0] == 'beta'):
				beta = float(aux[1])
			if( aux[0] == 'rho_c'):
				rho_c = float(aux[1])
			if( aux[0] == 'phi_m'):
				phi_m = float(aux[1])
			if( aux[0] == 'p_failure'):
				p_failure = float(aux[1])
			if( aux[0] == 'k'):
				k = float(aux[1])
			if( aux[0] == 'kb'):
				kb = float(aux[1])
			if( aux[0] == 'gamma'):
				gamma = float(aux[1])
			if( aux[0] == 'rho_atm'):
				rho_atm = float(aux[1])
			if( aux[0] == 'rho_0'):
				rho_0 = float(aux[1])
			if( aux[0] == 'H_atm'):
				H_atm = float(aux[1])
			if( aux[0] == 'Cp'):
				Cp = float(aux[1])
			if( aux[0] == 'mu_r'):
				mu_r = float(aux[1])
			if( aux[0] == 'Pr'):
				Pr = float(aux[1])
			if( aux[0] == 'ka'):
				ka = float(aux[1])
			if( aux[0] == 'Tinf'):
				Tinf = float(aux[1])
			if( aux[0] == 'relax_exsolution'):
				relax_exsolution = float(aux[1])
			if( aux[0] == 'relax_pressure'):
				relax_pressure = float(aux[1])
			if( aux[0] == 'relax_crystallization'):
				relax_crystallization = float(aux[1])
			if( aux[0] == 'N'):
				N = int(aux[1])
			if( aux[0] == 'plot_trajectory'):
				plot_trajectory = int(aux[1])
			if( aux[0] == 'plot_temperature'):
				plot_temperature = int(aux[1])
			if( aux[0] == 'plot_heat'):
				plot_heat = int(aux[1])
			if( aux[0] == 'plot_rind'):
				plot_rind = int(aux[1])
			if( aux[0] == 'plot_bubbles'):
				plot_bubbles = int(aux[1])
			if( aux[0] == 'Nsim'):
				Nsim = int(aux[1])
			if( aux[0] == 'r_var'):
				r_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'T0_var'):
				T0_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'p0_var'):
				p0_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'u0_var'):
				u0_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'ang0_var'):
				ang0_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'Tt_var'):
				Tt_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'alpha_g_0_var'):
				alpha_g_0_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'x_dg_var'):
				x_dg_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'alpha_c_0_var'):
				alpha_c_0_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'p_failure_var'):
				p_failure_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'relax_exsolution_var'):
				relax_exsolution_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'relax_pressure_var'):
				relax_pressure_var = max(float(aux[1]),1e-15)
			if( aux[0] == 'relax_crystallization_var'):
				relax_crystallization_var = max(float(aux[1]),1e-15)

# READING TOPOGRAPHY IF IT IS REQUIRED

if(sim_top == 2):

	file_txt = open('input_topography.py')

	line = file_txt.readlines()

	file_txt.close()

	data_x = []

	data_y = []

	for i in range(0,len(line)):

		if(len(aux) > 0):

			aux = line[i].split()
		
			data_x.append(float(aux[0]))

			data_y.append(float(aux[1]))

	data_x = np.array(data_x)

	data_y = np.array(data_y)

	Hi = data_y[0]

# DEFAULT PARAMETERS
# Gravity
g = 9.81

# Molar weight of water
mw_water = 0.018015

# Gas constant
R_gas = 8.31

# Atmospheric pressure
p_atm = 101325.0

# Reference temperature
T_r = 273.15

# Crystallization factors
aT = -2.39e-3
aw = -21.15
a0 = 3.33
alpha_min = 0.0
alpha_max = 1.0

# Solubility constants
sol_1 = 4.01e-6
sol_2 = 0.5

# Iteration for sim_mode = 3
if( sim_mode == 3 ):

	r_vector = np.random.normal(r, r_var, Nsim)
	T0_vector = np.random.normal(T0, T0_var, Nsim)
	u0_vector = np.random.normal(u0, u0_var, Nsim)
	p0_vector = np.random.normal(p0, p0_var, Nsim)
	ang0_vector = np.random.normal(ang0, ang0_var, Nsim)
	Tt_vector = np.random.normal(Tt, Tt_var, Nsim)
	alpha_g_0_vector = np.random.normal(alpha_g_0, alpha_g_0_var, Nsim)
	x_dg_vector = np.random.normal(x_dg, x_dg_var, Nsim)
	alpha_c_0_vector = np.random.normal(alpha_c_0, alpha_c_0_var, Nsim)
	p_failure_vector = np.random.normal(p_failure, p_failure_var, Nsim)
	relax_exsolution_vector = np.random.normal(relax_exsolution, relax_exsolution_var, Nsim)
	relax_pressure_vector = np.random.normal(relax_pressure,relax_pressure_var, Nsim)
	relax_crystallization_vector = np.random.normal(relax_crystallization, relax_crystallization_var, Nsim)

	for i in range(Nsim):
		if(r_vector[i] < 0.0):
			r_vector[i] = np.random.normal(r, r_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(T0_vector[i] < 0.0):
			T0_vector[i] = np.random.normal(T0, T0_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(u0_vector[i] < 0.0):
			u0_vector[i] = np.random.normal(u0, u0_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(u0_vector[i] < 101325.0):
			p0_vector[i] = np.random.normal(p0, p0_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(ang0_vector[i] < 0.0):
			ang0_vector[i] = np.random.normal(ang0, ang0_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(alpha_g_0_vector[i] < 0.0 or alpha_g_0_vector[i] > 1.0):
			alpha_g_0_vector[i] = np.random.normal(alpha_g_0, alpha_g_0_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(x_dg_vector[i] < 0.0 or x_dg_vector[i] > 1.0):
			x_dg_vector[i] = np.random.normal(x_dg, x_dg_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(alpha_c_0_vector[i] < 0.0 or alpha_c_0_vector[i] > 1.0):
			alpha_c_0_vector[i] = np.random.normal(alpha_c_0, alpha_c_0_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(p_failure_vector[i] < 0.0):
			p_failure_vector[i] = np.random.normal(p_failure, p_failure_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(relax_exsolution_vector[i] < 0.0):
			relax_exsolution_vector[i] = np.random.normal(relax_exsolution, relax_exsolution_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(relax_pressure_vector[i] < 0.0):
			relax_pressure_vector[i] = np.random.normal(relax_pressure, relax_pressure_var, 1)
			i = i - 1
	for i in range(Nsim):
		if(relax_crystallization_vector[i] < 0.0):
			relax_crystallization_vector[i] = np.random.normal(relax_crystallization, relax_crystallization_var, 1)
			i = i - 1
else:
	Nsim = 1

if( sim_mode == 3 ):
	Results = np.zeros((Nsim, 25))

for sim in range(Nsim):

	if( sim_mode == 3 ):
		r = r_vector[sim]
		T0 = T0_vector[sim]
		u0 = u0_vector[sim]
		p0 = p0_vector[sim]
		ang0 = ang0_vector[sim]
		Tt = Tt_vector[sim]
		alpha_g_0 = alpha_g_0_vector[sim]
		x_dg = x_dg_vector[sim]
		alpha_c_0 = alpha_c_0_vector[sim]
		p_failure = p_failure_vector[sim]
		relax_exsolution = relax_exsolution_vector[sim]
		relax_pressure = relax_pressure_vector[sim]
		relax_crystallization = relax_crystallization_vector[sim]
		Results[sim,0:13] = np.array([r, T0, u0, ang0, p0, Tt, alpha_g_0, x_dg, alpha_c_0, p_failure, relax_exsolution, relax_pressure, relax_crystallization])

	# DEFINITIONS FOR BUBBLE DYNAMICS
	if( alpha_g_0 < 0 or alpha_g_0 > 1 or alpha_c_0 < 0 or alpha_c_0 > 1):
		print 'Incompatible volume fraction of gas and crystals.'
		sys.exit(0)

	# Bulk density of exsolved gas
	rho_B_g = p0 * mw_water * alpha_g_0 / R_gas / T0

	# Bulk density of crystals
	rho_B_c = rho_c * ( 1 - alpha_g_0 ) * alpha_c_0

	# Density of melt
	rho_m = rho_r / (1 + beta * ( T0 - T_r2 ) )

	# Bulk density of melt (including dissolved water)
	rho_B_md = rho_m * ( 1 - alpha_c_0 ) * ( 1  - alpha_g_0 )

	# Bulk density of dissolved water
	rho_B_d = x_dg * ( rho_B_g + rho_B_c + rho_B_md ) - rho_B_g 

	skip_boolean = 0

	if( rho_B_d < 0 ):
		if( sim_mode < 3 ):
			print 'Incompatible crystal content and mass fraction of water.'
			sys.exit(0)
		else:
			skip_boolean = 1

	# Bulk density of melt (excluding dissolved water)
	rho_B_m = rho_B_md - rho_B_d

	# Volume fraction of melt and dissolved water
	alpha_md_0 = ( 1 - alpha_c_0 ) * ( 1  - alpha_g_0 )

	# Density of bombs
	rho_bomb = ( rho_B_g + rho_B_md + rho_B_c )

	if( sim_mode == 3 ):
		Results[sim,13] = rho_bomb

	# Strength
	p_failure = p_failure * 1e6

	# Factor for the glass transition range
	ft = 1.0

	if( sim_mode < 3 ):

		print 'Major results:'
		print 'Initial density: ' + str(round( rho_bomb , 2 )) + '.'

	mesh = Grid1D( nx = N, dx = r / N )

	Temperature = CellVariable( name = "Temperature", mesh = mesh , value = T0 , hasOld = True)
	MassGas = CellVariable( name = "MassGas", mesh = mesh , value = rho_B_g , hasOld = True)
	MassTotal = CellVariable( name = "MassTotal", mesh = mesh , value = rho_B_g + rho_B_md + rho_B_c, hasOld = True)
	VolumeGas = CellVariable( name = "VolumeGas", mesh = mesh , value = alpha_g_0 , hasOld = True)
	MassCrystals = CellVariable( name = "MassCrystals", mesh = mesh , value = rho_B_c , hasOld = True)

	current_time = 0

	mu = mu_r * ( Tinf / T_r ) ** 0.7
	if( sim_mode == 1 ):
		rho = rho_atm
	else:

		rho = rho_0 * exp( - Hi / H_atm )

	ang = ang0 * np.pi / 180.0
	u = u0
	vx = u0 * np.cos(ang)
	vy = u0 * np.sin(ang)

	itmax = 10000
	velocity = np.zeros( (itmax, 1) )
	velocity[0,0] = u

	T = np.zeros( ( itmax , N ) )
	T[ 0 , 0 : N ] = Temperature.getValue()[ 0 : N ]
	mass_g = np.zeros( ( itmax , N ) )
	mass_g[0,0:N] = MassGas.getValue()[ 0 : N ] 
	vol_g = np.zeros( ( itmax , N ) )
	vol_g[0,0:N] = VolumeGas.getValue()[ 0 : N ] 

	p_g = np.zeros( ( itmax , N ) )
	p_g[0,0:N] = mass_g[0,0:N] / vol_g[0,0:N] * R_gas * T[0,0:N] / mw_water

	vol_c = np.zeros( ( itmax , N ) )
	vol_c[0,0:N] =  MassCrystals.getValue()[ 0 : N ] / rho_c / (1 - vol_g[0,0:N])

	time = np.zeros( (itmax, 1) )

	hf_c = np.zeros( (itmax, 1) )
	hf_r = np.zeros( (itmax, 1) )

	current_height = Hi

	height = np.zeros( (itmax, 1) )
	height[0,0] = current_height

	current_distance = 0

	distance = np.zeros( (itmax, 1) )
	distance[0,0] = current_distance

	max_pressure_rind = np.zeros( (itmax, 1) )

	TemperatureAnalytical = CellVariable(name="analytical value", mesh = mesh)
	MassGasAnalytical = CellVariable(name="analytical value", mesh = mesh)
	MassTotalAnalytical = CellVariable(name="analytical value", mesh = mesh)
	VolumeGasAnalytical = CellVariable(name="analytical value", mesh = mesh)
	MassCrystalsAnalytical = CellVariable(name="analytical value", mesh = mesh)

	timeStepDuration = 0.01

	eqX = TransientTerm( var = Temperature ) == DiffusionTerm( coeff = k, var = Temperature ) +  2 * k * Temperature.grad[0] / mesh.cellCenters()[0] 

	eqY = TransientTerm( var = MassGas ) ==  ( 1 / ( 1 + np.exp( ft * (- Temperature + Tt) ) )) * ( MassTotal - MassGas - MassCrystals ) / relax_exsolution * ( ( x_dg * MassTotal - MassGas )/( MassTotal - MassGas - MassCrystals ) - sol_1 * ( MassGas / VolumeGas * R_gas * Temperature / mw_water ) ** sol_2 )

	eqZ = TransientTerm( var = MassTotal ) == 0.0

	eqW = TransientTerm( var = VolumeGas ) == ( - VolumeGas + 1 - ( ( MassTotal - MassGas ) /  ( (1 -  MassCrystals / ( rho_c * ( 1 - VolumeGas))) * rho_r / ( 1 + beta * ( Temperature *  ( 1 / ( 1 + np.exp( ft * (- Temperature + Tt) ) )) + Tt * (1 -  ( 1 / ( 1 + np.exp( ft * (- Temperature + Tt) ) ))) - T_r2) ) + MassCrystals / ( 1 - VolumeGas) ) )  ) / relax_pressure

	eqV = TransientTerm( var = MassCrystals ) == ( 1 / ( 1 + np.exp( ft * (- Temperature + Tt )))) / relax_crystallization * ( - MassCrystals + ( 1 - VolumeGas ) * rho_c * np.minimum( alpha_max , np.maximum( alpha_min , a0 + aT * Temperature + aw * ( x_dg * MassTotal - MassGas ) / ( MassTotal - MassGas - MassCrystals )  )  ) )

	eq = eqX & eqZ & eqY & eqW & eqV

	ka = Cp * mu / Pr

	boolean_ground = 0

	boolean_expansion = 0

	for step in range(itmax):

		current_time = current_time + timeStepDuration

		print(current_time, Temperature.getValue()[0])

		time[ step + 1 , 0 ] = current_time

		current_height = current_height + vy * timeStepDuration

		current_distance = current_distance + vx * timeStepDuration

		if( sim_mode == 2 or sim_mode == 3) :

			rho = rho_0 * exp( - current_height / H_atm )

		Re = u * 2 * r * rho / mu + 1e-20

		Cd = 24 / Re * ( 1 + 0.15 * Re **0.681 ) + 0.407 / ( 1 + 8710 / Re ) 

		if( sim_top == 2 ):

			for i in range(len(data_x)):

				if( data_x[i] > current_distance ):

					factor = ( data_x[i] - current_distance ) / (data_x[i] - data_x[i-1])

					Hf = factor * data_y[i-1] + ( 1 - factor ) * data_y[i]

					break

		if( current_height <= Hf ):

			current_height = Hf

			vx = 0

			vy = 0

		else:

			D = rho * Cd * u ** 2.0 * np.pi * r ** 2.0 / 2.0

			ax = - D * cos(ang) / ( rho_bomb * 4 * np.pi * (r ** 3.0) / 3 )

			ay = - g + ( rho / rho_bomb ) * g - D * sin(ang) / ( rho_bomb * 4 * np.pi * (r ** 3.0) / 3 )

			vx = vx + ax * timeStepDuration

			vy = vy + ay * timeStepDuration

			vt = - sqrt( 2 * (rho_bomb - rho) * g * r / ( 3 * rho * Cd ) )

			if( vy <= vt ):

				vy = vt

			if( vx <= 1e-10 ):

				vx = 0

				if( vy > 0):

					ang = 90 * np.pi / 180

				else:

					ang = - 90 * np.pi / 180

			else:

				ang = atan( vy / vx )

		u = sqrt( vx * vx + vy * vy )

		height[ step + 1, 0 ] = current_height

		distance[ step + 1, 0 ] = current_distance

		velocity[ step + 1 , 0 ] = u

		if( boolean_expansion == 0 or boolean_ground == 0 ):

			mu_s = mu_r * ( Temperature.getValue()[N-1] / 273.15 ) ** 0.7 

			hc = ka / ( 2 * r ) * ( 2 + (0.4 * Re ** 0.5 + 0.06 * Re ** (2/3) ) * Pr ** 0.4 * ( mu / mu_s ) ** ( 1.0 / 4.0 ) )

			Ft = ( gamma * ( Temperature.faceValue()[N-1] ** 4.0 - Tinf ** 4.0 ) + hc * ( Temperature.getValue()[N-1] - Tinf )) / ( kb )

			hf_r[ step: step+2, 0 ] = gamma * ( Temperature.getValue()[N-1] ** 4.0 - Tinf ** 4.0 )

			hf_c[ step: step+2, 0 ] = hc * ( Temperature.getValue()[N-1] - Tinf ) / ( kb )

			fluxLeft = 0.0

			fluxRight = - Ft

			Temperature.faceGrad.constrain([fluxRight], mesh.facesRight)

			Temperature.faceGrad.constrain([fluxLeft], mesh.facesLeft)

			fluxLeft = 0.0

			fluxRight = 0.0

			MassGas.faceGrad.constrain([fluxRight], mesh.facesRight)

			MassGas.faceGrad.constrain([fluxLeft], mesh.facesLeft)

			MassTotal.faceGrad.constrain([fluxRight], mesh.facesRight)

			MassTotal.faceGrad.constrain([fluxLeft], mesh.facesLeft)

			VolumeGas.faceGrad.constrain([fluxRight], mesh.facesRight)

			VolumeGas.faceGrad.constrain([fluxLeft], mesh.facesLeft)

			eq.solve( dt = timeStepDuration )

			Temperature.updateOld()

			MassGas.updateOld()

			MassTotal.updateOld()

			VolumeGas.updateOld()

			MassCrystals.updateOld()

			T[step + 1, 0 : N ] = Temperature.getValue()[ 0 : N ]

			aux_rind = N - 1

			aux_rind_factor = 0

			for it_rind in range(N):

				if( T[step + 1, it_rind] < Tt ):

					aux_rind = it_rind

					if(it_rind > 0):
					
						aux_rind_factor = ( Tt - T[step + 1, it_rind] ) / ( T[step + 1, it_rind - 1] -  T[step + 1, it_rind])

					break

			mass_g[step + 1, 0 : N ] = MassGas.getValue()[ 0 : N ]

			vol_g[step + 1, 0 : N ] = VolumeGas.getValue()[ 0 : N ]

			p_g[step + 1, 0 : N ] = mass_g[step + 1, 0 : N ] / vol_g[step + 1, 0 : N ] * R_gas * T[step + 1, 0 : N ] / mw_water

			vol_c[step + 1, 0 : N ] = MassCrystals.getValue()[ 0 : N ] / rho_c / (1 - vol_g[0,0:N])

			if(aux_rind > 0):

				max_pressure_rind[step + 1] =  max(max(p_g[step + 1, aux_rind : N ]), p_g[step + 1, aux_rind ] * (1 - aux_rind_factor) + p_g[step + 1, aux_rind -1 ] * aux_rind_factor)

			else:

				max_pressure_rind[step + 1] = max(p_g[step + 1, aux_rind : N ])

			if( p_failure < max_pressure_rind[step] and boolean_expansion == 0 ):

				boolean_expansion = 1

				factor_exp =  ( max_pressure_rind[step + 1] - p_failure ) / ( max_pressure_rind[step + 1 ] -max_pressure_rind[step] )

				T[ step + 1,  0 : N ] = factor_exp * T[ step , 0 : N ] + ( 1 - factor_exp ) * T[ step + 1, 0 : N ] 

				mass_g[ step + 1,  0 : N ] = factor_exp * mass_g[ step , 0 : N ] + ( 1 - factor_exp ) * mass_g[ step + 1, 0 : N ] 

				vol_g[ step + 1,  0 : N ] = factor_exp * vol_g[ step , 0 : N ] + ( 1 - factor_exp ) * vol_g[ step + 1, 0 : N ] 

				p_g[ step + 1,  0 : N ] = factor_exp * p_g[ step , 0 : N ] + ( 1 - factor_exp ) * p_g[ step + 1, 0 : N ]

				vol_c[ step + 1,  0 : N ] = factor_exp * vol_c[ step , 0 : N ] + ( 1 - factor_exp ) * vol_c[ step + 1, 0 : N ] 

				t_expansion = factor_exp * time[ step , 0 ] + ( 1 - factor_exp ) * time[ step + 1, 0 ]

				distance_expansion = factor_exp * distance[ step , 0 ] + ( 1 - factor_exp ) * distance[ step + 1, 0 ]

				height_expansion = factor_exp * height[ step , 0 ] + ( 1 - factor_exp ) * height[ step + 1, 0 ]

				velocity_expansion = factor_exp * velocity[ step , 0 ] + ( 1 - factor_exp ) * velocity[ step + 1, 0 ]

				if( boolean_ground == 1 ):

					time[ step + 1 , 0 ] = t_expansion

					height[ step + 1 , 0 ] = height_expansion

					velocity[ step + 1 , 0 ] = velocity_expansion

					distance[ step + 1 , 0 ] = distance_expansion

					tf = t_expansion

				index_expansion = step + 1

				volume_gas = 0.0

				volume_gas_after = 0.0

				solid_mass = 0.0

				factor_correction = np.zeros((N,1))

				vesicularity_before = np.zeros((N,1))

				vesicularity_after = np.zeros((N,1))

				for i in range(len( mesh.cellCenters()[0] )):
				
					sum_solid_volume = (1.0 - T[ step + 1 , i] * mass_g[ step + 1 , i] * R_gas / p_g[ step + 1, i] / mw_water)

					solid_mass = solid_mass + ( mesh.cellCenters()[0][i] ** 2.0 ) * ( rho_B_g + rho_B_md + rho_B_c - mass_g[ step + 1 , i])

					sum_volume_gas = T[ step + 1 , i] * mass_g[ step + 1 , i] * R_gas / p_g[ step + 1, i] / mw_water
 
					sum_volume_gas_after = T[step + 1,i] * mass_g[step + 1 , i] * R_gas / mw_water 

					deltaT = 1 / ( 1 + np.exp( ft*( Tt - T[ step + 1 , i]) ) )

					sum_volume_gas_after = sum_volume_gas_after * ( deltaT / p_atm + ( 1 - deltaT ) / p_g[ step + 1, i ] )

					volume_gas = volume_gas + ( mesh.cellCenters()[0][i] ** 2.0 ) * sum_volume_gas

					volume_gas_after = volume_gas_after +  ( mesh.cellCenters()[0][i] ** 2.0 ) * min(sum_volume_gas_after/ (sum_solid_volume + sum_volume_gas_after) , phi_m )

					aux_f = ( volume_gas_after - volume_gas ) * 4 * np.pi * mesh.cellCenters()[0][i] / N 

					factor_correction[i,0] = ( 3.0 / 4.0 / np.pi * ( 4.0 / 3.0 * np.pi * mesh.cellCenters()[0][i] ** 3.0 + aux_f ) ) ** (1.0/3.0)

					vesicularity_before[i,0] = sum_volume_gas

					vesicularity_after[i,0] = min(sum_volume_gas_after/ (sum_solid_volume + sum_volume_gas_after) , phi_m )

				solid_mass = 4 * np.pi * r / N * solid_mass

				volume_gas = volume_gas * 4 * np.pi * r / N

				volume_gas_after = volume_gas_after * 4 * np.pi * r / N

				r_e = ( 3.0 / 4.0 / np.pi * ( 4.0 / 3.0 * np.pi * r ** 3.0 + volume_gas_after - volume_gas ) ) ** (1.0/3.0)

				rho_bomb = solid_mass / ( 4.0 * np.pi / 3.0 * r_e ** 3.0 )

		if( Temperature.getValue()[0] < Tt - 100 and boolean_ground == 1 ):

			tf = current_time

			volume_gas = 0.0

			vesicularity_before = np.zeros((N,1))

			for i in range(len( mesh.cellCenters()[0] )):
				
				sum_solid_volume = (1.0 - T[ step + 1 , i] * mass_g[ step + 1 , i] * R_gas / p_g[ step + 1, i] / mw_water)

				sum_volume_gas = T[ step + 1 , i] * mass_g[ step + 1 , i] * R_gas / p_g[ step + 1, i] / mw_water

				vesicularity_before[i,0] = sum_volume_gas

			vesicularity_after = vesicularity_before

			r_e = r

			if( sim_mode == 3 ):

				Results[sim,14] = -1

				Results[sim,15] = t_ground

				Results[sim,16] = current_distance

				Results[sim,17] = 1000 * r

				Results[sim,18] = r

				Results[sim,19] = 1.0

				Results[sim,20] = Results[sim,13]

				Results[sim,21] = MassCrystals.faceValue()[0] / rho_c / (1 - VolumeGas.faceValue()[0] )

				Results[sim,22] = MassCrystals.faceValue()[N-1] / rho_c / (1 - VolumeGas.faceValue()[N-1] )

				Results[sim,23] = (VolumeGas.faceValue()[0] )

				Results[sim,24] = (VolumeGas.faceValue()[N-1] )

				print ' Simulation finished (N = ' + str(sim+1) + ')'

			break

		if( boolean_expansion == 1 and boolean_ground == 1):	

			if( sim_mode == 3 ):

				Results[sim,14] = t_expansion

				Results[sim,15] = t_ground

				Results[sim,16] = current_distance

				Results[sim,18] = r_e

				Results[sim,19] = r * r / r_e / r_e

				Results[sim,20] = rho_bomb

				Results[sim,21] = MassCrystals.faceValue()[0] / rho_c / (1 - VolumeGas.faceValue()[0] )

				Results[sim,22] = MassCrystals.faceValue()[N-1] / rho_c / (1 - VolumeGas.faceValue()[N-1] )

				Results[sim,23] = vesicularity_after[0]

				Results[sim,24] = vesicularity_after[N-1]

				data_r = np.zeros((N,1))

				data_t = np.zeros((N,1))

				counter = 0

				time_expansion = time[ 0 : index_expansion + 1 , 0 ]

				time_expansion[ index_expansion ] = t_expansion

				for i in range(len( mesh.cellCenters()[0])-1, -1, -1):
	
					for j in range(len(time_expansion)):

						if( Tt > T[j,i] ):

							factor = (T[j,i] - Tt) / (T[j,i] - T[j-1,i])

							taux = (1 - factor) * time_expansion[ j ] + factor * time_expansion[ j - 1 ]

							data_r[counter,0] = mesh.cellCenters()[0][i] - r / 2.0 / N

							data_t[counter,0] = taux	

							counter = counter + 1

							break

				data_t[ counter ] = t_expansion

				m_aux = ( data_r[ counter - 1] - data_r[ counter - 2] ) / ( data_t[ counter - 1] - data_t[ counter - 2] ) 

				n_aux = data_r[ counter - 1 ] - m_aux * data_t[ counter - 1 ]

				data_r[ counter ] = m_aux * data_t[ counter ] + n_aux

				Results[sim,17] = (1000 * (r - data_r[ counter ]) )
			
				print ' Simulation finished (N = ' + str(sim+1) + ')'

			break

		Aux_pg = (p_g[ step + 1 ,  2 : N ] - p_g[ step+1,  1 : N - 1 ]) *  (p_g[ step+1 ,  1 : N - 1 ] - p_g[ step + 1,  0 : N - 2 ])

		wh = np.where(Aux_pg < 0)

		if( len(wh[0]) < 3 ) : 
			timeStepDuration = min(timeStepDuration * 1.05, 10 * r  )
		else:
			timeStepDuration = max(timeStepDuration * 0.5, r )

		if( sim_top == 2 ):

			for i in range(len(data_x)):

				if( data_x[i] > current_distance + vx * timeStepDuration ):

					factor = ( data_x[i] - ( current_distance + vx * timeStepDuration ) ) / (data_x[i] - data_x[i-1])

					Hf = factor * data_y[i-1] + ( 1 - factor ) * data_y[i]

					break

		if( current_height + timeStepDuration * vy <= Hf and boolean_ground == 0 ):

			boolean_ground  = 1

			for j in range(1000):

				dt = timeStepDuration * j / 100.0

				if( sim_top == 2 ):

					for i in range(len(data_x)):

						if( data_x[i] > current_distance + vx * dt ):

							factor = ( data_x[i] - ( current_distance + vx * dt ) ) / (data_x[i] - data_x[i-1])

							Hf = factor * data_y[i-1] + ( 1 - factor ) * data_y[i]

							break

				if( current_height + dt * vy <= Hf ):
				
					timeStepDuration = dt

					tf = current_time + timeStepDuration

					break

			t_ground = current_time + timeStepDuration

if( sim_mode == 3 ):

	np.savetxt('Results.txt', Results)

if( sim_mode < 3 ):

	if( boolean_expansion == 1):
		
		print 'Time for expansion: ' + str(round(t_expansion,2)) + ' s.'

	else:

		print 'Bomb expansion does not occur.'

	print 'Time for reaching the ground: ' + str(round(t_ground,2)) + ' s.'

	print 'Horizontal distance: ' + str(round(max(distance[0:step+2,:]),2)) + ' m.'

	time = time[0 : step + 2, :]

	velocity = velocity[0 : step + 2, : ]

	distance = distance[0 : step + 2, : ]

	height = height[0 : step + 2, : ]

	if( plot_trajectory == 1 ):

		# FIGURES RELATED TO TRAJECTORY

		# Figure of trajectory

		plt.figure()

		plt.plot(distance, height, 'b', label = 'Ballistic projectile')

		if( sim_top == 2 ):

			plt.plot(data_x, data_y, 'k', label = 'Topography')

		if( boolean_expansion == 1 ):

			plt.plot( distance_expansion, height_expansion, 'ro', label = 'Expansion position')

		plt.xlabel('Horizontal distance $[m]$')

		plt.ylabel('Height $[m]$')

		plt.title('Trajectory')

		plt.legend()

		# Horizontal distance

		plt.figure()

		plt.plot(time, distance, 'b',  label = 'Ballistic projectile')

		plt.xlabel('Time $[s]$')

		plt.ylabel('Horizontal distance $[m]$')

		plt.title('Horizontal distance')

		if( boolean_expansion == 1):

			plt.plot( t_expansion, distance_expansion, 'ro', label = 'Expansion position')

		plt.legend()

		# Vertical position

		plt.figure()

		plt.plot(time, height, 'b',  label = 'Ballistic projectile')

		plt.xlabel('Time $[s]$')

		plt.ylabel('Height $[m]$')

		plt.title('Vertical position')

		if( boolean_expansion == 1):

			plt.plot( t_expansion, height_expansion, 'ro', label = 'Expansion position')

		plt.legend()

		# Velocity

		plt.figure()

		plt.plot(time, velocity, 'b', label = 'Ballistic projectile')

		plt.xlabel('Time $[s]$')

		plt.ylabel('Velocity $[m/s]$')

		plt.title('Velocity')

		if( boolean_expansion == 1):

			plt.plot( t_expansion, velocity_expansion, 'ro', label = 'Expansion position')

		plt.legend()

	if( boolean_expansion == 1 ):

		T = T[ 0 : index_expansion + 1, : ]

		hf_c = hf_c[ 0 : index_expansion + 1 , :]

		hf_r = hf_r[ 0 : index_expansion + 1 , : ]

		mass_g = mass_g[ 0 : index_expansion + 1, : ]

		vol_g = vol_g[ 0 : index_expansion + 1 , :]

		p_g = p_g[ 0 : index_expansion + 1 , : ]

		time_expansion = time[ 0 : index_expansion + 1 , 0 ]

		time_expansion[ index_expansion ] = t_expansion

	else:

		T = T[ 0 : step + 2, : ]

		hf_c = hf_c[ 0 : step + 2, : ]

		hf_r = hf_r[ 0 : step + 2, : ]

		mass_g = mass_g[ 0 : step + 2, : ]

		vol_g = vol_g[ 0 : step + 2, : ]

		p_g = p_g[ 0 : step + 2, : ]

		time_expansion = time

		t_expansion = time_expansion[ step + 1, 0]

	if( plot_temperature == 1 ):

		# FIGURES RELATED TO THERMAL EVOLUTION

		# Temporal evolution of temperature along the ballistic projectile

		plt.figure()

		timesplot = np.arange(0, t_expansion+1e-5, t_expansion / 5.0)

		for i in range(len(timesplot)):

			time_obj = timesplot[i]

			for j in range( len( time_expansion ) ):

				if(time_obj == time_expansion[j] ):

					plt.plot(100 * mesh.cellCenters()[0], T[j,:] - 273.15, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

				elif(time_obj < time[j]):

					factor = (time_expansion[j] - time_obj)/(time_expansion[j] - time_expansion[j-1])

					Taux = (1 - factor) * T[j,:] + factor * T[j-1,:]

					plt.plot(100 * mesh.cellCenters()[0], Taux - 273.15, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

		plt.plot(100 * mesh.cellCenters()[0], 0*mesh.cellCenters()[0] + Tt - 273.15, 'k:', label = 'Glass transition temperature')

		plt.legend()

		plt.xlabel('Distance along the ballistic projectile $[cm]$')

		plt.ylabel('Temperature $[degC]$')

		plt.title('Temperature profiles in the ballistic projectile before expansion')

		plt.xlim([0, 100 * r])

		# Temporal evolution of temperature along the ballistic projectile

		plt.figure()

		distancesplot = np.arange(0, r, r/5.0)

		for i in range(len(distancesplot)):

			distance_obj = distancesplot[i]

			for j in range(len( mesh.cellCenters()[0] )):

				if(distance_obj == mesh.cellCenters()[0][j]):

					plt.plot(time_expansion, T[:,j] - 273.15, label = 'r = ' + str(100 * distance_obj) + ' cm')

					break

				elif(distance_obj < mesh.cellCenters()[0][j]):

					factor = (mesh.cellCenters()[0][j] - distance_obj)/(mesh.cellCenters()[0][j] - mesh.cellCenters()[0][j - 1])

					Taux = (1 - factor) * T[:,j] + factor * T[:,j-1]

					plt.plot(time_expansion, Taux - 273.15, label = 'r = ' + str(100 * distance_obj) + ' cm from bomb core')

					break

		plt.plot(time_expansion, T[:,N-1] - 273.15, label = 'r = ' + str(100 * r) + ' cm from bomb core')

		plt.plot(time_expansion, 0*time_expansion + Tt - 273.15, 'k:', label = 'Glass transition temperature')

		plt.legend()

		plt.xlabel('Time $[s]$')

		plt.ylabel('Temperature $[degC]$')

		plt.title('Temperature evolution in the ballistic projectile before expansion')

		plt.xlim([0, t_expansion])

	if( plot_heat == 1 ):

		# Heat flux

		plt.figure()

		plt.semilogy(time_expansion, hf_c, label = 'Convective heat flux')

		plt.semilogy(time_expansion, hf_r, label = 'Radiative heat flux')

		plt.ylabel('Heat flux per unit area $[W/m^2]$')

		plt.xlabel('Time $[s]$')

		plt.title('Heat flux per unit area before expansion')

		plt.legend()

		plt.xlim([0, t_expansion])

	counter = 0

	data_r = np.zeros((N,1))

	data_t = np.zeros((N,1))
	
	for i in range(len( mesh.cellCenters()[0])-1, -1, -1):
	
		for j in range(len(time_expansion)):

			if( Tt > T[j,i] ):

				factor = (T[j,i] - Tt) / (T[j,i] - T[j-1,i])

				taux = (1 - factor) * time_expansion[ j ] + factor * time_expansion[ j - 1 ]

				data_r[counter,0] = mesh.cellCenters()[0][i] - r / 2.0 / N

				data_t[counter,0] = taux	

				counter = counter + 1

				break

	if( boolean_expansion ):

		data_t[ counter ] = t_expansion

		m_aux = ( data_r[ counter - 1] - data_r[ counter - 2] ) / ( data_t[ counter - 1] - data_t[ counter - 2] ) 

		n_aux = data_r[ counter - 1 ] - m_aux * data_t[ counter - 1 ]

		data_r[ counter ] = m_aux * data_t[ counter ] + n_aux

	data_r = data_r[ 0 : counter + 1,0]

	data_t = data_t[ 0 : counter + 1,0]

	print 'Rind thickness: ' + str(round(max(1000 * (r - data_r)),2)) + ' mm.'

	if( plot_rind == 1 ) :

		# Rind growth

		plt.figure()

		plt.plot(data_t, 100 * (r - data_r), 'b')

		plt.ylabel('Rind thickness $[cm]$')

		plt.xlabel('Time $[s]$')

		plt.title('Rind thickness')

	if( boolean_expansion == 1):

		print 'Volume of gas before expansion: ' + str(round(volume_gas*1e6,2)) + ' cc.'

		print 'Volume of gas after expansion: ' + str(round(volume_gas_after*1e6,2)) + ' cc.'

		print 'Equivalent radius of expanded bomb: ' + str(round(r_e * 100, 2)) + ' cm.'

		print 'Ratio between the external surface of original and expanded bomb: ' + str(round( r * r / r_e / r_e , 3)) + '.'

		print 'Final density: ' + str(round( rho_bomb , 2 )) + '.'

	if(boolean_expansion == 1):

		print 'Rind vesicularity: ' + str(round(vesicularity_after[N-1]*100,2)) + ' vol%.'

		print 'Core vesicularity: ' + str(round(vesicularity_after[0]*100,2)) + ' vol%.'

	else:

		print 'Rind vesicularity: ' + str(round((VolumeGas.faceValue()[N-1] )*100,2)) + ' vol%.'

		print 'Core vesicularity: ' + str(round((VolumeGas.faceValue()[0] )*100,2)) + ' vol%.'

	print 'Rind crystallinity: ' + str( round(100 * MassCrystals.faceValue()[N-1] / rho_c / (1 - VolumeGas.faceValue()[N-1] ),2)) + ' vol%.'

	print 'Core crystallinity: ' + str(round(100 * MassCrystals.faceValue()[0] / rho_c / (1 - VolumeGas.faceValue()[0]),2)) + ' vol%.'

	if( plot_bubbles == 1 ):

		plt.figure()

		timesplot = np.arange(0, t_expansion + 1e-5, t_expansion / 5.0)

		for i in range(len(timesplot)):

			time_obj = timesplot[i]

			for j in range( len(time_expansion) ):

				if(time_obj == time_expansion[j]):

					plt.plot(100 * mesh.cellCenters()[0], mass_g[j,:], label = 't = ' + str(round(time_obj,2)) + ' s')

					break

				elif(time_obj < time_expansion[j]):

					factor = (time_expansion[j] - time_obj)/(time_expansion[j] - time_expansion[j-1])

					Taux = (1 - factor) * mass_g[j,:] + factor * mass_g[j-1,:]

					plt.plot(100 * mesh.cellCenters()[0], Taux, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

		plt.legend()

		plt.xlabel('Distance along the ballistic projectile $[cm]$')

		plt.ylabel(r'$\rho_2^{B}$' + ' ' + r'$[kg/m^3]$')

		if( boolean_expansion == 1):

			plt.title('Bulk density of gas before expansion')

		else:

			plt.title('Final bulk density of gas')

		plt.xlim([0, 100 * r])

		plt.ylim( ymin = 0 )

		plt.ylim( ymax = np.max( np.max( mass_g[:,:] ) ) * 1.1 )
	
		plt.figure()

		timesplot = np.arange(0, t_expansion + 1e-5, t_expansion / 5.0)

		for i in range(len(timesplot)):

			time_obj = timesplot[i]

			for j in range(len(time_expansion)):

				if(time_obj == time_expansion[j]):

					plt.plot(100 * mesh.cellCenters()[0], vol_g[j,:], label = 't = ' + str(round(time_obj,2)) + ' s')

					break

				elif(time_obj < time_expansion[j]):

					factor = (time_expansion[j] - time_obj)/(time_expansion[j] - time_expansion[j-1])

					Taux = (1 - factor) * vol_g[j,:] + factor * vol_g[j-1,:]

					plt.plot(100 * mesh.cellCenters()[0], Taux, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

		plt.legend()

		plt.xlabel('Distance along the ballistic projectile $[cm]$')

		plt.ylabel(r'$\alpha_2$')

		if( boolean_expansion == 1):

			plt.title('Volume fraction of gas before expansion')

		else:

			plt.title('Final volume fraction of gas')

		plt.xlim([0, 100 * r])

		plt.ylim( ymin = 0 )

		plt.ylim( ymax = np.max( np.max( vol_g[:,:] ) ) * 1.1 )

		plt.figure()

		timesplot = np.arange(0, t_expansion + 1e-5, t_expansion / 5.0)

		for i in range(len(timesplot)):

			time_obj = timesplot[i]

			for j in range(len(time_expansion)):

				if(time_obj == time_expansion[j]):

					plt.plot(100 * mesh.cellCenters()[0], p_g[j,:]/1e6, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

				elif(time_obj < time_expansion[j]):

					factor = (time_expansion[j] - time_obj)/(time_expansion[j] - time_expansion[j-1])

					Taux = (1 - factor) * p_g[j,:] + factor * p_g[j-1,:]

					plt.plot(100 * mesh.cellCenters()[0], Taux/1e6, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

		plt.legend()

		plt.xlabel('Distance along the ballistic projectile $[cm]$')

		plt.ylabel(r'$p_2$' + ' ' + r'$[MPa]$')

		if( boolean_expansion == 1):

			plt.title('Pressure of bubbles before expansion')

		else:

			plt.title('Final pressure of bubbles')

		plt.xlim([0, 100 * r])

		plt.ylim( ymin = 0 )

		plt.ylim( ymax = np.max( np.max( p_g[:,:]/1e6 ) ) * 1.1 )

		plt.figure()

		if( boolean_expansion == 1):

			plt.subplot(1,2,1)

			plt.plot(100 * mesh.cellCenters()[0], vesicularity_before * 100, 'b')

			plt.xlabel('Distance along the expanded ballistic projectile $[cm]$')

			plt.ylabel('Vesicularity $[vol.\%]$')

			plt.title('Vesicularity before expansion')

			plt.ylim([0, 100])

			plt.xlim([0, 100 * r])

			plt.subplot(1,2,2)

			plt.plot(factor_correction * 100, vesicularity_after * 100, 'r')

			plt.xlabel('Distance along the expanded ballistic projectile $[cm]$')

			plt.ylabel('Vesicularity $[vol.\%]$')

			plt.title('Vesicularity after expansion')

			plt.ylim([0, 100])
	
			plt.xlim([0, 100 * r_e])

		else:

			plt.plot(100 * mesh.cellCenters()[0], vesicularity_before * 100, 'b')

			plt.xlabel('Distance along the expanded ballistic projectile $[cm]$')

			plt.ylabel('Vesicularity $[vol.\%]$')

			plt.title('Final vesicularity')

			plt.ylim([0, 100])

			plt.xlim([0, 100 * r])

		plt.figure()

		timesplot = np.arange(0, t_expansion + 1e-5, t_expansion / 5.0)

		for i in range(len(timesplot)):

			time_obj = timesplot[i]

			for j in range(len(time_expansion)):

				if(time_obj == time_expansion[j]):

					plt.plot(100 * mesh.cellCenters()[0], vol_c[j,:], label = 't = ' + str(round(time_obj,2)) + ' s')

					break

				elif(time_obj < time_expansion[j]):

					factor = (time_expansion[j] - time_obj)/(time_expansion[j] - time_expansion[j-1])

					Taux = (1 - factor) * vol_c[j,:] + factor * vol_c[j-1,:]

					plt.plot(100 * mesh.cellCenters()[0], Taux, label = 't = ' + str(round(time_obj,2)) + ' s')

					break

		plt.legend()

		plt.xlabel('Distance along the ballistic projectile $[cm]$')

		plt.ylabel('Volume fraction of crystals')

		plt.title('Crystals')

		plt.xlim([0, 100 * r])

		plt.ylim( ymin = 0 )

	if( plot_trajectory == 1 or plot_temperature == 1 or plot_heat == 1 or plot_rind == 1 or plot_bubbles == 1 ):

		plt.show()
