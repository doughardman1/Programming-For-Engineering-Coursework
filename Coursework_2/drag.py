############## 650022047 - drag.py ##############

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import ode
import scipy.interpolate

d = 0.1 #[m]
rho = 1.225 #[kg/m^3]
Cd = 0.479 #[dimensionless]
area = math.pi*(d/2)**2 #[m^2]
k = rho*Cd*area # drag coefficient
m = 5.5 # [kg] projectile mass
g = 9.81 # [kg/ms^2] acceleration due to gravity

def drag_ode(t,Y):
    """
    Takes arguments of time t and the vector y(t) = [vx,vy,rx,ry] for projectile
    motion with aerodynamic drag. Returns function F(t, y) = [ax,ay,vx,vy] as a vector.

    >>> drag_ode(0.0, [500 * math.cos(math.pi/2), 500 * math.sin(math.pi/2), 0, 1])
    [-6.413419723344337e-15, -114.54909257444041, 3.061616997868383e-14, 500.0]
    """
    absVal = ((Y[0])**2+(Y[1])**2)**0.5
    
    return [ (1/m)*(-0.5*k*absVal*Y[0]),(1/m)*(-0.5*k*absVal*Y[1]- m*g) ,Y[0], Y[1] ]
            

def solve_ode_euler(v0,alpha,h,dt,t_max = 1000):
    """
    Takes initial projectile velocity v0 [m/s], initial attack angle alpha [deg],
    initial height h [m], timestep dt [s] and maximum integration time t_max [s].
    
    Solves drag_ode() function using foward Euler time integration method.

    Returns 5 column numpy array containing [t,vx,vy,rx,ry]

    >>> solution = solve_ode_euler(1000.0, 20.0, 0.5, 0.1)
    >>> solution.shape
    (276, 5)
    """

    if v0 <= 0:
        raise ValueError("v0 must be positive numbers")
    if alpha <= 0 or alpha >= 90:
        raise ValueError("alpha must be between 0 and 90 degress")
    if h < 0:
        raise ValueError("Initial height cannot be less than 0")
    if dt <= 0:
        raise ValueError('timestep must be positive')
    
    alphaRad = math.radians(alpha)
    
    # set projectile system initial conditions
    rx = 0
    ry = h
    vx = v0*math.cos(alphaRad)
    vy = v0*math.sin(alphaRad)
    t = 0
    
    Y = np.array([vx,vy,rx,ry])
    result = np.array([t,vx,vy,rx,ry])
    
    while True: # Foward Euler time integration method
        t += dt
        Y1 = drag_ode(t,Y)
        vx = vx + dt * Y1[0]
        vy = vy + dt * Y1[1]
        rx = rx + dt * Y1[2]
        ry = ry + dt * Y1[3]
       
        if ry < 0:
            Y = np.array([vx,vy,rx,ry])
            result = np.vstack((result, np.insert(Y,0,t)))
            return result
        elif t >= t_max:
            return result
        else:
            Y = np.array([vx,vy,rx,ry])
            result = np.vstack((result, np.insert(Y,0,t)))

        
def solve_ode_scipy(v0,alpha,h,dt,t_max = 1000):
    """
    Takes initial projectile velocity v0 [m/s], initial attack angle alpha [deg],
    initial height h [m], timestep dt [s] and maximum integration time t_max [s].
    
    Solves drag_ode() function using scipy.integrate.ode() function.

    Returns 5 column numpy array containing [t,vx,vy,rx,ry]

    >>> solution = solve_ode_scipy(1000.0, 20.0, 0.5, 0.1)
    >>> solution.shape
    (277, 5)
    """
    
    if v0 <= 0:
        raise ValueError("v0 must be positive numbers")
    if alpha <= 0 or alpha >= 90:
        raise ValueError("alpha must be between 0 and 90 degress")
    if h < 0:
        raise ValueError("Initial height cannot be less than 0")
    if dt <= 0:
        raise ValueError('timestep must be positive')
    
    alphaRad = math.radians(alpha)
    
    # set projectile system initial conditions
    t = 0 
    rx = 0
    ry = h
    vx = v0*math.cos(alphaRad)
    vy = v0*math.sin(alphaRad)
    Y = np.array([vx,vy,rx,ry])
    result = np.array([t,vx,vy,rx,ry])  # setup array to hold system data
    
    system = ode(drag_ode).set_integrator('dopri5')
    system.set_initial_value(Y,t)

    while system.successful() and system.t <= t_max: # scipy ode integration method
        solution = system.integrate(system.t+dt)
        
        if solution[3] < 0:
            result = np.vstack((result, np.insert(solution,0,t)))
            # adds final solution to result array to ensure that the trajectory falls below y=0 when plotted
            return result
        else:
            result = np.vstack((result, np.insert(solution,0,t)))


    
if __name__ == "__main__":

    import doctest
    doctest.testmod()

    dt = 0.1 # set timestep for trajectory calculations

    while True:
        try:
            v0 = float(input('Enter an initial speed [m/s]: '))
            alpha = float(input('Enter an initial angle of attack [deg]: '))
            h = float(input('Enter an initial height [m]: '))

            break
        except ValueError:
            print("Not a valid input, please try again!")

    if v0 <= 0:
        raise ValueError("v0 must be positive numbers")
    if alpha <= 0 or alpha >= 90:
        raise ValueError("alpha must be between 0 and 90 degress")
    if h < 0:
        raise ValueError("Initial height cannot be less than 0")

    alphaRad = math.radians(alpha)

    eulerResult = solve_ode_euler(v0, alpha, h, dt) # compute trajectory using solve_ode_euler function

    if eulerResult[:,4][-1:]: # if trajectory ends below y = 0, linear interpolation can be used 
        x_axis_interception = eulerResult[:,3][-2:] # last 2 x points in trajectory where x-axis is crossed by projectile
        y_axis_interception = eulerResult[:,4][-2:] # last 2 y points in trajectory where x-axis is crossed by projectile
        t_axis_interception = eulerResult[:,0][-2:] # last 2 time points in trajectory where x-axis is crossed by projectile

        dist_travelled = scipy.interpolate.interp1d(y_axis_interception,x_axis_interception)(0) # linear interpolation to find final distance travelled
        time_travelled = scipy.interpolate.interp1d(y_axis_interception,t_axis_interception)(0) # linear interpolation to find final time travelled

    else: # if max integration time is hit and trajectory ends above x-axis then dist_travelled and dist_travelled are just the last values in their respective array colums
        dist_travelled = eulerResult[:,3][-1:]
        time_travelled = eulerResult[:,0][-1:]
        
    print('Travel time [s]       : ',time_travelled)
    print('Distance travelled [m]: ',dist_travelled)
    print('Maximum height [m]    : ',max(eulerResult[:,4]))

    v0_sin_alpha = v0 * math.sin(alphaRad)
    v0_cos_alpha = v0 * math.cos(alphaRad)

    # Compute projectile trajectory with no drag
    maxDist_noDrag = (v0_cos_alpha / g) * (v0_sin_alpha + ((v0_sin_alpha**2) + 2*g*h) **0.5)
    maxTime_noDrag = maxDist_noDrag / v0_cos_alpha
    numberOftimeSteps_noDrag = int(maxTime_noDrag / dt) + 1 # add extra step to ensure y value goes through 0
    distPerStep_noDrag = maxDist_noDrag / numberOftimeSteps_noDrag
    
    x = 0 # initial system conditions
    y = h
    x_noDrag = [x]
    y_noDrag = [y]

    for step in range(0,numberOftimeSteps_noDrag):
        x += distPerStep_noDrag
        y = h + (x * math.tan(alphaRad) - (g * (x**2))/(2*(v0_cos_alpha)**2))
        x_noDrag.append(x)
        y_noDrag.append(y)

    # plot trajectory from Euler method and trajectory with no drag
    plt.plot(x_noDrag,y_noDrag)
    plt.plot(eulerResult[:,3],eulerResult[:,4])
    plt.legend(('No drag','drag'))
    plt.xlabel('Downrange distance [m]')
    plt.ylabel('Height [m]')
    plt.ylim(bottom = 0)
    plt.xlim(left = 0)
    plt.show()
    
    
    







