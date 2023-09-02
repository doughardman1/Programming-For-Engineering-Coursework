############## 650022047 - target.py ############## 

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import cm

import drag

def objective(initial,target_xy):
    """
    Takes tuple of initial projectile velocity and angle (initial position fixed at (0,0)),
    and tuple of target x and y coordinates.
    
    Computes projectile trajectory using drag.solve_ode_scipy() and the minimum distance [m] from
    the trajectory to the target.

    Return float of minimum distance value

    >>> objective((100.0, 80.0), (1000.0, 50.0))
    737.44292337306604
    """
    
    if target_xy[0] <= 0 or target_xy[1] <= 0:
        raise ValueError('Target x or y coordinates must be positive')
    if initial[0] <= 0:
        raise ValueError("v0 must be positive numbers")
    if initial[1] <= 0 or initial[1] >= 90:
        raise ValueError("alpha must be between 0 and 90 degress")

    
    trajecData = drag.solve_ode_scipy(initial[0],initial[1],0,0.01) # 0.01 timestep used to generate accurate data
    # larger timesteps can be used to speed up processing time but the data will be less accurate

    x_targ = target_xy[0] - trajecData[:,[3]]
    y_targ = trajecData[:,[4]] - target_xy[1]

    minDist = np.amin(((x_targ)**2 + (y_targ)**2)**0.5)

    return minDist

def objective_image(target_xy,nx,ny,generateInitialGuess = False):
    """
    Takes tuple of target x and y coordinates, and integers nx and ny for number of x and y pixels
    Generates colourmap of minimum distance from projectile to given target for
    initial projectile velocities and angles within ranges:

        Initial velocity: 100m/s to 800 m/s
        Initial angle   : 10 deg to 80deg

    Also takes boolean argument generateInitialGuess which, when True, allows the
    objective_image() function to be used to generate initial angle and velocity guesses
    for use in the target() function which are not usually returned by the
    objective_image() function.
    """

    if nx < 1 or ny < 1:
        raise ValueError('Number of x or y pixels cannot be less than 1')
    
    if target_xy[0] <= 0 or target_xy[1] <= 0:
        raise ValueError('Target x or y coordinates must be positive')
    
    distanceData = []
    initialData = []
    v0_values = np.linspace(100,800,nx)
    alpha_values = np.linspace(10,80,ny)

    for alpha in alpha_values:
        for v0 in v0_values:
            distance = objective((v0,alpha),target_xy)
            distanceData.append(distance)
            initialData.append((v0,alpha))
            
    distances = np.asarray([distanceData])

    if generateInitialGuess == True: # return data for target() initial guesses without plotting colormap  
        initialData = np.asarray(initialData)
        return (initialData,distanceData)
    
    distances = np.flipud(distances.reshape((nx,ny)))
    # reshapes distances into matrix of dimension nx by ny and transforms to fit required colormap axis

    image = plt.imshow(distances,cmap=plt.get_cmap('viridis'),extent=[100,800,10,80],interpolation='bilinear',aspect='auto') 
    
    plt.colorbar(image)
    plt.xlabel('Initial velocity [m/s]')
    plt.ylabel('Initial angle [deg]')
    plt.title('Distance to target [m]')
    plt.savefig('objective.pdf')
    return 

def target(tx, ty):
    """
    Takes x and y coordinates of target, tx and ty.
    Uses the objective_image() function to generate an initial guess of the required initial projectile
    velocity and angle required to hit the given target.
    Uses scipy.optimize.minimize() function with these initial velocity and angle guesses to find the exact
    initial velocity and angle required to hit the target to within 10cm (0.1m) of it.

    If optimization is sucessful, returns (True, v0, alpha, minimum distance to target)

    If optimization is unsucessful, returns (False, 0, 0, 0)
    """

    if tx <= 0 or ty <= 0:
        raise ValueError('Target x or y coordinates must be positive')
    
    # uses objective_image() with generateInitialGuess = True to find estimates of the 
    # initial velocity and angle required to hit the given target
    
    initialData,distanceData = objective_image((target_x,target_y),30,30,generateInitialGuess = True)
    # 30 x 30 generates sufficient initial guesses such that the optimization is usually sucessful
    # smaller values can be used to increase processing speed of the program but this could lead to unsucessful optimization
    
    minIndex = np.argmin(distanceData) # find index of minimum distance to target value
    
    initialCondition = initialData[minIndex] # finds initial velocity and angle at minIndex

    result = minimize(lambda x: objective((x[0], x[1]), (tx, ty)), np.array([initialCondition[0],initialCondition[1]]),
                      constraints = [
                          {'type': 'ineq', 'fun': lambda x: 800 - x[0]},
                          {'type': 'ineq', 'fun': lambda x: x[0] - 100},
                          {'type': 'ineq', 'fun': lambda x: 80 - x[1]},
                          {'type': 'ineq', 'fun': lambda x: x[1] - 10}])
    if result['success']:
        minDist = float(objective((result['x'][0],result['x'][1]),(tx,ty)))
        if minDist < 0.1: # set tolerance for minimum distance
            # optimization can still finish sucessfully even if target is not hit so a distance tolerance is needed
            return (True,result['x'][0],result['x'][1],minDist)
    else:
        print("Could not optimise trajectory initial angle and velocity")
    return (False,0,0,0)



if __name__ == "__main__":
    
    import doctest
    doctest.testmod()

    while True:
        try:
            target_x = float(input('Enter target downrange position [m]: '))
            target_y = float(input('Enter a height [m]: '))
            break
        except ValueError:
            print("Not a valid input, please try again!")
    
    target_solution = target(target_x, target_y) # optimize for target coordinates

    if target_solution[0] == False:
        print("Target could not be hit")
    else:
        # target is hit if target() sucessfully returns target_solution[0] == True
        print('We hit the target!')
        print('Initial velocity [m/s]: ',target_solution[1])
        print('Initial angle    [deg]: ',target_solution[2])
        print('Closest distance  [m] : ',target_solution[3])

        yn_answer = None
        while yn_answer not in ("y", "n"):
            yn_answer = input('Generate objective image? [y/n]: ')
            if yn_answer == "y":
                objective_image((target_x,target_y),50,50)
                # number of pixels, nx and ny are set to large values to generate high resoloution images
                print('Saved image to file \'objective.pdf\'.')
                break
            elif yn_answer == "n":
                break
            else:
                print("Not a valid input, please enter either \'y\' or \'n\'.")

        plt.close() # close any other figures open before plotting target graph
        scipyResult = drag.solve_ode_scipy(target_solution[1], target_solution[2], 0, 0.01)
        plt.plot(scipyResult[:,3],scipyResult[:,4])
        plt.plot(target_x,target_y, marker='*', markersize=12,linestyle="None",color="red") # plot target point
        plt.legend(('Trajectory','Target'))
        plt.xlabel('Downrange distance [m]')
        plt.ylabel('Height [m]')
        plt.ylim(bottom = 0)
        plt.xlim(left = 0)
        plt.show()
    
