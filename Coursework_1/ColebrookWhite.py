############## - FRICTION FACTORS ##############

import math
import numpy as np
import matplotlib.pyplot as plt

def colebrook(Fd,Re,epsD):
    '''
    Calculates the value of the Colebrook-White equation for values of Friction factor Fd, Reynolds number Re,
    Relative roughness/pipe diameter epsD. Returns a float
    '''
    
    term1 = 1/math.sqrt(math.fabs(Fd))
    term2 = 2.51/(Re*math.sqrt(math.fabs(Fd)))
    term3 = epsD/3.72
    return term1 + 2*math.log10(term2 + term3)

def colebrook_deriv(Fd,Re,epsD):
    '''
    Calculates the vaule of the derivitive of the Colebrook-White equation for values of Friction factor Fd, Reynolds number Re,
    Relative roughness/pipe diameter epsD.
    Returns a float
    '''
    
    term1 = epsD/3.72
    term2 = 2.51/(Re*math.sqrt(math.fabs(Fd)))
    return -0.5 * (Fd**(-3/2)) * (1 + (2*2.51)/(math.log(10)*(term1 + term2)*Re))
    
def newton(x0, Re, epsD, maxiter=100, tol=1e-9):
    '''
    Finds roots of the Colebrook-white equation for turbulant flow using Newton-Raphson iterative method.
    Inputs are x0 as initial root guess, Reynolds number Re, Relative roughness/pipe diameter epsD, max number of iterations maxiter
    and tolerance tol.
    Returns a tuple of root of equation and number of iterations required to find root within tolerance limit
    '''
    
    x0 = math.fabs(x0) # current x
    i = 0

    while i < maxiter:
        # Newton iteration algorithm
        res = colebrook(x0,Re,epsD)/colebrook_deriv(x0,Re,epsD) 
        x1 = x0 - res
        x0 = x1
        
        i += 1
        if abs(res) < tol: # end iterations when res is smaller than given tolerance
            break
        elif i == (maxiter - 1):
            raise Exception("Root could not found after maximum number of iterations")
    return (x0,i)

def moody(filename, points=[]):
    '''
    Plots a Moody diagram for flows of varying epsD values including laminar flow, laminar-turbulent transition and turbulent flow.
    Saves moody diagram as moody.pdf
    '''

    ######### LAMINAR FLOW ########
    # Re >= 500 and Re <= 2500:
    # Use Poiseulle flow definition to find Fd: Fd = 64/Re
    laminarReVals = []
    laminarFdVals = []
    
    for Re in range(500,2500):
        laminarFdVals.append(64/Re)
        laminarReVals.append(Re)
        Re += 1
    
    ######### TURBULENT FLOW ########
    # generate data to plot moody diagram
    # for turbulent flow, 2500 < Re < 10e8
    epsDvals = (0,10**-6,10**-5,10**-4,10**-3,10**-2)
    dataPointNum = 50000 # number of data points to generate, 30000 points produces reliable moody chart
    Re_range = tuple(np.linspace(2000,10**8,dataPointNum)) #represents Re number range from 2000 to 10e8 in 100 points

    Fd_vals = np.zeros((dataPointNum,len(epsDvals))) # creates matrix to store Fd data from newton() in

    epsD_index = 0
    Fd_guess = 0.01 # first guess for Fd value to start newton function
    
    for epsD in epsDvals:
        Re_range_index = 0
        for Re in Re_range:
            Fd_vals[Re_range_index,epsD_index] = newton(Fd_guess,Re,epsD)[0] # save Fd value from newton function into Fd_vals matrix
            current_Fd = newton(Fd_guess,Re,epsD)[0]
            Fd_guess = current_Fd # set Fd_guess for next value to be the current value of Fd (for efficiency so less iterations of newton are required)
            Re_range_index += 1
        epsD_index += 1

    plt.figure(figsize=(8, 6))
    plt.plot(laminarReVals,laminarFdVals) 
    plt.plot(Re_range,Fd_vals)
    plt.axvline(x=2040, color='r', linestyle='--') # plot turbulence transition line
    plt.loglog() # set axis scales to be logarithmic
    plt.ylim(top=laminarFdVals[0])
    plt.grid(b=None, which='major', axis='both', linestyle='-')
    plt.grid(b=None, which='minor', axis='both', linestyle='--')
    plt.legend(('Laminar','Smooth','$\epsilon/D = 10^{-6}$','$\epsilon/D = 10^{-5}$','$\epsilon/D = 10^{-4}$','$\epsilon/D = 10^{-3}$','$\epsilon/D = 10^{-2}$','Turbulence transition'),loc='lower left',prop={'size': 7})
    plt.xlabel('Reynolds number Re')
    plt.ylabel('Friction factor $f_d$')
    plt.title('Moody diagram for laminar/turbulent pipe flow')

    if len(points) > 0: # block only runs if a list of optional points has been given to function
        suppliedRe_vals = [p[0] for p in points] # creates lists of supplied Re and Fd values from points
        suppliedFd_vals = [p[1] for p in points]
        plt.plot(suppliedRe_vals,suppliedFd_vals,'kx') # plot supplied points as black crosses
        plt.legend(('Laminar','Smooth','$\epsilon/D = 10^{-6}$','$\epsilon/D = 10^{-5}$','$\epsilon/D = 10^{-4}$','$\epsilon/D = 10^{-3}$','$\epsilon/D = 10^{-2}$','Turbulence transition','Supplied points'),loc='lower left',prop={'size': 7})

    plt.savefig(filename)
    plt.show()
    return 

if __name__ == "__main__":

    inputFilename = input("Enter name of input file: ")

    points = [] # set up list to store any supplied points read from input file in
    
    try:
        with open(inputFilename,'r') as f_in, open('pressure_loss.txt','w') as f_out:
            f_out.write("Friction factor  | Pressure loss (Pa/m)")
            for line in f_in:
                f_out.write("\n")
                diameter, velocity, density, viscosity, roughness = [float(a) for a in line.strip().split()]
                Re = (diameter*velocity*density)/viscosity
                epsD = (roughness/1000)/diameter
                if Re <= 2040:
                    # Laminar flow
                    Fd = 64/Re
                else:
                    # Turbulent flow
                    Fd = newton(0.01,Re,epsD)[0]
                points.append((Re,Fd))
                pressureLoss = Fd*(density*(velocity**2))/(2*diameter)
                f_out.write(str(Fd)[0:16] + " | " + str(pressureLoss)[0:16]) # formats string to 16 characters
    except IOError:
        print("Input file not found")
    except:
        print("Unexpected input / Input file may contains unexpected errors")
        
    moody('moody.pdf',points)
    
  

