# runvsichydro should be in the search path
import runvischydro as rnv
import numpy as np
import pprint
import matplotlib.pyplot as plt

# Script to run bjorken hydro tests with gaussian initial conditions
# The folllowing script can run either uniform or gaussian bjorken hydro
# tests. In the uniform case we are comparing the code result to the ODE
# solution. In the gaussian case we just run and plot the result.
#
# The usage is:
# python bjorkentests.py --testcase uniform
# or
# python bjorkentests.py --testcase gaussian 
#
# It may be necessary to adjust the runcommand variable below to point to
# the correct viscous hydro executable on your system.
runcommand='../../vischydro'

def getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width):
    """ Helper function for getdata, returns the filename based on parameters """
    s='bjnbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    return s.replace('.', 'd')

def runbj(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width, tmin, tmax):

    """ Run bjorken hydro with given parameters """
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)

    # run viscous hydro in the density frame
    rnv.data['initial_time'] = tmin
    rnv.data['final_time'] = tmax
    rnv.data['xmin'] = -80
    rnv.data['xmax'] = 80
    rnv.data['NX'] = 513
    rnv.data['cfl_max'] = 0.1
    rnv.data['dt_max'] = 0.02
    rnv.data['eta_over_s'] = fourpietabys/(4.0*np.pi)
    rnv.data['iofilename'] = name + '.h5'
    rnv.data['bjorken_expansion'] = True

    rnv.options.update({'-ts_exact_final_time': 'INTERPOLATE'})
    pprint.pprint(rnv.data)
    xarray, initialdata = rnv.ic1(rnv.data, A=gaussian_amplitude, delta=gaussian_const, w=gaussian_width) 

    rnv.runcode(xarray, initialdata, rnv.data, inputs=name + '.json', runcommand=runcommand, actually_run=True) 


# Solve an ode for uniform bjorken hydro with a parameter etabys 
# This is the equation to be solved. 
def de_dt(t, e, fourpietabys, nc=3.0): 
    """ ODE for uniform bjorken hydro with viscosity correction """

    Ce = 2 * (nc**2 - 1.) * np.pi**2 /30. 
    T = (e / Ce)**(1./4.) 
    s = (4.0/3.0) * e/T 

    eta  = (fourpietabys/(4.0*np.pi)) *  s
    pi = (4.0/3.0) * eta / t  # shear stress
    dedt = - (e + e/3.0 - pi) / t
    return dedt

# This function solves the ODE and plots the result compared to the code result
def solve_uniform_bjorken(e0, t0, tmax, fourpietabys, nc=3.0):
    """ Solve the uniform bjorken hydro with viscosity correction ODE, returns the solution on a grid between t0 and tmax """
    from scipy.integrate import solve_ivp
    sol = solve_ivp(de_dt, [t0, tmax], [e0], args=(fourpietabys, nc), dense_output=True)
    tgrid = np.linspace(t0, tmax, 100)
    sol = sol.sol(tgrid)[0] 
    plt.plot(tgrid, tgrid* sol, '-', label='Ideal+viscous $4\\pi\\eta/s={}$'.format(fourpietabys))

def plot_uniform_bj(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width):
    import h5py as h5
    import json
    import matplotlib.pyplot as plt
    """ Plot the bjorken hydro result for given parameters. Compare to the ODE solution """

    # get the filename
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    data = json.load(open(name + '.json'))
    times = np.loadtxt(name + '_grid_t.txt')
    with h5.File(name + '.h5', 'r') as file:
        # Get the central point as  a function of time
        center_index = file['x'].shape[0] // 2
        solution = file['solution'][:, center_index, 0] # assuming energy density is the first variable 

        e0 = gaussian_const ;
        t0 = data['initial_time']
        t = np.linspace(t0, data['final_time'], 100)
        plt.plot(t, t * e0 * (t0 / t)**(4.0/3.0), '-', label='Ideal hydro')
        solve_uniform_bjorken(e0, t0, data['final_time'], fourpietabys)
        plt.plot(times[:, 0], times[:, 0] * solution, '.', label='code $4\\pi\\eta/s={}$'.format(fourpietabys))

    nc=3.0
    Ce = 2 * (nc**2 - 1.) * np.pi**2 /30. 
    T0 = (e0 / Ce)**(1./4.) 
    Kn = 1./(4 * np.pi) * (4.0 * fourpietabys) / (3.0 * t0 * T0)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('t * e(t)')
    plt.title('Bjorken hydro with uniform initial condition, $1/Kn = {:.2f}, 4\\pi\\eta/s={}$'.format(1/Kn, fourpietabys))
    plt.tight_layout()
    plt.savefig('bjorken_uniform_4pietabys_{}_Kninverse_{:.1f}.png'.format(fourpietabys, 1/Kn))
    plt.show()
    

def plot_gaussian_bjorken(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width, Kn):
    import h5py as h5
    import json
    import matplotlib.pyplot as plt
    """ Plot the bjorken hydro result for given parameters. This is for the gaussian initial condition case """

    # get the filename
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    data = json.load(open(name + '.json'))
    times = np.loadtxt(name + '_grid_t.txt')
    with h5.File(name + '.h5', 'r') as file:
        x = file['x'][:]
        # Get the central point as  a function of time
        solution = file['solution'][:, :, 0] # assuming energy density is the first variable 

    plt.figure()
    plt.title('Bjorken hydro with gaussian initial condition $1/Kn={Kninverse:.1f}, 4\\pi\\eta/s={fourpietabys:.1f}$'.format(Kninverse=1/Kn, fourpietabys=fourpietabys))
    plt.xlabel('x')
    for t in range(0, times.shape[0], max(1, times.shape[0]//5)):
        plt.ylabel('t * E(t, x)')

        # plot the y-axis on a logscale
        plt.plot(x, times[t, 0] * solution[t, :], '-', linewidth=0.5, label='t={}'.format(times[t,0]))


    plt.legend()
    plt.tight_layout()
    plt.savefig('bjorken_gaussian_4pietabys_{}_Kninverse_{:.1f}.png'.format(fourpietabys, 1/Kn))
    plt.show()

if __name__ == '__main__':

    # The folllowing script can run either uniform or gaussian bjorken hydro
    # tests. In the uniform case we are comparing the code result to the ODE
    # solution. In the gaussian case we just run and plot the result.
    #
    # The usage is:
    # python bjtests.py --testcase uniform
    # or
    # python bjtests.py --testcase gaussian 

    import argparse
    parser = argparse.ArgumentParser(description='Run bjorken gaussian viscous hydro runs')
    parser.add_argument('--testcase', type=str, required=True, default='gaussian', help='Which testcase to run: uniform or gaussian')
    args = parser.parse_args()  
    
    if args.testcase == 'uniform':
        # collect the input parameters using argparse testcase
        fourpietabys = [3.]
        gaussian_const = 0.000001

        # Gluon gas with nc=3
        Ce = 2. * (3.0**2 -1.) * np.pi**2 /30.0 
        Cs = 4.0/3.0 * Ce

        # Parameters for initial condition
        tau0T0 = 1.
        tau0 = 1.0
        T0 = tau0T0 / tau0
        e0 = Ce * T0**4
        R = 3.5  #irrelevant here

        Kn = 1./(4 * np.pi) * (4.0 * fourpietabys[0]) / (3.0 * tau0 * T0)
        print("Initial inverse Knudsen number for 4pi eta/s = {} is 1/Kn = {}".format(fourpietabys[0], 1/Kn))

        gaussian_const = e0
        gaussian_amplitude = 0
        gaussian_width = 2.*R**2
        tmax = 16.0

        runbj(fourpietabys[0], gaussian_const, gaussian_amplitude, gaussian_width, tau0, tmax)
        
        plot_uniform_bj(fourpietabys[0], gaussian_const, gaussian_amplitude, gaussian_width)

    elif args.testcase == 'gaussian':
        fourpietabys = [5.]
        gaussian_const = 0.000001

        Ce = 2. * (3.0**2 -1.) * np.pi**2 /30.0 
        tau0T0 = 1.
        tau0 = 1.0
        T0 = tau0T0 / tau0
        e0 = Ce * T0**4
        R = 3.5 

        Kn = 1./(4 * np.pi) * (4.0 * fourpietabys[0]) / (3.0 * tau0 * T0)
        print("Initial inverse Knudsen number for 4pi eta/s = {} is 1/Kn = {}".format(fourpietabys[0], 1/Kn))
        
        gaussian_amplitude = e0
        gaussian_width = 2.*R**2
        tmax = 16.0

        # Run the code
        runbj(fourpietabys[0], gaussian_const, gaussian_amplitude, gaussian_width, tau0, tmax)

        # Plot the result
        plot_gaussian_bjorken(fourpietabys[0], gaussian_const, gaussian_amplitude, gaussian_width, Kn)
        

    
