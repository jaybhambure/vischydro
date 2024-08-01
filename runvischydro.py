import json
import subprocess
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import pdb


# This strcture defines the inputs that will passed through a .json file to the
# c++ vischyrdo code.  It is relativiely important and its structure should not
# be changed without deliberation. The values can be modified, e.g.
# data['eta_over_s'] = 20.
input_definition = { 
                   "NDOF": 7,   # Number of fields in a cell on the grid. 
                   "NX": 201,
                   "xmin": -60.,
                   "xmax": 60.,
                   "initial_time": 0.,
                   "final_time": 47,
                   "cfl_max": 0.49,
                   "eta_over_s": 1./(4.0*np.pi), 
                   "iofilename": "vischydro.h5",
                   "steps_per_print": 10,
                   }

class FixedDataDictionary(dict):
    """
    A dictionary with a fixed set of keys
    """

    def __init__(self, dictionary):
        dict.__init__(self)
        for key in dictionary.keys():
            dict.__setitem__(self, key, dictionary[key])

    def __setitem__(self, key, item):
        if key not in self:
            raise KeyError("The key {} is not defined".format(key))
        dict.__setitem__(self, key, item)

# 
# Creaate a dictionary with fixed keys.
#
data = FixedDataDictionary(input_definition)

# one shoudl keep these options unless you know what you are doing
options = {
    '-ts_type': 'eimex',
    '-ts_max_snes_failures': '5',
    '-pc_type': 'jacobi',
    '-ksp_type': 'bcgs',
    '-ts_adapt_type': 'none', 
}

# Dump the current input data to a json file, which serves as input to the
# vischydro code.  Additional arguments can be passed to the code by adding
# them to the extra_args list and the command line. 
#
# The initialdata array is written to an HDF5 file with filename
# data['iofilename'], creating an array initialdata. This is then read by the
# vischydro code.
def runcode(xarray, initialdata, data, inputs='inputs.json'):

    with open(inputs, 'w') as file:
        json.dump(data, file, indent=4)

    # Create an HDF5 file
    iofilename = data['iofilename']
    with h5py.File(iofilename, 'w') as file:
        # Create a dataset in the file and write the array to it
        file.create_dataset('initialdata', data=initialdata) 
        file.create_dataset('x', data=xarray) 
        file.attrs.update(data)
        file.attrs['dt'] = data['cfl_max']*(xarray[1] - xarray[0])

    extra_args = []
    for kev, val in options.items():
        if val is None or val == '':
            extra_args += [kev]
        else:
            extra_args += [kev, val]

    command = ['./vischydro', '-inputs', inputs] + extra_args[:] 
    print("Executing command: ", command)
    subprocess.call(command)


# Create a grid of x values based on the data dictionary
def xgrid(data):
    # create an array of linear spaced data with NX points between xmin and xmax
    xmin = data['xmin']
    xmax = data['xmax']
    NX = data['NX']
    xarray = np.linspace(xmin, xmax, NX)
    dx = xarray[1] - xarray[0] 
    return xarray, dx

# Create the initial conditions considered by Pretorious and Pandyas fig 1
def ic1(data, A=0.4, delta=0.1, w=25.):
    xarray, dx = xgrid(data) 
    yarray = A*np.exp(-xarray**2/w) + delta
    initialdata = np.zeros((data['NX'], data['NDOF'])) 
    initialdata[:,2] = yarray
    return xarray, initialdata

# Create the step initial conditions considered by Pretorious and Pandyas.
def ic2(data, eps_l=1.0, eps_r=0.1):
    xarray, dx = xgrid(data) 
    yarray = eps_l * np.piecewise(xarray, [xarray < 0, xarray >= 0], [1, 0]) + eps_r * np.piecewise(xarray, [xarray < 0, xarray >= 0], [0, 1])
    initialdata = np.zeros((data['NX'], data['NDOF'])) 
    initialdata[:,2] = yarray
    return xarray, initialdata


if __name__ == '__main__':
    # The code will run with the default options. The options can be changed by updating the options dictionary.
    options.update({'-ts_exact_final_time': 'INTERPOLATE'})

    # construct the initial data in an array
    xarray, initialdata = ic1(data)

    # Set the values of other paraemters
    data['eta_over_s'] = 3./(4.0*np.pi) 

    # This data file is the one used to communicate the initial conditions and final cons
    data['iofilename'] = 'vischydro_fig1.h5'

    # Run the code, with the initial data and the chosen options
    runcode(xarray, initialdata, data)

    # Open the HDF5 file which is used for input and output
    with h5py.File('vischydro_fig1.h5', 'r') as file:
        # Read the 'finaldata' dataset
        finaldata = file['finaldata'][:]
        initialdatain = file['initialdatain'][:]

        plt.plot(xarray, finaldata[:, 2], xarray, finaldata[:, 0], '--')
        ax = plt.gca()
        ax.set_xlim(-60, 60)
        ax.set_ylim(0.09, 0.25)
        plt.legend([r'$\epsilon$', r'$T^{tt}$'])
        plt.xlabel('x')
        plt.title(r'$\eta$/s = {}/4$\pi$'.format(data['eta_over_s']*4.0*np.pi)) 
        plt.show()
        
