import runvischydro as rnv
import numpy as np
import pprint

# Script to run test1 or test2 gaussian viscous hydro runs used in the first
# numerical density frame paper

# Run either test1 or test2 based on command line arguments. The usage is:
# python test1_test2_run.py --testcase test1
# or
# python test1_test2_run.py --testcase test2

# The command to run the vischydro code is assumed to be ../vischydro/vischydro
# but can be changed here:
runcommand = '../../vischydro'

# Possible runcommands are for example, 'mpiexec -n 1 path_to_vischydro/vischydro' 


def getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width):
    s='Jnbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    return s.replace('.', 'd')

def rungaussian(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width, tmax):
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)

    # run viscous hydro in the density frame
    rnv.data['final_time'] = tmax
    rnv.data['xmin'] = -80
    rnv.data['xmax'] = 80
    rnv.data['NX'] = 513
    rnv.data['cfl_max'] = 0.1
    rnv.data['eta_over_s'] = fourpietabys/(4.0*np.pi)
    rnv.data['iofilename'] = name + '.h5'

    rnv.options.update({'-ts_exact_final_time': 'INTERPOLATE'})
    pprint.pprint(rnv.data)
    xarray, initialdata = rnv.ic1(rnv.data, A=gaussian_amplitude, delta=gaussian_const, w=gaussian_width) 
    rnv.runcode(xarray, initialdata, rnv.data, inputs=name + '.json', runcommand=runcommand, actually_run=True)

if __name__ == '__main__':
    
    # Run either test1 or test2 based on command line arguments. The usage is:
    # python test1_test2_run.py --testcase test1
    # or
    # python test1_test2_run.py --testcase test2
    import argparse
    parser = argparse.ArgumentParser(description='Run test1 or test2 gaussian viscous hydro runs')
    parser.add_argument('--testcase', type=str, required=True, help='Which testcase to run: test1 or test2')
    args = parser.parse_args()

    if args.testcase == 'test1':
        etabys = [0.0, 1.0, 3.0, 6.0, 20.0]
        gaussian_const = 0.12
        gaussian_amplitude = 0.48
        gaussian_width = 25.0
    elif args.testcase == 'test2':
        etabys = [0.0, 1.0, 3.0, 6.0, 10.0, 12.0]
        gaussian_const = 0.06
        gaussian_amplitude = 9.6
        gaussian_width = 25.0

    tmax = 50.
    for fourpietabys in etabys:
        rungaussian(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width, tmax)
    
