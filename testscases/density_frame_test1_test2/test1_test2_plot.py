#
# Script to plot the results from the density frame paper 
# Derek Teaney, November 2025
#
# The usage is:
#
# python test1_test2_plot.py --testcase test1
# or
# python test1_test2_plot.py --testcase test2
#
# The purpose of this script is to compare the results from the
# current vischydro code to the results presented in the density frame paper.
# The results from the density frame paper are stored in h5 files
# named nbys_...h5 while the current results are stored in Jnbys_...h5.
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import json
import h5py as h5



def getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width, J= False):
    """ Helper function for getdata, returns the filename based on parameters """
    if J:
        s='Jnbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    else:
        s='nbys_{}_d_{}_A_{}_w_{}'.format(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width)
    return s.replace('.', 'd')

def getdata(fourpietabys,gaussian_const, gaussian_amplitude, gaussian_width, tag='Ttt', J= False):
    """ Get the data from the h5 files for given parameters """
    variables = {'Ttt':0, 'Ttx':1, 'eps':2, 'ux':3 } 
    name = getnames(fourpietabys, gaussian_const, gaussian_amplitude, gaussian_width, J=J)
    name0 = getnames(0.0, gaussian_const, gaussian_amplitude, gaussian_width, J=J)
    data = json.load(open(name + '.json'))

    with h5.File(name + '.h5', 'r') as file:
        finaldata = file['finaldata'][:,variables[tag]]
        x = file['x'][:]

    try :
        with h5.File(name0 + '.h5', 'r') as file:
            finaldata_ideal = file['finaldata'][:,variables[tag]]
 
    except:
        name0 = getnames(0, gaussian_const, gaussian_amplitude, gaussian_width)
        with h5.File(name0 + '.h5', 'r') as file:
            finaldata_ideal = file['finaldata'][:,variables[tag]]   
    
    return x, finaldata,  finaldata_ideal, data

def plot_test1():
    gaussian_const = 0.12
    gaussian_amplitude = 0.48
    gaussian_width = 25.0


    # Make the plot of Ttt vs position for test2
    fig, axs = plt.subplots(1,1, figsize=(10,10))
    
    plt.sca(axs)
    plt.gca().set_xlim(-75, 75)

    x, df, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width)

    plt.plot(x, ideal, 'k', linewidth=0.5, label='Ideal hydro')

    plt.plot(x, df, 'C1', linewidth=0.5, label='$4\\pi\\eta/s=1$')

    x, df, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )


    x, df, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C2', linewidth=0.5, label='$4\\pi\\eta/s=3$')
    x, df, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )

    x, df, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C3', linewidth=0.5, label='$4\\pi\\eta/s=6$')
    x, df, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )

    x, df, ideal, data = getdata(20.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C4', linewidth=0.5, label='$4\\pi\\eta/s=20$')
    x, df, ideal, data = getdata(20.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )

    # set the title 
    plt.gca().set_title('Density Frame Test: Lines Reference. Points are Current code')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    plt.legend()
    plt.tight_layout()

def plot_test2():
    gaussian_const = 0.06
    gaussian_amplitude = 9.6
    gaussian_width = 25.0


    # Make the plot of Ttt vs position for test2
    fig, axs = plt.subplots(1,1, figsize=(10,10))
    
    plt.sca(axs)
    plt.gca().set_xlim(-75, 75)

    x, df, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width)

    plt.plot(x, ideal, 'k', linewidth=0.5, label='Ideal hydro')

    plt.plot(x, df, 'C1', linewidth=0.5, label='$4\\pi\\eta/s=1$')

    x, df, ideal, data = getdata(1.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )


    x, df, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C2', linewidth=0.5, label='$4\\pi\\eta/s=3$')
    x, df, ideal, data = getdata(3.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )

    x, df, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C3', linewidth=0.5, label='$4\\pi\\eta/s=6$')
    x, df, ideal, data = getdata(6.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )

    x, df, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width)
    plt.plot(x, df, 'C4', linewidth=0.5, label='$4\\pi\\eta/s=10$')
    x, df, ideal, data = getdata(10.0, gaussian_const, gaussian_amplitude, gaussian_width, J=True)
    plt.plot(x, df, '.' )


    # set the title 
    plt.gca().set_title('Density Frame Test: Lines Reference. Points are Current code')
    plt.gca().set_xlabel('x')
    plt.gca().set_ylabel('$T^{tt}$')
    plt.legend()
    plt.tight_layout()

    #plt.savefig('smoothtest2.pdf')

if __name__ == '__main__':
    # Use argparse to select which test to plot
    import argparse
    parser = argparse.ArgumentParser(description='Plot the results from test1 or test2')
    parser.add_argument('--testcase', type=str, required=True, help='Which testcase to plot: test1 or test2')
    args = parser.parse_args()     
    if args.testcase == 'test1':
        plot_test1()
    elif args.testcase == 'test2':
        plot_test2()
    else:
        print("Invalid testcase. Please choose 'test1' or 'test2'.")
    plt.show()
