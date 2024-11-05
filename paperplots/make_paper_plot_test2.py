import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import h5py
plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 9})
# Open the HDF5 file which is used for input and output
# old eta/s
#listetas=[1.56,0.53,0.19]
# eta/s from 2407.09605
listetas=[0.180, 0.513,1.48]
etas4pi=4*np.pi*np.array(listetas)

listlambda=[20,10,5]
cols = ['r','g','b']

# define figure widths
fw=4
gr=1.618
fig1, ax1 = plt.subplots(figsize=(fw,fw/1.3))
fig2, ax2 = plt.subplots(figsize=(fw,fw/1.3))

Nz=150
nug=16




for i,et in enumerate(listetas):
    with h5py.File('test2_eta%g.h5' %et, 'r') as file:
        EKTdata=np.loadtxt("./new_EKT/test2_L%d_gluon_Tmunu_vs_time.out" % listlambda[i])
        nx,ny = EKTdata.shape
        EKTdata.shape=(nx//Nz,Nz,ny)
        # Read the 'finaldata' dataset
        finaldata = file['finaldata'][:]
        #initialdata = file['initialdata'][:]
        #solution = file['solution'][:]
        #print("final time", file.attrs['final_time'])
        finaltime=file.attrs['final_time']
        initialtime=file.attrs['initial_time']
        xarray = file['x'][:]
        initialdatain = file['initialdatain'][:]
        #plt.plot(xarray, finaldata[:, 2], xarray, finaldata[:, 0], '--')
        sl=-1
        # label=r'$4\pi\eta/s=%.1f$' % (4*np.pi*float(file.attrs['eta_over_s']))
        ax1.plot(xarray, finaldata[:, 0], color="mediumblue", ls="-", label= 'Density Frame $4\pi\eta/s={:.1f},{:.1f},{:.1f}$'.format(etas4pi[0],etas4pi[1],etas4pi[2]) if i==0 else "" )
        ax1.plot(EKTdata[0,:,18], nug*EKTdata[sl,:,8], color="darkorange", ls="-.", label='Kinetic Theory $\lambda={:d},{:d},{:d}$'.format(listlambda[0],listlambda[1],listlambda[2]) if i==0 else ""  )

        ax2.plot(xarray, finaldata[:, 1], color="mediumblue", ls="-", label= 'Density Frame\n $4\pi\eta/s={:.1f},{:.1f},{:.1f}$'.format(etas4pi[0],etas4pi[1],etas4pi[2]) if i==0 else "" )
        ax2.plot(EKTdata[0,:,18], nug*EKTdata[sl,:,11], color="darkorange", ls="-.", label='Kinetic Theory $\lambda={:d},{:d},{:d}$'.format(listlambda[0],listlambda[1],listlambda[2]) if i==0 else "" )
    
with h5py.File('test2_eta%g.h5' % 0, 'r') as file:
        finaldata = file['finaldata'][:]
        finaltime=file.attrs['final_time']
        initialtime=file.attrs['initial_time']
        xarray = file['x'][:]
        initialdatain = file['initialdatain'][:]
        ax1.plot(xarray, finaldata[:, 0], color="black", ls="-", label= 'Ideal Hydro')
        ax2.plot(xarray, finaldata[:, 1], color="black", ls="-", label= 'Ideal Hydro')

# plot analytical initial conditions;
x=np.linspace(-75.1,75.1,500)

delta=0.06
A=9.6
w=5

tc=finaltime
analytic=delta+np.sqrt(np.pi)/2*A*w/(2*tc)*(sp.erf((x+tc)/w)-sp.erf((x-tc)/w))
analyticflux= A*w/(4*tc**2)*( w*( np.exp(-(tc+x)**2/w**2  ) - np.exp(-(tc-x)**2/w**2) ) +   np.sqrt(np.pi)*x*(sp.erf((x+tc)/w)-sp.erf((x-tc)/w))  )

ax1.plot(x, analytic,ls=":", color='forestgreen', label=r'Free Streaming')
ax2.plot(x, analyticflux,ls=":",color='forestgreen', label=r'Free Streaming')

ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$T^{tt}$')
ax1.set_xlim(-75, 75)
ax1.set_xticks([-75,-50,-25,0,25,50,75])
ax1.set_ylim(0.0, 4)
#ax1.set_title(r'EKT final time $t = {:3g}$'.format(EKTdata[sl,0,0])) 
ax1.legend(frameon=False,loc="upper left")

ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$T^{tz}$')
ax2.set_xlim(-75, 75)
ax2.set_xticks([-75,-50,-25,0,25,50,75])
ax2.set_ylim(-2.5, 2.5)
#ax2.set_title(r'EKT final time $t = {:3g}$'.format(EKTdata[sl,0,0])) 
ax2.legend(frameon=False, loc="upper left")
fig1.tight_layout() 
fig1.savefig("plot_test2_Ttt_EKT.pdf")
fig2.tight_layout() 
fig2.savefig("plot_test2_Ttz_EKT.pdf")

        

