# Created on Wed Mar 15 11:20:52 2022
# Author: RON93902 @ Mott MacDonald
import math 
import numpy as np
import matplotlib.pyplot as plt

from plxscripting.easy import *
# your server setting
server_port = 10001
server_password = "mottmac"
s_o, g_o = new_server('localhost', server_port, password=server_password)

# import pandas as pd #   not in use
# import easygui      #   not in use
# import pylab        #   not in use

# MMG I/II Hoek-Brown parameters
# USC, mi, GSI, D are the primary input parameters
# mb, s, a are derivative parameters
# USC or qi in the unit of kPa
class HoekBrownModel():
    def __init__(self, ucs=2000, mi=4, gsi=45, disturbance=0):
        self.ucs = ucs  
        self.mi = mi
        self.gsi = gsi
        self.disturbance = disturbance
        self.mb = mi * math.exp((gsi-100)/(28-14*disturbance))
        self.s = math.exp((gsi-100)/(9-3*disturbance))
        self.a = 1/2 + 1/6*(math.exp(-1*gsi/15)-math.exp(-20/3))
    
    # conversion method to calculate equivalent c and phi at given sigma_eff_3 (minor principle effective stress )  
    def convertMC(self, sig_eff_3):
        # normalised sig_3n
        sig_3n = -sig_eff_3/self.ucs
        
        # unfactored cohesion, as a function of sig_3'
        # in kPa
        cohesion =  ( 
                        self.ucs*((1 + 2*self.a)*self.s + (1 - self.a)*self.mb*sig_3n )*  
                        ( self.s + self.mb*sig_3n)**(self.a-1) 
                    ) / (   
                        (1 + self.a)*(2 + self.a)*
                        (math.sqrt(1 + (6*self.a*self.mb*(self.s + self.mb*sig_3n)**(self.a - 1))/((1 + self.a)*(2 + self.a)))) 
        )
        
        # unfactored friction angle, as a function of sig_3'
        # in degrees
        phi = math.degrees(math.asin(
                    ( 6*self.a*self.mb*(self.s + self.mb*sig_3n)**(self.a-1) 
                    ) / ( 
                        2*(1 + self.a)*(2 + self.a) + 6*self.a*self.mb*(self.s + self.mb*sig_3n)**(self.a-1) )
            )
        )
        
        return cohesion, phi

# function to apply partial factors to MC model parameters c' and phi'
# apply partial factor to reduce c' and phi
# default value = 1.0 (unfactored) 
def factorMC(cohesion, phi, partial_factor=1.0):
    # factor down cohesion
    cohesion_factored = cohesion/partial_factor
    
    # factor down phi
    tanphi = math.tan(math.radians(phi))
    phi_factored = math.atan(tanphi)/partial_factor
    return cohesion_factored, phi_factored

# A rectangular region which acts as a lens to sample/manipulate/grid the data for contouring
class SampleGrid():
    def __init__(self,  x_min, x_max, x_inter,  
                        y_min, y_max, y_inter):
        
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        # spacing of sampling grids
        self.x_inter = x_inter
        self.y_inter = y_inter
        self.x_range = np.arange(x_min, x_max, x_inter)
        self.y_range = np.arange(y_min, y_max, y_inter)
        # xx, yy, zz, same size np array with size of (len(x_range), len(y_range))
        # zz is the array to store values for plotting
        self.xx, self.yy, self.vv = self.griddata()

    def griddata(self):
        # xs, ys = np.meshgrid(x_range, y_range, sparse=True)
        xx, yy = np.meshgrid(self.x_range, self.y_range)
        vv = np.zeros_like(xx, dtype=float)  
        return xx, yy, vv

# A model interogation object which extract data from PLAXIS model
# plx_phase is a Plx phase object
class TargetModelPhase():
    def __init__(self, g_o, plx_phase):
        self.g_o = g_o
        self.phase = plx_phase
        self.phasename = plx_phase.Identification.value

    # method to sample sigma3 in the model based on a given grid
    # phase - phase to interact with
    # varray - the numpy array size of (len(x_range), len(y_range)) to store sample values, to get sigma'_3 into varray
    # x_range, y_range are range of position to sample the stress
    def sample_sig3(self, phase, x_range, y_range, varray):
        for j in range(len(x_range)):
            for i in range(len(y_range)):
                x = x_range[j]
                y = y_range[i]
                sig_3 = self.g_o.getsingleresult(phase, g_o.ResultTypes.Soil.SigmaEffective3, (x,y), True)
                varray[i][j] = sig_3
        return varray

# plotting function
# arrange sub plot in 1 row 3 col: left to right, sig_3 --> c' --> phi'
# vv0 - array of sigma_3 values
# vv1 - array of c' values
# vv2 - array of phi values
# headertitle - main fig title (stage in plaxis)
def contour_plot(xx, yy, vv0, vv1, vv2, headertitle):
    fig, axs = plt.subplots(1,3, constrained_layout = True)
    fig.suptitle(headertitle, fontsize=14)
    
    # colorbar legends parameters
    # for sigma3'
    cbar0_min = -540 
    cbar0_max = 100 
    cbar0_step = 40
    # for cohesion
    cbar1_min = 10 
    cbar1_max = 58 
    cbar1_step = 4
    # for friction angle
    cbar2_min = 14 
    cbar2_max = 46 
    cbar2_step = 1

    levels0 = np.arange(cbar0_min, cbar0_max+cbar0_step, cbar0_step)
    levels1 = np.arange(cbar1_min, cbar1_max+cbar1_step, cbar1_step)
    levels2 = np.arange(cbar2_min, cbar2_max+cbar2_step, cbar2_step)

    # plot1 -sigma3 plot
    cs_0 = axs[0].contourf(xx, yy, vv0, levels0, cmap="rainbow")
    axs[0].set_title("sig'_3")
    axs[0].set_ylabel('Y (m)', fontsize = 12, weight = 'bold')
    axs[0].set_xlabel('X (m)', fontsize = 12, weight = 'bold')
    cbar0 = fig.colorbar(cs_0, ax=axs[0], ticks=levels0)
    cbar0.ax.set_yticklabels(['{:.0f}'.format(lvl) for lvl in levels0], fontsize=12, weight = 'bold')
    cbar0.ax.set_title('[kPa]', fontsize=12, weight="bold")
    
    # plot2 - c plot
    cs_1 = axs[1].contourf(xx, yy, vv1, levels1, cmap="rainbow")
    axs[1].set_title("cohesion")
    axs[1].set_ylabel('Y (m)', fontsize = 12, weight = 'bold')
    axs[1].set_xlabel('X (m)', fontsize = 12, weight = 'bold')
    cbar1 = fig.colorbar(cs_1, ax=axs[1], ticks=levels1)
    cbar1.ax.set_yticklabels(['{:.0f}'.format(lvl) for lvl in levels1], fontsize=12, weight = 'bold')
    cbar1.ax.set_title('[kPa]', fontsize=12, weight="bold")
    
    # plot3 - phi' plot
    cs_2 = axs[2].contourf(xx, yy, vv2, levels2, cmap="rainbow")
    axs[2].set_title("friction angle")
    axs[2].set_ylabel('Y (m)', fontsize = 12, weight = 'bold')
    axs[2].set_xlabel('X (m)', fontsize = 12, weight = 'bold')
    cbar2 = fig.colorbar(cs_2, ax=axs[2], ticks=levels2)
    cbar2.ax.set_yticklabels(['{:.0f}'.format(lvl) for lvl in levels2], fontsize=12, weight = 'bold')
    cbar2.ax.set_title('[deg]', fontsize=12, weight="bold")

    # show plots
    plt.show()
    return plt

# main module to run the script
def main():
    
    # initiate a HB model 
    HBmod = HoekBrownModel(ucs=2000, mi=4, gsi=45, disturbance=0)

    # initiate a sampling grid
    sample_grid = SampleGrid(x_min=-23, 
                            x_max=22,
                            x_inter=1, 
                            y_min=50,
                            y_max=73,
                            y_inter=1)
    xarr = sample_grid.xx
    yarr = sample_grid.yy
    varr = sample_grid.vv
    xrng = sample_grid.x_range
    yrng = sample_grid.y_range
    
    # initiate a model data extraction
    plx_phases = [g_o.Phase_8, g_o.Phase_7]
    
    for ph in plx_phases:
        target_model_phase = TargetModelPhase(g_o, ph)
        figtitle = target_model_phase.phasename
        
        # get sigma 3 from plaxis model
        sig3_arr = target_model_phase.sample_sig3(ph, xrng, yrng, varr)
        
        # get cohesion converted from HB model
        # get phi converted from HB model
        c_arr = np.zeros_like(sig3_arr, dtype=float)
        phi_arr = np.zeros_like(sig3_arr, dtype=float)
        for i in range(sig3_arr.shape[0]):
            for j in range(sig3_arr.shape[1]):
                c_arr[i][j], phi_arr[i][j] = HBmod.convertMC(sig3_arr[i][j])
        
        # do plot per phase
        cplot = contour_plot(xarr, yarr, vv0=sig3_arr, vv1=c_arr, vv2=phi_arr, headertitle=figtitle)
        savepath = input('Save to folder:')
        # cplot.savefig(savepath + "\\" + str(ph.Name) + ".png" )

main()
