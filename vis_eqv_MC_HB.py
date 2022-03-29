# Created on Wed Mar 15 11:20:52 2022
# Author: RON93902 @ Mott MacDonald
# Refactored by Yitan Lu, March 2022
import math 
import numpy as np
import matplotlib.pyplot as plt

from plxscripting.easy import *
# your server setting
server_port = 10001
server_password = "mottmac"
s_o, g_o = new_server('localhost', server_port, password=server_password)

# MMG I/II Hoek-Brown parameters
# USC, mi, GSI, D are the primary input parameters
# mb, s, a are derivative parameters
# USC or qi in the unit of [kPa]
class HoekBrownModel():
    def __init__(self, ucs=2000, mi=4, gsi=45, disturbance=0):
        self.ucs = ucs  
        self.mi = mi
        self.gsi = gsi
        self.disturbance = disturbance
        self.mb = mi * math.exp((gsi-100)/(28-14*disturbance))
        self.s = math.exp((gsi-100)/(9-3*disturbance))
        self.a = 1/2 + 1/6*(math.exp(-1*gsi/15)-math.exp(-20/3))
    
    # conversion method to calculate equivalent c and phi at given sigma_eff_3 (minor principal effective stress)  
    def convertMC(self, sig_eff_3):
        # normalised sig_3n
        sig_3n = -sig_eff_3/self.ucs
        
        # unfactored cohesion, as a function of sigma_eff_3
        # in [kPa]
        cohesion =  ( 
                        self.ucs*((1 + 2*self.a)*self.s + (1 - self.a)*self.mb*sig_3n )*  
                        ( self.s + self.mb*sig_3n)**(self.a-1) 
                    ) / (   
                        (1 + self.a)*(2 + self.a)*
                        (math.sqrt(1 + (6*self.a*self.mb*(self.s + self.mb*sig_3n)**(self.a - 1))/((1 + self.a)*(2 + self.a)))) 
        )
        
        # unfactored friction angle, as a function of sig_3'
        # in [degrees]
        phi = math.degrees(math.asin(
                    ( 6*self.a*self.mb*(self.s + self.mb*sig_3n)**(self.a-1) 
                    ) / ( 
                        2*(1 + self.a)*(2 + self.a) + 6*self.a*self.mb*(self.s + self.mb*sig_3n)**(self.a-1) )
            )
        )
        
        return cohesion, phi

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

# class object to create contour (heatmap) plot of subplots
# arrange sub plot in 1 row 3 col: left to right, sig_3 --> c' --> phi'
# vv_list - vv_llist is a list of arrays in which vv[0] - sig3 on xx/yy grid, vv[1] - c' on xx/yy grid, vv[2] phi'on xx/yy grid
# headertitle - main fig title (stage in plaxis)
# ftsize - font size used in subplots
class ContourByStage():
    def __init__(self, xx, yy, vv_list, headertitle, ftsize):
        self.xx = xx
        self.yy = yy
        self.vv_list = vv_list
        self.headertitle = headertitle
        self.ftsize = ftsize

        # list corresponding to subplot position/sequence
        self.subplot_seq = ["sig_3'", "c'", "phi'"]
        
        # list correspond to the unit of subplot
        self.unit_dict = ["[kPa]", "[kPa]", "[degree]"] 
                        

    # method to set up the plots in 1 row 3c col subplot style
    # hardwired sup-plot title with font size = 14
    def contour_plot(self):
        self.fig, self.axs = plt.subplots(1,3, constrained_layout = True)
        self.fig.suptitle(self.headertitle, fontsize=14)
        return plt

    # method to fill plot with sub plots
    # input cbar_min, cbar_max and cbar_step to control the legend styles
    # input col_num (0, 1, 2) to identify plot type and location (sig3, c, or phi)
    def sub_plots(self, col_num: int, 
                        cbar_min, cbar_max, cbar_step):

        plot_type = self.subplot_seq[col_num]
        unit_txt = self.unit_dict[col_num]
        # define colour bar params
        clevels = np.arange(cbar_min, cbar_max+cbar_step, cbar_step)
        
        cs = self.axs[col_num].contourf(self.xx, self.yy, self.vv_list[col_num], clevels, cmap="rainbow")
        self.axs[col_num].set_title(plot_type)
        self.axs[col_num].set_ylabel('Y (m)', fontsize = self.ftsize, weight = 'bold')
        self.axs[col_num].set_xlabel('X (m)', fontsize = self.ftsize, weight = 'bold')
        self.axs[col_num].set_aspect('equal')
        self.cbar = self.fig.colorbar(cs, ax=self.axs[col_num], ticks=clevels)
        self.cbar.ax.set_yticklabels(['{:.0f}'.format(lvl) for lvl in clevels], fontsize=self.ftsize, weight = 'bold')
        self.cbar.ax.set_title(unit_txt, fontsize=self.ftsize, weight="bold")

# function to apply partial factors to MC model parameters c' and phi'
# apply partial factor to reduce c' and phi
# default value = 1.0 (unfactored) 
# return factored cohesion in [kPa]
# return factored phi' in [degree]
def factorMC(cohesion, phi, partial_factor=1.0):
    # factor down cohesion
    cohesion_factored = cohesion/partial_factor
    
    # factor down phi
    tanphi = math.tan(math.radians(phi))
    phi_factored = math.degrees(
                                math.atan(tanphi)/partial_factor)

    return cohesion_factored, phi_factored

# function to return nearest integer by user definition
def round_to_base(value, base:int) -> int:
    return int(value) - int(value) % int(base)

# main module to run the script
def main():
    
    # initiate a HB model 
    HBmod = HoekBrownModel(ucs=2000, mi=4, gsi=45, disturbance=0)

    # initiate a sampling grid
    sample_grid = SampleGrid(x_min=-23, 
                            x_max=22,
                            x_inter=1, 
                            y_min=64,
                            y_max=73,
                            y_inter=1)
    xarr = sample_grid.xx
    yarr = sample_grid.yy
    varr = sample_grid.vv
    xrng = sample_grid.x_range
    yrng = sample_grid.y_range
    
    # initiate data extraction from defined list of stages
    plx_phases = [g_o.Phase_7, g_o.Phase_12]
    
    for ph in plx_phases:
        target_model_phase = TargetModelPhase(g_o, ph)
        figtitle = target_model_phase.phasename
        
        # get sigma 3 from plaxis model
        sig3_arr = target_model_phase.sample_sig3(ph, xrng, yrng, varr)
        
        # get cohesion converted from HB model
        c_arr = np.zeros_like(sig3_arr, dtype=float)
        # get phi converted from HB model
        phi_arr = np.zeros_like(sig3_arr, dtype=float)
        
        for i in range(sig3_arr.shape[0]):
            for j in range(sig3_arr.shape[1]):
                c_arr[i][j], phi_arr[i][j] = HBmod.convertMC(sig3_arr[i][j])
        
        # do plot per phase by initiating CoutourByStage object
        cplot = ContourByStage(xarr, yarr, vv_list=[sig3_arr, c_arr, phi_arr], headertitle=figtitle, ftsize=12)
        current_chart = cplot.contour_plot()
        
        # generator obj to generate 3 sub plots with index 0, 1, 2
        plt_num = iter(range(3))

        # generate sigma3 sub plot
        cplot.sub_plots(next(plt_num), 
                        cbar_min=-550, 
                        cbar_max=100, 
                        cbar_step=50)
        # generate cohesion sub plot
        cplot.sub_plots(next(plt_num), 
                        cbar_min=round_to_base(math.floor(np.nanmin(c_arr)), base=5), 
                        cbar_max=round_to_base(math.ceil(np.nanmax(c_arr)), base=5), 
                        cbar_step=5)
        # generate phi sub plot
        cplot.sub_plots(next(plt_num), 
                        cbar_min=math.floor(np.nanmin(phi_arr)), 
                        cbar_max=math.ceil(np.nanmax(phi_arr)), 
                        cbar_step=1)

        current_chart.show()
        input()
        # savepath = input('Save to folder:')
        # cplot.savefig(savepath + "\\" + str(ph.Name) + ".png" )
        current_chart.close()

main()
