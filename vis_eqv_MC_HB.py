# Created on Wed Mar 15 11:20:52 2022
# Author: RON93902 @ Mott MacDonald

from plxscripting.easy import *
# your server setting
server_port = 10001
server_password = "12345"
s_o, g_o = new_server('localhost', server_port, password=server_password)
import easygui
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import math

# MMG I/II Hoek-Brown parameters
UCS = 2000 # unit kPa
mi = 4
GSI = 45
D = 0 

mb = mi * math.exp((GSI-100)/(28-14*D))
s = math.exp((GSI-100)/(9-3*D))
a = 1/2 + 1/6*(math.exp(-1*GSI/15)-math.exp(-20/3))

# define boundary of interest
x_min = -23
x_max = 22
y_min = 50
y_max = 73
x_inter = 1
y_inter = 1

# store extracted results in a dataframe
x_range = np.arange(x_min, x_max, x_inter)
y_range = np.arange(y_min, y_max, y_inter)
xs, ys = np.meshgrid(x_range, y_range, sparse=True)
xx, yy = np.meshgrid(x_range, y_range)
sample = np.zeros_like(xx)

# specify the phases of interest to loop
# Phase_2 - Stage3 EL1_84.3 - index number 2 in Phases[] 
# Phase_8 - Stage5 EL2_82.3 - index number 4 in Phases[] 
# Phase_5 - Stage7 EL3_75.4 - index number 6 in Phases[] 
# Phase_7 - Stage9 ExcavationFL - index number 8 in Phases[]
phases = [2,4,6,8]
# create an empty list storing dataframe in each phase
df_sig3 = []
df_cohesion = []
df_degree = []

for p in phases:
    # define a list that will store the extracted results
    xysig3 = []
    xycohesion = []
    xydegree = []
    # sample (x,y) point by looping through the boundary of interest
    for x in range(x_min, x_max, x_inter):
        for y in range(y_min, y_max, y_inter):
            # interact with PLAXIS output to get result out of plaxis
            sig_3 = g_o.getsingleresult(g_o.Phases[p], g_o.ResultTypes.Soil.SigmaEffective3, (x,y), True)
            # convert to factored MOhr Coulomb parameters
            sig_3n = -sig_3/UCS
            degree = math.degrees(0.8*math.tan(math.asin((6*a*mb*(s+mb*sig_3n)**(a-1))/(2*(1+a)*(2+a)+6*a*mb*(s+mb*sig_3n)**(a-1)))))
            cohesion = 0.8*(UCS*((1+2*a)*s+(1-a)*mb*sig_3n)*(s+mb*sig_3n)**(a-1))/((1+a)*(2+a)*(math.sqrt(1+(6*a*mb*(s+mb*sig_3n)**(a-1))/((1+a)*(2+a)))))
            xysig3.append([x,y,sig_3])
            xycohesion.append([x,y,cohesion])
            xydegree.append([x,y,degree])
    # populate sigma3, factored cohesion, factored degree in a reset df for each phase
    df1 = pd.DataFrame(data = sample, index = ys[0:,0], columns = xs[0,0:])
    df2 = pd.DataFrame(data = sample, index = ys[0:,0], columns = xs[0,0:])
    df3 = pd.DataFrame(data = sample, index = ys[0:,0], columns = xs[0,0:])
    for i in range(len(xysig3)):
        df1.loc[xysig3[i][1], xysig3[i][0]] = xysig3[i][2]
        df2.loc[xycohesion[i][1], xycohesion[i][0]] = xycohesion[i][2]
        df3.loc[xydegree[i][1], xydegree[i][0]] = xydegree[i][2]
    # reconstructure to show elevation in the dataframe
    df1 = df1[::-1]
    df2 = df2[::-1]
    df3 = df3[::-1]    
    df_sig3.append(df1)
    df_cohesion.append(df2)
    df_degree.append(df3)

# present phase data stored in list    
fig1, axs1 = plt.subplots(nrows=2,ncols=2, figsize=(16,9),sharex='col', sharey='row')
(ax1, ax2), (ax3, ax4) = axs1
fig2, axs2 = plt.subplots(nrows=2,ncols=2, figsize=(16,9),sharex='col', sharey='row')
(ax5, ax6), (ax7, ax8) = axs2
fig3, axs3 = plt.subplots(nrows=2,ncols=2, figsize=(16,9),sharex='col', sharey='row')
(ax9, ax10), (ax11, ax12) = axs3

# define a font size
ft = 12

# define colorbar range and scale interval
# for stress
cbar1_min = -540 
cbar1_max = 100 
cbar1_step = 40
# for cohesion
cbar2_min = 10 
cbar2_max = 58 
cbar2_step = 4
# for friction angle
cbar3_min = 14 
cbar3_max = 60 
cbar3_step = 1
cbar3_step1 = 4

levels1 = np.arange(cbar1_min, cbar1_max+cbar1_step, cbar1_step)
levels2 = np.arange(cbar2_min, cbar2_max+cbar2_step, cbar2_step)
levels3 = np.arange(cbar3_min, cbar3_max+cbar3_step, cbar3_step)
levels31 = np.arange(cbar3_min, cbar3_max+cbar3_step, cbar3_step1) # for colorbar legend only

# add subplots
f1 = ax1.contourf(xx,yy[::-1],df_sig3[0], levels=levels1, cmap="rainbow")
f2 = ax2.contourf(xx,yy[::-1],df_sig3[1], levels=levels1, cmap="rainbow")
f3 = ax3.contourf(xx,yy[::-1],df_sig3[2], levels=levels1, cmap="rainbow")
f4 = ax4.contourf(xx,yy[::-1],df_sig3[3], levels=levels1, cmap="rainbow")

f5 = ax5.contourf(xx,yy[::-1],df_cohesion[0], levels=levels2, cmap="rainbow")
f6 = ax6.contourf(xx,yy[::-1],df_cohesion[1], levels=levels2, cmap="rainbow")
f7 = ax7.contourf(xx,yy[::-1],df_cohesion[2], levels=levels2, cmap="rainbow")
f8 = ax8.contourf(xx,yy[::-1],df_cohesion[3], levels=levels2, cmap="rainbow")

f9 = ax9.contourf(xx,yy[::-1],df_degree[0], levels=levels3, cmap="rainbow")
f10 = ax10.contourf(xx,yy[::-1],df_degree[1], levels=levels3, cmap="rainbow")
f11 = ax11.contourf(xx,yy[::-1],df_degree[2], levels=levels3, cmap="rainbow")
f12 = ax12.contourf(xx,yy[::-1],df_degree[3], levels=levels3, cmap="rainbow")

# show subplot titles indicating stages
ax1.set_title('Phase_2: Stage3 EL1_84.3', fontsize = ft)
ax1.set_ylabel('Elevation (m)', fontsize = ft, weight = 'bold')
ax1.tick_params(labelsize = ft)
ax2.set_title('Phase_8: Stage5 EL2_82.3', fontsize = ft)
ax3.set_title('Phase_5: Stage7 EL3_75.4', fontsize = ft)
ax3.set_ylabel('Elevation (m)', fontsize = ft, weight = 'bold')
ax3.set_xlabel('X (m)', fontsize = ft, weight = 'bold')
ax3.tick_params(labelsize = ft)
ax4.set_title('Phase_7: Stage9 ExcavationFL', fontsize = ft)
ax4.set_xlabel('X (m)', fontsize = ft, weight = 'bold')
ax4.tick_params(labelsize = ft)

ax5.set_title('Phase_2: Stage3 EL1_84.3', fontsize = ft)
ax5.set_ylabel('Elevation (m)', fontsize = ft, weight = 'bold')
ax5.tick_params(labelsize = ft)
ax6.set_title('Phase_8: Stage5 EL2_82.3', fontsize = ft)
ax7.set_title('Phase_5: Stage7 EL3_75.4', fontsize = ft)
ax7.set_ylabel('Elevation (m)', fontsize = ft, weight = 'bold')
ax7.set_xlabel('X (m)', fontsize = ft, weight = 'bold')
ax7.tick_params(labelsize = ft)
ax8.set_title('Phase_7: Stage9 ExcavationFL', fontsize = ft)
ax8.set_xlabel('X (m)', fontsize = ft, weight = 'bold')
ax8.tick_params(labelsize = ft)

ax9.set_title('Phase_2: Stage3 EL1_84.3', fontsize = ft)
ax9.set_ylabel('Elevation (m)', fontsize = ft, weight = 'bold')
ax9.tick_params(labelsize = ft)
ax10.set_title('Phase_8: Stage5 EL2_82.3', fontsize = ft)
ax11.set_title('Phase_5: Stage7 EL3_75.4', fontsize = ft)
ax11.set_ylabel('Elevation (m)', fontsize = ft, weight = 'bold')
ax11.set_xlabel('X (m)', fontsize = ft, weight = 'bold')
ax11.tick_params(labelsize = ft)
ax12.set_title('Phase_7: Stage9 ExcavationFL', fontsize = ft)
ax12.set_xlabel('X (m)', fontsize = ft, weight = 'bold')
ax12.tick_params(labelsize = ft)

# show legend
cbar1 = fig1.colorbar(f1, ax=axs1.ravel().tolist(), ticks=levels1)
cbar1.ax.set_yticklabels(['{:.0f}'.format(x) for x in levels1], fontsize = ft, weight = 'bold')
cbar1.ax.set_title('[kPa]', fontsize = ft, weight="bold")

cbar2 = fig2.colorbar(f5, ax=axs2.ravel().tolist(), ticks=levels2)
cbar2.ax.set_yticklabels(['{:.0f}'.format(x) for x in levels2], fontsize = ft, weight = 'bold')
cbar2.ax.set_title('[kPa]', fontsize = ft, weight="bold")

cbar3 = fig3.colorbar(f9, ax=axs3.ravel().tolist(), ticks=levels31)
cbar3.ax.set_yticklabels(['{:.0f}'.format(x) for x in levels31], fontsize = ft, weight = 'bold')
cbar3.ax.set_title('[degree]', fontsize = ft, weight="bold")

# shared properties for subplots
fig1.suptitle('Effective Sigma_3 for MMG I/II during Excavation', fontsize = ft, weight = 'bold')
fig2.suptitle('Converted Factored Cohesion for MMG I/II during Excavation (Factor: 1.25)', fontsize = ft, weight = 'bold')
fig3.suptitle('Converted Factored Friction Angle for MMG I/II during Excavation (Factor: 1.25)', fontsize = ft, weight = 'bold')

# show contour plot
plt.show()