#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy.stats.stats import pearsonr
from AMO.misc_fns import an_ave, spatial_ave

#preliminary analysis
#1. look at maps of SST and THF
#2. look at time series of global annual mean SST and THF
#3. look at maps of SST' and THF' (annual mean anomalies)

#reproduce O'Reilly et al. 2016 result, i.e., that 10-year smoothed SST is positively correlated with THF in extratropical North Atlantic 
#(define AMO_mid as area average of annual mean detrended SST anomalies over the region (60∘–20∘W, 40∘--60∘N)

#make global maps of grid point correlations between deltaSST'/deltat (and delta(SST'^2))/deltat, and THF'.
#1. compute annual anomalies of SST and THF (i.e. subtract global annual mean SST and THF to get SST' and THF')
#2. time-smooth SST' and THF' (as in O'Reilly et al. 2016) using varied window length (e.g. 10 years)
#3. calculate deltaSST'/deltat for deltat equal to the smoothing-window length? 
#4. correlate deltaSST'/deltat with THF' and plot map 
#5.  correlate SST' with THF' and plot map.. positive correlations indicate regions where THF' contributes to the magnitude of SST' growing in time (i.e. dSST'^2/dt > 0, a positive THF feedback)
#6. question: will the maps produced in #4 and #5 look the same? 
#7. can I see the O'Reilly 2016 result in any of these maps?
#8. are positive correlations found in regions where ocean dynamics are known to be important (e.g. boundary currents)

#LONG TERM: do complete surface energy budget analysis (i.e. including cloud radiative effects). 
#look at global atmospheric CRE to quantify component of THF that results from atmospheric CRE.

#fin = '/Users/cpatrizio/data/ECMWF/'
fin = '/Users/cpatrizio/data/MERRA2/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/ocean-atmosphere/'

#fsst = cdms2.open(fin + 'era_interim_moda_SST_1979to2010.nc')
#fTHF = cdms2.open(fin + 'era_interim_moda_THF_1979to2010.nc')

f =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1970to2017.nc')


matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (8,6)})
matplotlib.rcParams.update({'lines.linewidth': 3})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})

ulat = 90
llat = -90

#sst = fsst('sst')
sst = f('TSKINWTR')
sst = sst.subRegion(latitude=(llat, ulat), longitude=(0, 360))
#cdutil.setTimeBoundsMonthly(sst)
#lhf and shf are accumulated over 12 hours (units of J/m^2)
#lhf = fTHF('slhf')/(12*60*60)
lhf = f('EFLUXWTR')
lhf = lhf.subRegion(latitude=(llat, ulat), longitude=(0, 360))
#cdutil.setTimeBoundsMonthly(lhf)
#shf = fTHF('sshf')/(12*60*60)
shf = f('HFLUXWTR')
shf = shf.subRegion(latitude=(llat, ulat), longitude=(0, 360))


#thf is positive down (energy input into ocean surface by surface fluxes)
thf = lhf + shf

#convert to (energy input into atmosphere by surface fluxes)
#thf = -thf
t = sst.getTime().asRelativeTime("months since 1970")
t = np.array([x.value for x in t])
t = np.round(t[0]) + t/12.

tyears = np.arange(np.round(t[0]), np.round(t[-1]))

#compute annual averages
sstan = an_ave(sst)
lhfan = an_ave(lhf)
shfan = an_ave(shf)
thfan = an_ave(thf)

lats = sst.getLatitude()
lons = sst.getLongitude()

sst_globave = spatial_ave(sstan, lats)
thf_globave = spatial_ave(thfan, lats)
lhf_globave = spatial_ave(lhfan, lats)
shf_globave = spatial_ave(shfan, lats)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(211)
ax.plot(tyears, sst_globave)
ax.set_ylabel(r'$\overline{SST}$ (K)')
ax = fig.add_subplot(212)
ax.plot(tyears, thf_globave)
ax.set_ylabel(r'$\overline{THF}$ (W m$^{-2}$)')
ax.set_xlabel('time (years)')
plt.savefig(fout + 'sstglobav_thfglobav_timeseries.pdf')

sstprime = sstan.T - sst_globave
sstprime = sstprime.T
thfprime = thfan.T - thf_globave
thfprime = thfprime.T

par = np.arange(-90.,91.,30.)
mer = np.arange(-180.,181.,60.)

fig.add_subplot(211)
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#deltasstave, lonss = shiftgrid(180, np.ma.average(sstprime, axis=0), lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
levels=np.linspace(-0.1, 0.1, 51)
m.contourf(x, y, np.ma.average(sstprime, axis=0), cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar(label=r'$SST''$ (K)')
ax.set_title(r'$SST''$')
fig.add_subplot(212)
m = Basemap(projection='moll',lon_0=0,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#thfsave, lonss = shiftgrid(180, thfsave, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
levels=np.linspace(100,300,51)
m.contourf(x, y, np.ma.average(thfprime, axis=0), cmap=plt.cm.viridis_r, levels=levels, extend='both')
m.colorbar(label=r'$THF'' (W/m$^{{-2}}$)')
plt.savefig(fout + 'sstprime_thfprime_map.pdf')
plt.close()


