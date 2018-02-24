#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:24:47 2017

@author: cpatrizio

"""
#direc='/Users/cpatrizio/repos/cloud-radiative-kernels/data/'

fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import glob
import cdtime
import vcs
import cdutil
import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy.stats.stats import pearsonr
from AMO.misc_fns import an_ave, spatial_ave

fsst = cdms2.open(fin + 'era_interim_moda_SST_1979to2010.nc')
fTHF = cdms2.open(fin + 'era_interim_moda_THF_1979to2010.nc')


matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (8,6)})
matplotlib.rcParams.update({'lines.linewidth': 3})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})


sst = fsst('sst')
#cdutil.setTimeBoundsMonthly(sst)
#lhf and shf are accumulated over 12 hours (units of J/m^2)
lhf = fTHF('slhf')/(12*60*60)
#cdutil.setTimeBoundsMonthly(lhf)
shf = fTHF('sshf')/(12*60*60)
#cdutil.setTimeBoundsMonthly(shf)

#fsst.close()
#fTHF.close()

#thf is positive down (energy input into ocean surface by surface fluxes)
thf = lhf + shf

#convert to (energy input into atmosphere by surface fluxes)
thf = -thf

#cdutil.setTimeBoundsMonthly(thf)

#extract North Atlantic sst 
#nasst = sst.subRegion(latitude=(0, 60), longitude=(280, 360))
#nathf = thf.subRegion(latitude=(0, 60), longitude=(280, 360))



#NA average sst over entire time period
#nasst_ave = MV.average(nasst, axis=0)

sst_annave = an_ave(sst)
lhf_annave = an_ave(lhf)
shf_annave = an_ave(shf)
thf_annave = an_ave(thf)


#nalons = nasst.getLongitude()[:]
#nalats = nasst.getLatitude()[:]

#compute spatially weighted means
#sstbar = cdutil.averager(sst, axis='xy', weights='weighted')
#nasstbar = cdutil.averager(nasst, axis='xy', weights='weighted')
#thfbar = cdutil.averager(thf, axis='xy', weights='weighted')
#nathfbar = cdutil.averager(nathf, axis='xy', weights='weighted')

#convert hours since 1900 to years since 1979 (as float)
t = sst.getTime().asRelativeTime("months since 1979")
t = np.array([x.value for x in t])
t = 1979 + t/12.


#CONSTRUCT MAPS OF CORRELATIONS BETWEEN THF AND DELTA SST FOR DIFFERENT DELTA T. 

nlat = sst.shape[2]
nlon = sst.shape[1]
nt = sst_annave.shape[0]

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]
grid = cdms2.createGenericGrid(lats,lons)
    
#NUMBER OF YEARS FOR TIME DIFFERENCING
tdiff = 11
#dt = int((tdiff - 1)/2.)  
#deltasst = MV.zeros((nt-tdiff+1, nlon, nlat))
#thfs = MV.zeros((nt-tdiff+1, nlon, nlat))

deltasst = (sst_annave[(tdiff-1):,:,:] - sst_annave[:-(tdiff-1),:,:])/tdiff
#deltasst.setGrid(grid)

#need to construct mask for continents for THF??

ci = (tdiff-1)/2
thfs = thf_annave[ci:-ci,:]
#thfs.setGrid(grid)

#nadeltasst = deltasst.subRegion(latitude=(0,60), longitude=(280, 360))
#nathfs = thfs.subRegion(latitude=(0,60), longitude=(280,360))

deltasst_flat = deltasst.reshape(nt-tdiff+1, nlat*nlon)
thfs_flat = thfs.reshape(nt-tdiff+1, nlat*nlon)

corrs = MV.zeros(nlon*nlat)
pvals = MV.zeros(nlon*nlat)
for i in range(nlon*nlat):
    corrs[i], pvals[i] = pearsonr(deltasst_flat[:,i], thfs_flat[:,i])
    #print 'corr', corrs[i]
    
corrs = corrs.reshape((nlon, nlat))
pvals = pvals.reshape((nlon, nlat))

par = np.arange(-90.,91.,30.)
mer = np.arange(-180.,181.,60.)

#pattern='...'
plt.figure(figsize=(8,6))
m = Basemap(projection='moll',lon_0=0,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
corrs, lonss = shiftgrid(180, corrs, lons, start=False)
pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lonss, lats))
levels=np.linspace(-0.7, 0.7, 71)
m.contourf(x, y, corrs, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, pvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
plt.title('$\Delta t = {:1.0f}$ years'.format(tdiff))
plt.savefig(fout + 'thf_deltasst_corr_{:1.0f}year.pdf'.format(tdiff))
plt.close()

#corrsbar = cdutil.averager(corrs, axis='xy', weights='weighted')
#plt.plot

deltasstbar = spatial_ave(deltasst, lats)
thfsbar = spatial_ave(thfs, lats)

#nadeltasstbar = weighted_average(nadeltasst)
#nathfsbar = weighted_average(nathfs)

ts = np.arange(t[0], t[-1])
tsplot = ts[ci:-ci]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(211)
ax.plot(tsplot, deltasstbar)
plt.title('$\Delta t = {:1.0f}$ years'.format(tdiff))
#ax.set_xlabel('time (years)')
ax.set_ylabel(r'$\frac{{\Delta T_s}}{\Delta t}}$ (K/year)')
ax = fig.add_subplot(212)
plt.plot(tsplot, thfsbar)
ax.set_ylabel(r'THF (W m$^{{-2}}$)')
ax.set_xlabel('time (years)')
plt.savefig(fout + 'thf_deltasst_ts_{:1.0f}year.pdf'.format(tdiff))
plt.close()

sstbar = spatial_ave(sst_annave-273.15, lats)
fig = plt.figure(figsize=(8,6))
plt.plot(ts, sstbar)
#plt.title('$\Delta t = {:1.0f}$ years'.format(tdiff))
#ax.set_xlabel('time (years)')
plt.ylabel(r'$T_s$ ($^{{\circ}}$C)')
plt.xlabel('time (years)')
plt.savefig(fout + 'sst_ts_{:1.0f}year.pdf'.format(tdiff))
plt.close()


#
#ts = np.arange(t[0], t[-1])
#tsplot = ts[ci:-ci]
#fig = plt.figure(figsize=(6,4))
#ax = fig.add_subplot(211)
#ax.plot(tsplot, nadeltasstbar.getValue())
#plt.title('$\Delta t = {:1.0f}$ years'.format(tdiff))
##ax.set_xlabel('time (years)')
#ax.set_ylabel(r'$\frac{{\Delta T_s}}{\Delta t}}$ (K/year)')
#ax = fig.add_subplot(212)
#plt.plot(tsplot, nathfsbar.getValue())
#ax.set_ylabel(r'THF (W m$^{{-2}}$)')
#ax.set_xlabel('time (years)')
#plt.savefig(fout + 'NA_thf_deltasst_ts_{:1.0f}year.pdf'.format(tdiff))
#plt.close()

#PLOT MAPS OF TIME AVERAGED THF AND DELTA SST OVER ENTIRE TIME PERIOD
deltasstave = MV.average(deltasst, axis=0)
thfsave = MV.average(thfs, axis=0)
fig = plt.figure(figsize=(8,6))
fig.add_subplot(211)
m = Basemap(projection='moll',lon_0=0,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
deltasstave, lonss = shiftgrid(180, deltasstave, lons, start=False)
x, y = m(*np.meshgrid(lonss, lats))
levels=np.linspace(-0.01, 0.01, 51)
m.contourf(x, y, deltasstave, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar(label=r'$\overline{{\frac{{\Delta T_s}}{{\Delta t}}}}$ (K/year)')
plt.suptitle('$\Delta t = {:1.0f}$ years'.format(tdiff))
ax.set_title(r'$\frac{{\Delta T_s}}{{\Delta t}}$')
fig.add_subplot(212)
m = Basemap(projection='moll',lon_0=0,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
thfsave, lonss = shiftgrid(180, thfsave, lons, start=False)
x, y = m(*np.meshgrid(lonss, lats))
levels=np.linspace(100, 400, 51)
m.contourf(x, y, thfsave, cmap=plt.cm.viridis_r, levels=levels, extend='both')
m.colorbar(label=r'$\overline{{THF}}$ (W/m$^{{-2}}$)')
plt.savefig(fout + 'thf_deltasst_map_{:1.0f}year.pdf'.format(tdiff))
plt.close()











