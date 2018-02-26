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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy.stats.stats import pearsonr
from AMO.misc_fns import an_ave, spatial_ave, running_mean

fsst = cdms2.open(fin + 'era_interim_moda_SST_1979to2010.nc')
fTHF = cdms2.open(fin + 'era_interim_moda_THF_1979to2010.nc')

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
N = 11
#dt = int((tdiff - 1)/2.)  
#deltasst = MV.zeros((nt-tdiff+1, nlon, nlat))
#thfs = MV.zeros((nt-tdiff+1, nlon, nlat))

sst_smooth = running_mean(sst_annave, N)
thfs_smooth = running_mean(thf_annave, N)

nt_smooth = sst_smooth.shape[0]

#deltasst = (sst_annave[(tdiff-1):,:,:] - sst_annave[:-(tdiff-1),:,:])/tdiff
#deltasst.setGrid(grid)

#need to construct mask for continents for THF??

ci = (N-1)/2
#thfs = thf_annave[ci:-ci,:]
#thfs.setGrid(grid)

#nadeltasst = deltasst.subRegion(latitude=(0,60), longitude=(280, 360))
#nathfs = thfs.subRegion(latitude=(0,60), longitude=(280,360))

sst_flat = sst_smooth.reshape(nt_smooth, nlat*nlon)
thfs_flat = thfs_smooth.reshape(nt_smooth, nlat*nlon)

zlag = len(sst_flat)


corrs = MV.zeros((2*N, nlon*nlat))
#pvals = MV.zeros(nlon*nlat)
for i in range(nlon*nlat):
    a = (sst_flat[:,i] - np.mean(sst_flat[:,i])) / (np.std(sst_flat[:,i]) * len(sst_flat[:,i]))
    b = (thfs_flat[:,i] - np.mean(thfs_flat[:,i])) / (np.std(thfs_flat[:,i]))
    corrs_alllags = np.correlate(a, b, 'full')
    corrs[:,i] = corrs_alllags[zlag-N:zlag+N]
    #corrs[i], pvals[i] = pearsonr(sst_flat[:,i], thfs_flat[:,i])
    #print 'corr', corrs[i]
    
corrs = corrs.reshape((2*N, nlon, nlat))
#pvals = pvals.reshape((nlon, nlat))


#CALCULATE CORRELATION AT DIFFERENT LAGS?

par = np.arange(-90.,91.,30.)
mer = np.arange(-180.,181.,60.)

corrsplot = corrs[zlag,:,:]

#pattern='...'
plt.figure(figsize=(6,4))
m = Basemap(projection='moll',lon_0=0,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
corrss, lonss = shiftgrid(180, corrsplot, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lonss, lats))
levels=np.linspace(-0.7, 0.7, 71)
m.contourf(x, y, corrss, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, pvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
plt.title('{:1.0f}-year smoothing'.format(N))
plt.savefig(fout + 'thf_sst_corr_{:1.0f}yearsmooth.pdf'.format(N))
plt.close()

sstbar = spatial_ave(sst_smooth, lats)
thfsbar = spatial_ave(thfs_smooth, lats)

ts = np.arange(t[0], t[-1])
tsplot = ts[ci:-ci]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(211)
ax.plot(tsplot, sstbar.getValue()-273.15)
plt.title('{:1.0f}-year smoothing'.format(N))
#ax.set_xlabel('time (years)')
ax.set_ylabel(r'T$_s$ ($^{{\circ}}$C)')
ax = fig.add_subplot(212)
plt.plot(tsplot, thfsbar.getValue())
ax.set_ylabel(r'THF (W m$^{{-2}}$)')
ax.set_xlabel('time (years)')
plt.savefig(fout + 'thf_deltasst_ts_{:1.0f}yearsmooth.pdf'.format(N))
plt.close()












