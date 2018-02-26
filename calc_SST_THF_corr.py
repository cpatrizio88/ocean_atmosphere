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
from scipy.stats.stats import pearsonr, linregress
from AMO.misc_fns import an_ave, spatial_ave, running_mean

#preliminary analysis
#1. look at maps of SST and THF
#2. look at time series of global annual mean SST and THF
#3. look at maps of SST' and THF' (annual mean anomalies)

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

f =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')


matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (8,6)})
matplotlib.rcParams.update({'lines.linewidth': 3})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})

ulat = 90
llat = -90

tskip = 6

#sst = fsst('sst')
sst = f('TSKINWTR')
sst = sst.subRegion(latitude=(llat, ulat), longitude=(0, 360))
sst = sst[tskip:,:]
#cdutil.setTimeBoundsMonthly(sst)
#lhf and shf are accumulated over 12 hours (units of J/m^2)
#lhf = fTHF('slhf')/(12*60*60)
lhf = f('EFLUXWTR')
lhf = lhf.subRegion(latitude=(llat, ulat), longitude=(0, 360))
lhf = lhf[tskip:,:]
#cdutil.setTimeBoundsMonthly(lhf)
#shf = fTHF('sshf')/(12*60*60)
shf = f('HFLUXWTR')
shf = shf.subRegion(latitude=(llat, ulat), longitude=(0, 360))
shf = shf[tskip:,:]


#thf is positive down (energy input into ocean surface by surface fluxes)
thf = lhf + shf

#convert to (energy input into atmosphere by surface fluxes)
#thf = -thf
t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1981 + t/12.

tyears = np.arange(np.round(t[0]), np.round(t[-1]))

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]

nlat = len(lats)
nlon = len(lons)

sst_anave = an_ave(sst)
thf_anave = an_ave(thf)
sst_globeave = spatial_ave(sst_anave, lats)
thf_globeave = spatial_ave(thf_anave, lats)
sstprime = sst_anave.T - sst_globeave
sstprime = sstprime.T
thfprime = thf_anave.T - thf_globeave
thfprime = thfprime.T


#UNCOMMENT FOR MONTHLY CORRELATIONS
#windows = np.array([3,5,11,25,49,121])
windows = np.array([3,5,7,11])
nwindow = len(windows)

corrs = MV.zeros((nwindow, nlat, nlon))
corrs_deltasst = MV.zeros((nwindow, nlat, nlon))

#calculate correlation between sst and thf for different smoothing windows 
#(should have negative correlation for short windows, positive correlation for long windows?)
for k, N in enumerate(windows):
    #UNCOMMENT FOR MONTHLY CORRELATIONS
    #sst_smooth = running_mean(sst, N)
    sst_smooth = running_mean(sstprime, N)
    #thf_smooth = running_mean(thf, N)
    thf_smooth = running_mean(thfprime, N)
    for i in range(nlat):
        for j in range(nlon):
         #regr = linear_model.LinearRegression()
         #regr.fit(sst_globe_an[:,i,j], AMO_std)
         slope, intercept, r_value, p_value, std_err = linregress(sst_smooth[:,i,j], thf_smooth[:,i,j])
         corrs[k,i,j]=r_value
         deltasst = sst[(N-1):,i,j] - sst[:-(N-1),i,j]
         ci = (N-1)/2
         thfs = thf[ci:-ci,i,j]
         slope, intercept, r_value, p_value, std_err = linregress(deltasst, thfs)
         corrs_deltasst[k,i,j] = r_value


par = np.arange(-90.,91.,15.)
mer = np.arange(-180.,181.,30.)

lindx=np.where(lats > 0)[0][0]
uindx = -1
corrs_NHmean = spatial_ave(corrs[:,lindx:uindx,:], lats[lindx:uindx])

corrs_deltasst_NHmean = spatial_ave(corrs_deltasst[:,lindx:uindx,:], lats[lindx:uindx])

plt.figure(1)
plt.plot(windows, corrs_NHmean)
plt.xlabel('smoothing (years)')
plt.title('NH mean correlation between smoothed SST and THF')
plt.savefig(fout + 'THF_sst_corr_smooth_globemean.pdf')
plt.close()

plt.figure(2)
plt.plot(windows, corrs_deltasst_NHmean)
plt.xlabel(r'$\Delta t$ (years)')
plt.title('NH mean correlation between $\Delta$SST and THF')
plt.savefig(fout + 'THF_deltasst_corr_globemean.pdf')
plt.close()

lstep = 0.01
levels = np.arange(-1.0, 1.0+lstep, lstep)
         
plt.figure(figsize=(10,8))
#m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=280,urcrnrlon=360,resolution='i')
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-1.1, 1.1, 43)
m.contourf(x, y, corrs[0,:,:], cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
#plt.title(r'correlation between {:2.1f}-year smoothed SST and THF'.format(windows[0]/12.))
#plt.savefig(fout + 'THF_sst_corr_{:3.0f}monthsmooth_map.pdf'.format(windows[0]))
plt.title(r'correlation between {:2.1f}-year smoothed SST and THF'.format(windows[0]))
plt.savefig(fout + 'THF_sst_corr_{:3.0f}yearsmooth_map.pdf'.format(windows[0]))
plt.close()

plt.figure(figsize=(10,8))
#m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=280,urcrnrlon=360,resolution='i')
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-1.1, 1.1, 43)
m.contourf(x, y, corrs[3,:,:], cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
#plt.title(r'correlation between {:2.1f}-year smoothed SST and THF'.format(windows[4]/12.))
#plt.savefig(fout + 'THF_sst_corr_{:3.0f}monthsmooth_map.pdf'.format(windows[4]))
plt.title(r'correlation between {:2.1f}-year smoothed SST and THF'.format(windows[3]))
plt.savefig(fout + 'THF_sst_corr_{:3.0f}yearsmooth_map.pdf'.format(windows[3]))
plt.close()

plt.figure(figsize=(10,8))
#m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=280,urcrnrlon=360,resolution='i')
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-1.1, 1.1, 43)
m.contourf(x, y, corrs_deltasst[0,:,:], cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
#plt.title(r'correlation between $\Delta$SST and THF, $\Delta t$ = {:3.1f} years'.format(windows[0]/12.))
#plt.savefig(fout + 'THF_deltasst_corr_{:3.0f}month_map.pdf'.format(windows[0]))
plt.title(r'correlation between $\Delta$SST and THF, $\Delta t$ = {:3.1f} years'.format(windows[0]))
plt.savefig(fout + 'THF_deltasst_corr_{:3.0f}year_map.pdf'.format(windows[0]))
plt.close()

plt.figure(figsize=(10,8))
#m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=280,urcrnrlon=360,resolution='i')
m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-1.1, 1.1, 43)
m.contourf(x, y, corrs_deltasst[3,:,:], cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
#plt.title(r'correlation between $\Delta$SST and THF, $\Delta t$ = {:3.1f} years'.format(windows[4]/12.))
#plt.savefig(fout + 'THF_deltasst_corr_{:3.0f}month_map.pdf'.format(windows[4]))
plt.title(r'correlation between $\Delta$SST and THF, $\Delta t$ = {:3.1f} years'.format(windows[3]))
plt.savefig(fout + 'THF_deltasst_corr_{:3.0f}year_map.pdf'.format(windows[3]))
plt.close()








#compute annual averages

#sstan = an_ave(sst)
#lhfan = an_ave(lhf)
#shfan = an_ave(shf)
#thfan = an_ave(thf)
#
#lats = sst.getLatitude()
#lons = sst.getLongitude()
#lons[0] = 0
#
#sst_globave = spatial_ave(sstan, lats)
#thf_globave = spatial_ave(thfan, lats)
#lhf_globave = spatial_ave(lhfan, lats)
#shf_globave = spatial_ave(shfan, lats)
#
#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(211)
#ax.plot(tyears, sst_globave)
#ax.set_ylabel(r'global mean $T$ (K)')
#ax = fig.add_subplot(212)
#ax.plot(tyears, thf_globave)
#ax.set_ylabel(r'global mean $THF$ (W m$^{-2}$)')
#ax.set_xlabel('time (years)')
#plt.savefig(fout + 'sstglobav_thfglobav_timeseries.pdf')
#plt.close()
#
#sstprime = sstan.T - sst_globave
#sstprime = sstprime.T
#thfprime = thfan.T - thf_globave
#thfprime = thfprime.T
#
#par = np.arange(-90.,91.,30.)
#mer = np.arange(-180.,181.,60.)
#
#fig = plt.figure(figsize=(10,8))
#ax = fig.add_subplot(211)
#m = Basemap(projection='moll',lon_0=180,resolution='i')
#m.drawcoastlines(linewidth=0.1)
#m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
##deltasstave, lonss = shiftgrid(180, np.ma.average(sstprime, axis=0), lons, start=False)
#x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-20, 20, 51)
#m.contourf(x, y, np.ma.average(sstprime, axis=0), cmap=plt.cm.RdBu_r, levels=levels, extend='both')
#m.colorbar(label=r'$T - \overline{T}$ (K)')
#ax.set_title(r'mean $T$ anomaly')
#ax = fig.add_subplot(212)
#m = Basemap(projection='moll',lon_0=180,resolution='i')
#m.drawcoastlines(linewidth=0.1)
#m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
##thfsave, lonss = shiftgrid(180, thfsave, lons, start=False)
#x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-100,100,51)
#m.contourf(x, y, np.ma.average(thfprime, axis=0), cmap=plt.cm.viridis_r, levels=levels, extend='both')
#m.colorbar(label=r'$THF - \overline{THF}$ (W/m$^{{-2}}$)')
#ax.set_title(r'mean $THF$ anomaly')
#plt.savefig(fout + 'sstprime_thfprime_map.pdf')
#plt.close()
#
#
