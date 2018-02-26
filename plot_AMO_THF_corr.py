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
from AMO.misc_fns import an_ave, spatial_ave, calc_AMO, running_mean

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

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]
lons[0] = 0
nlat = len(lats)
nlon = len(lons)

#thf is positive down (energy input into ocean surface by surface fluxes)
thf = lhf + shf

#convert to (energy input into atmosphere by surface fluxes)
#thf = -thf
t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1980 + t/12.

tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

latbounds = [40, 60]

#initial/final indices for base period
baseti = 0
basetf = 20

AMO, sstanom_globe_an, sstanom_na_an = calc_AMO(sst, latbounds, baseti, basetf)

sst_an = an_ave(sst)
thf_an = an_ave(thf)

AMOstd = (AMO - np.ma.mean(AMO))/np.ma.std(AMO)

sstcorrs = MV.zeros((nlat,nlon))
thfcorrs = MV.zeros((nlat,nlon))
sstpvals = MV.zeros((nlat,nlon))

#compute correlation between AMO and sst
for i in range(nlat):
    for j in range(nlon):
         #regr = linear_model.LinearRegression()
         #regr.fit(sst_globe_an[:,i,j], AMO_std)
         slope, intercept, r_value, p_value, std_err = linregress(AMOstd, sst_an[:,i,j])
         sstcorrs[i,j] = r_value
         sstpvals[i,j] = p_value
         slope, intercept, r_value, p_value, std_err = linregress(AMOstd, thf_an[:,i,j])
         thfcorrs[i,j] = r_value
         
#Lagged correlation of THFmid and AMOmid        
THFmid = thf.subRegion(latitude=(latbounds[0],latbounds[1]), longitude=(280,360))
THFmid = spatial_ave(THFmid, THFmid.getLatitude()[:])
THFmid = an_ave(THFmid)

nlag=10
windows = np.arange(1,nlag+1)
lags = np.arange(-nlag,nlag)

ll, ww = np.meshgrid(lags, windows)

THFmid_laggedcorr = np.zeros((len(windows), 2*nlag))
#AMOmid_autocorr = np.zeros((len(windows), 2*nlag))

#commpute lagged correlation between smoothed AMO and THF for different smoothing window lengths
for k, N in enumerate(windows):
    AMOstd_smooth = running_mean(AMOstd, N)
    THFmid_smooth = running_mean(THFmid, N)
    a = (AMOstd_smooth - np.mean(AMOstd_smooth)) / (np.std(AMOstd_smooth) * len(AMOstd_smooth))
    b = (THFmid_smooth - np.mean(THFmid_smooth)) / (np.std(THFmid_smooth))
    lagzero = len(THFmid_smooth)/2
    THFmid_laggedcorr[k,:] = np.correlate(a,b, 'full')[lagzero-nlag:lagzero+nlag]
    #AMOmid_autocorr[k,:] = np.correlate(AMOstd_smooth, AMOstd_smooth, 'full')[lagzero-nlag:lagzero+nlag]
    
    
fig=plt.figure(figsize=(10,8))
ax = fig.add_subplot(211)
ax.contourf(ll, ww, THFmid_laggedcorr, 40, cmap=plt.cm.RdBu_r)
ax.set_title('long term correlation between AMO and THF'.format(N))
ax.set_xlabel('THF lag (years)')
ax.set_ylabel('smoothing (years)')
plt.savefig(fout + 'AMO_thf_corr_smoothlag_hist.pdf')
plt.close()

fig=plt.figure(figsize=(12,10))
ax = fig.add_subplot(211)
ax.plot(tyears, sstanom_globe_an)
ax.set_title(r'global mean SST anomaly (base period: {:3.0f} to {:3.0f})'.format(tyears[baseti], tyears[basetf]))
ax.axhline(0, color='black')
ax = fig.add_subplot(212)
ax.plot(tyears, sstanom_na_an)
ax.axhline(0, color='black')
ax.set_title(r'mean NA SST anomaly ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + 'global_NA_SST_anomaly_timeseries.pdf')
plt.close()

N=5
ci = (N-1)/2
smooth_AMO = running_mean(AMO, N)
plt.figure()
plt.plot(tyears, AMO)
plt.plot(tyears[ci:-ci],smooth_AMO)
plt.title(r'AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.axhline(0, color='black')
plt.savefig(fout + 'AMO_timeseries.pdf')
plt.close()

par = np.arange(-90.,91.,15.)
mer = np.arange(-180.,181.,30.)

lstep = 0.01
levels = np.arange(-1.0, 1.0+lstep, lstep)

plt.figure(figsize=(10,8))
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=280,urcrnrlon=360,resolution='i')
#m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-1.1, 1.1, 43)
m.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
plt.title(r'regression of AMO index ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N) onto SST'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + 'AMO_sst_corr_map.pdf')
plt.close()

plt.figure(figsize=(10,8))
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=280,urcrnrlon=360,resolution='i')
#m = Basemap(projection='moll',lon_0=180,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-1.1, 1.1, 23)
m.contourf(x, y, thfcorrs, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
plt.title(r'regression of AMO index ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N) onto THF'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + 'AMO_thf_corr_map.pdf'.format(nlag))
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
#ax.set_title(r'detrended $T$')
#ax = fig.add_subplot(212)
#m = Basemap(projection='moll',lon_0=180,resolution='i')
#m.drawcoastlines(linewidth=0.1)
#m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
##thfsave, lonss = shiftgrid(180, thfsave, lons, start=False)
#x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-100,100,51)
#m.contourf(x, y, np.ma.average(thfprime, axis=0), cmap=plt.cm.PRGn, levels=levels, extend='both')
#m.colorbar(label=r'$THF - \overline{THF}$ (W/m$^{{-2}}$)')
#ax.set_title(r'detrended $THF$')
#plt.savefig(fout + 'sstprime_thfprime_map.pdf')
#plt.close()


