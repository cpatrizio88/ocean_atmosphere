#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:38:53 2017

@author: cpatrizio
"""
import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import glob
from scipy import signal
from sklearn import linear_model
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, shiftgrid
from scipy.stats.stats import pearsonr, linregress
from ocean_atmosphere.misc_fns import an_ave, spatial_ave, running_mean, calc_AMO
#from misc_fns import spatial_ave, an_ave

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'figure.figsize': (12,10)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 18})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})


fin = '/Users/cpatrizio/data/HadSST/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'
            
#fsst = cdms2.open(fin + 'HadSST.3.1.1.0.median.nc')
#fsst = cdms2.open(fin + 'HadISST1_sst.nc')
fsst = cdms2.open(fin + 'HadISST_sst_SAVE.03.12.2010.nc')
#fsstabs = cdms2.open(fin + 'absolute.nc')
#fTHF = cdms2.open(fin + 'era_interim_moda_THF_1979to2010.nc')

fsst2 = cdms2.open(fin + 'AMO_HADLEY.1870-2010.CLM_1901-1970.nc')

AMO_WARM = fsst2('AMO_WARM')
AMO_WARM_REMOVED = fsst2('AMO_WARM_REMOVED')
AMO_WARM_REMOVED_SMTH = fsst2('AMO_WARM_REMOVED_SMTH')
SST_GLOBAL_MEAN = fsst2('SST_GLOBAL_MEAN')

sst = fsst('sst')

ti=31
tf=100
latbounds = [0,60]

times = sst.getTime()[:]

tyears = np.arange(1870, 2010)

tplot=tyears

AMO, sstanom_anave, nasstanom_anave = calc_AMO(sst, latbounds, ti, tf)

N=11
ci = (N-1)/2

AMO_smooth = running_mean(AMO, N)

AMO_std = (AMO - np.ma.mean(AMO))/np.ma.std(AMO)

AMO_test_std = (AMO_WARM_REMOVED - np.ma.mean(AMO_WARM_REMOVED)/np.ma.std(AMO_WARM_REMOVED))

sst_na = sst.subRegion(latitude=(latbounds[0], latbounds[1]),longitude=(280,360))
nalats = sst_na.getLatitude()[:]
#sst_na_ave = cdutil.averager(sst_na, axis='xy', weights='weighted')
sst_na_an = an_ave(sst_na)
#sst_na_an_detr = signal.detrend(sst_na_an, axis=0)

#sst_globe = sst.subRegion(latitude=(-60,60))
sst_an = an_ave(sst)
#sst_globe_an = an_ave(sst_globe)
#sst_globe_an_detr = signal.detrend(sst_globe_an, axis=0)
#sst_globe_ave = cdutil.averager(sst_globe, axis='xy', weights='weighted')
#sst_globe_ave_an = an_ave(sst_globe_ave)
#sstbase_globe = MV.average(sst_globe_an[ti:tf], axis=0).getValue()


#sst_globeanom_an = (sst_globe_an.T - sst_globe_ave_an).T

#sstglobeanom_na_an_ave = spatial_ave(sstglobeanom_na_an, nalats)


#lhf_annave = an_ave(lhf)
#shf_annave = an_ave(shf)
#thf_na_an = an_ave(thf_na)

nlon = sst.shape[2]
nlat = sst.shape[1]
nt = AMO.shape[0]

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]

#grid = cdms2.createGenericGrid(lats,lons)
#sst_flat = sst_na_an.reshape(nt,nlat*nlon)
#thf_flat = thf_na_an.reshape(nt,nlat*nlon)


sstcorrs = MV.zeros((nlat,nlon))
#thfcorrs = MV.zeros(nlon*nlat)
sstpvals = MV.zeros((nlat,nlon))
sstpvals_test = MV.zeros((nlat, nlon))
#thfpvals = MV.zeros(nlon*nlat)

sstcorrs_test = MV.zeros((nlat, nlon))

nlag = 5

for i in range(nlat):
    for j in range(nlon):
         #regr = linear_model.LinearRegression()
         #regr.fit(sst_globe_an[:,i,j], AMO_std)
         slope, intercept, r_value, p_value, std_err = linregress(AMO_std, sst_an[:,i,j])
         sstcorrs[i,j] = r_value
         sstpvals[i,j] = p_value
         #sstcorrs[i,j], sstpvals[i,j] = pearsonr(sst_globe_an[:,i,j], AMO_std)
         #sstcorrs_test[i,j], sstpvals_test[i,j] = pearsonr(sst_globe_an[:,i,j], AMO_test_std[:-1])
         #thfcorrs[i], thfpvals[i] = pearsonr(thf_flat[:-nlag,i], AMO_std[nlag:])
    
    
#sstcorrs = sstcorrs.reshape(nlat,nlon)
    
#par = np.arange(0.,90,15.)
#mer = np.arange(270,375,15.)
        
par = np.arange(-90.,90,30.)
mer = np.arange(0,360,60.)
    
corrsplot = sstcorrs



#plt.figure(figsize=(6,4))
#m = Basemap(projection='moll',lon_0=0,llcrnrlat=0,urcrnrlat=60,llcrnrlon=280,urcrnrlon=360,resolution='i')
#m.drawcoastlines(linewidth=0.1)
#m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
##m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
##corrss, lonss = shiftgrid(180, corrsplot, lons, start=False)
##pvals, lonss = shiftgrid(180, pvals, lons, start=False)
#x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-0.5, 0.5, 21)
#m.contourf(x, y, sst_globe_an[100,:,:], cmap=plt.cm.RdBu_r, levels=levels, extend='both')
#m.colorbar()
##m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
#m.fillcontinents(color='white')
##plt.title(r'regression of AMO index ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N) onto SST'.format(latbounds[0], latbounds[1]))
##plt.savefig(fout + 'AMO_sst_corr_map.pdf')
#plt.show()
#plt.close()

lstep = 0.01
levels = np.arange(-1.0, 1.0+lstep, lstep)
    
plt.figure(figsize=(10,8))
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=75,llcrnrlon=-90,urcrnrlon=0,resolution='i')
#m = Basemap(projection='moll',lon_0=0,resolution='i')
m.drawcoastlines(linewidth=0.1)
m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
#corrss, lonss = shiftgrid(180, sstcorrs, lons, start=False)
#pvals, lonss = shiftgrid(180, pvals, lons, start=False)
x, y = m(*np.meshgrid(lons, lats))
m.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
m.colorbar()
#m.contourf(x, y, sstpvals, colors='none', levels=np.linspace(0.01,1.0,50), alpha=0.2, hatch='...')
m.fillcontinents(color='white')
plt.title(r'regression of AMO index ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N) onto SST'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + 'HadSST1_AMO_sst_corr_map.pdf')
plt.close()

#    
#plt.figure(figsize=(6,4))
#m = Basemap(projection='moll',lon_0=0,llcrnrlat=0,urcrnrlat=60,llcrnrlon=280,urcrnrlon=360,resolution='i')
#m.drawcoastlines(linewidth=0.1)
#m.drawparallels(par, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
##m.drawmeridians(mer, dashes=[100,0.001], labels=[1,0,0,1], linewidth=0.1)
#m.drawmeridians(mer, dashes=[100,0.001], linewidth=0.1)
##corrss, lonss = shiftgrid(180, corrsplot, lons, start=False)
##pvals, lonss = shiftgrid(180, pvals, lons, start=False)
#x, y = m(*np.meshgrid(lons, lats))
#levels=np.linspace(-0.5, 0.5, 21)
#m.contourf(x, y, sstcorrs_test, cmap=plt.cm.RdBu_r, levels=levels, extend='both')
#m.colorbar()
###m.contourf(x, y, pvals, colors='none', levels=np.linspace(0.2,1.0,50), alpha=0.2, hatch='...')
#m.fillcontinents(color='white')
#plt.title(r'regression of AMO index ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N) onto SST'.format(latbounds[0], latbounds[1]))
#plt.savefig(fout + 'AMO_sst_corr__test_map.pdf')
#plt.close()

#plt.title('regression of AMO$_{{mid}}$ onto SST')
##plt.savefig(fout + 'AMO_sst_corr_map.pdf')
#plt.close()


tplot_test = tplot

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot(tplot, sstanom_anave)
#plt.plot(tplot_test, SST_GLOBAL_MEAN.getValue()[:-1])
ax = plt.gca()
#ax.fill_between(tplot, 0, sstanom_anave, where= sstanom_anave>0, color='red')
#ax.fill_between(tplot, 0, sstanom_anave, where= sstanom_anave<0, color='blue')
ax.axhline(0, color='black')
ax.set_ylim(-1,1)
#ax.set_xlabel('time (years)')
ax.set_ylabel(r'SST ($^{{\circ}}$C)')
ax.set_title('Global SST anomaly')

ax = fig.add_subplot(212)
ax.plot(tplot, nasstanom_anave)
#plt.plot(tplot_test, AMO_WARM.getValue()[:-1])
ax = plt.gca() 
#ax.fill_between(tplot, 0, nasstanom_anave, where= nasstanom_anave>0, color='red')
#ax.fill_between(tplot, 0, nasstanom_anave, where= nasstanom_anave<0, color='blue')
ax.set_title('NA SST anomaly')
ax.set_ylim(-1,1)
ax.set_xlabel('time (years)')
ax.set_ylabel(r'SST ($^{{\circ}}$C)')
ax.axhline(0, color='black')
plt.savefig(fout + 'HadSST1_global_NA_SST_anomaly_timeseries.pdf')
plt.close()

nasstanom_detr_anave = nasstanom_anave - sstanom_anave
#nasstanom_detr_ave = cdutil.averager(nasstanom_detr.getValue(), axis='xy', weights='weighted')
#nasstanom_detr_ave = spatial_ave(nasstanom_detr, nalats)
#nasstanom_detr_an = an_ave(nasstanom_detr_ave)

#nasstanom_detr_ananve = nasstanom_detr_anave.getValue()


N=11
AMO_smooth = running_mean(AMO, N)
ci = (N-1)/2

plt.figure()
ax = plt.gcf().gca()
plt.plot(tplot, AMO, linewidth=1)
#plt.plot(tplot_test, AMO_WARM_REMOVED.getValue()[:-1])
#ax.fill_between(tplot, 0, AMO, where= AMO>0, color='red')
#ax.fill_between(tplot, 0, AMO, where= AMO<0, color='blue')
plt.axhline(0, color='black')
#plt.ylim(-1,1)
#plt.xlabel('time (years)')
#plt.ylabel(r'SST ($^{{\circ}}$C)')
#plt.title('AMO')
#plt.savefig(fout + 'HadSST1_AMO.pdf')
#plt.close()
plt.plot(tplot[ci:-ci], AMO_smooth)
#plt.plot(tplot_test, AMO_WARM_REMOVED_SMTH.getValue()[:-1])
ax.fill_between(tplot[ci:-ci], 0, AMO_smooth, where=AMO_smooth>0, color='red')
ax.fill_between(tplot[ci:-ci], 0, AMO_smooth, where=AMO_smooth<0, color='blue')
plt.title('AMO')
plt.xlabel('time (years)')
plt.ylabel(r'SST ($^{{\circ}}$C)')
plt.ylim(-1,1)
plt.savefig(fout + 'HadSST1_AMO.pdf')
plt.close()




