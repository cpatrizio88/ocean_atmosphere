#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
#from netCDF4 import Dataset
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats.stats import pearsonr, linregress
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from ocean_atmosphere.misc_fns import an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, calc_NA_globeanom3D, detrend_separate, detrend_common

#fin = '/Users/cpatrizio/data/ECMWF/'
fin = '/Users/cpatrizio/data/MERRA2/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

#fsst = cdms2.open(fin + 'era_interim_moda_SST_1979to2010.nc')
#fTHF = cdms2.open(fin + 'era_interim_moda_THF_1979to2010.nc')

fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fcf3D = cdms2.open(fin + 'MERRA2_cldfrac3D_monthly1980to2017.nc')
#fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
#matplotlib.rcParams.update({'figure.autolayout': True})

maxlat = 70
minlat = -70

maxlon = 360
minlon = 0

tskip = 6

#sst = fsst('sst')
sst = fsst('TSKINWTR')
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:,:]

cf = fcf3D['CLOUD']

#lhf = fsst('EFLUXWTR')
#shf = fsst('HFLUXWTR')
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)
#thf = lhf + shf

#ps = fSLP('SLP')
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#ps = ps[tskip:,:]/1e2

#LW_net_surf = radfile['LWGNT']
#LW_net_surf_cs = radfile('LWGNTCLR')
#SW_net_surf = radfile['SWGNT']
#SW_net_surf_cs = radfile('SWGNTCLR')
#LW_net_TOA = radfile['LWTUP']
#lwnettoa_cs = radfile('LWTUPCLR')
#swnettoa = radfile('SWTNT')
#swnettoa_cs = radfile('SWTNTCLR')

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

#Q_net_surf = LW_net_surf + SW_net_surf

#CRE_surf = Q_net_surf - Q_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH AMO
#field = Q_net_surf
#ftitle = r'Q$_{net,surf}$'
#fsave = 'Qnetsurf'

#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

#field = thf
#ftitle = r'THF'
#fsave = 'thf'
#units = r'W m$^{-2}$'


#field = Q_net_surf
#ftitle = r'Q$_{net,surf}$'
#fsave = 'Qnetsurf'

#cf = cffile['MDSCLDFRCTTL']
#cf = cf[tskip:,:]

#field = cf*100.
#ftitle = r'$f_{total}$'
#fsave = 'ftotal'
#units = '%'

field = cf*100.
ftitle = r'$f$'
fsave = 'cldfrac3D'
units = '%'

#True for detrending data, False for raw data
detr=True


field = field.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
field = field[tskip:,:]

p = field.getLevel()[:]
nplev = len(p)

lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
nt = field.shape[0]
lons[0] = 0
nlat = len(lats)
nlon = len(lons)

grid = cdms2.createGenericGrid(lats,lons)

#horizontally interpolate SST to coarser 3D field grid 
sst = sst.regrid(grid, regridTool="esmf", regridMethod = "linear")

t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1980 + t/12.

tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#initial/final years for base period 
baseti = 0
basetf = 10

sst = an_ave(sst)
#CRE_surf_an = an_ave(thf)
#ps_an = an_ave(ps)
field = an_ave(field)

nt = sst.shape[0]

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst, params = detrend_separate(sst)
 #ps_an, params = detrend_separate(ps_an)
 field, params = detrend_separate(field)


sst_globe_an = spatial_ave(sst, lats)
field_globe_an = spatial_ave(field, lats)

#subtract global annual mean to isolate processes in NA
sstprime = sst.T - sst_globe_an
sstprime = sstprime.T

fieldprime = field.T - np.tile(field_globe_an.T, (nlon, nlat, 1,1))
fieldprime = fieldprime.T

#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5
ci = (N_map-1)/2
ltlag = 5
stlag = 1

lagmax=11
lags = np.arange(-lagmax,lagmax+1)

field_lt = running_mean(fieldprime, N_map)
field_st = fieldprime[ci:-ci,:] - field_lt


nt_lt = field_lt.shape[0]


#EDIT BOUNDS FOR AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
#latitude bounds for SST averaging
latboundar = np.array([[0,60],[0,20],[20,45],[45,60]])
#latboundar = np.array([[0,20],[20,45],[45,60]])
#latboundar = np.array([[0,60]])


for latbounds in latboundar:
    
    AMO, sstanom_globe_an, sstanom_na_an = calc_NA_globeanom(sst, latbounds, lats, lons, baseti, basetf)
    NAfield2, fieldanom_globe_an, fieldanom_na_an = calc_NA_globeanom3D(field, latbounds, lats, lons, baseti, basetf)
    
    sstcorrs = MV.zeros((nlat,nlon))
    fieldcorrs = MV.zeros((nplev, nlat*nlon))
    fieldcorrs_lt = MV.zeros((nplev, nlat*nlon))
    fieldcorrs_st = MV.zeros((nplev, nlat*nlon))
    
    fieldlagcorrs = np.zeros((len(lags), nplev, nlat*nlon))
    fieldlagcorrs_lt = np.zeros((len(lags), nplev, nlat*nlon))
    fieldlagcorrs_st = np.zeros((len(lags), nplev, nlat*nlon))
    
    AMO_lt = running_mean(AMO, N_map)
    AMO_st = AMO[ci:-ci] - AMO_lt
    
    scaler = StandardScaler()
    AMOstd = scaler.fit_transform(AMO.reshape(-1,1))
    AMOstd_lt = scaler.fit_transform(AMO_lt.reshape(-1,1))
    AMOstd_st = scaler.fit_transform(AMO_st.reshape(-1,1))
    
    #need to normalize in this manner in order to get pearson correlation coefficient from np.correlate
    AMOstd2 = (AMO - np.mean(AMO)) / (np.std(AMO) * len(AMO))
    AMOstd_lt2 = (AMO_lt - np.mean(AMO_lt)) / (np.std(AMO_lt) * len(AMO_lt))
    AMOstd_st2 = (AMO_st - np.mean(AMO_st)) / (np.std(AMO_st) * len(AMO_st))
    
    #calculate SST pattern of AMO without looping
    sstprime_g = sstprime.reshape(nt, nlat*nlon)
    clf = linear_model.LinearRegression()
    clf.fit(AMOstd.reshape(-1,1), sstprime_g)
    sstcorrs = clf.coef_.reshape(nlat, nlon)
    
    fieldprime_temp = fieldprime.reshape(nt, nplev, nlat*nlon)
    field_lt_temp = field_lt.reshape(nt_lt, nplev, nlat*nlon)
    field_st_temp = field_st.reshape(nt_lt, nplev, nlat*nlon)
    
    lonbounds = [280,359.5]
        
    NAminlati = np.where(lats > latbounds[0])[0][0]
    NAmaxlati = np.where(lats > latbounds[1])[0][0]
    NAminloni = np.where(lons > lonbounds[0])[0][0]
    NAmaxloni = np.where(lons > lonbounds[1])[0]
    
    if len(NAmaxloni) == 0:
        NAmaxloni=-1
    else:
        Namaxloni = NAmaxloni[0]

    NAlats = lats[NAminlati:NAmaxlati]
    NAfield = spatial_ave(fieldprime[:,:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], NAlats)
    
    NAfield_laggedcorr_lt = np.zeros((len(p), len(lags)))
    NAfield_laggedcorr_st = np.zeros((len(p), len(lags)))
    #    
    #compute correlation between long-term/short-term AMO and 3D field
    print r'calculating correlations between AMO and {:s}...'.format(ftitle)
    for i in range(nplev):         
    
         print 'pressure', p[i]
           
         #sstprime_g = sstprime[:,i,:]
         fieldprime_g = fieldprime_temp[:,i,:]
         field_lt_g = field_lt_temp[:,i,:]
         field_st_g = field_st_temp[:,i,:]
        
         #clf = linear_model.LinearRegression()
         #clf.fit(AMOstd.reshape(-1,1), sstprime_g)
         #sstcorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd.reshape(-1,1), fieldprime_g)
         fieldcorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_lt.reshape(-1,1), field_lt_g)
         fieldcorrs_lt[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_st.reshape(-1,1), field_st_g)
         fieldcorrs_st[i,:] = np.squeeze(clf.coef_)
         
         for lag in lags:
             
             scaler = StandardScaler()
             fieldstd = scaler.fit_transform(fieldprime_g)
             fieldstd_lt = scaler.fit_transform(field_lt_g)
             fieldstd_st = scaler.fit_transform(field_st_g)
        
             fieldclf = linear_model.LinearRegression()
             fieldclf_lt = linear_model.LinearRegression()
             fieldclf_st = linear_model.LinearRegression()
        
             #THF LAGS SST
             if lag > 0:
                fieldclf.fit(AMOstd[:-lag], fieldstd[lag:,:])
                fieldclf_lt.fit(AMOstd_lt[:-lag], fieldstd_lt[lag:,:])
                fieldclf_st.fit(AMOstd_st[:-lag], fieldstd_st[lag:,:])
        
            #THF LEADS SST
             elif lag < 0: 
                fieldclf.fit(AMOstd[-lag:], fieldstd[:lag,:])
                fieldclf_lt.fit(AMOstd_lt[-lag:], fieldstd_lt[:lag,:])
                fieldclf_st.fit(AMOstd_st[-lag:], fieldstd_st[:lag,:])
        
             else:
                fieldclf.fit(AMOstd, fieldstd)
                fieldclf_lt.fit(AMOstd_lt, fieldstd_lt)
                fieldclf_st.fit(AMOstd_st, fieldstd_st)
        
             fieldlagcorrs[lag+lagmax,i,:] = np.squeeze(fieldclf.coef_)
             fieldlagcorrs_lt[lag+lagmax,i,:] = np.squeeze(fieldclf_lt.coef_)
             fieldlagcorrs_st[lag+lagmax,i,:] = np.squeeze(fieldclf_st.coef_)

         
         #commpute lagged correlation between AMO and NA THF for specified RM window (N_map) at pressure level i
         NAfield_p = NAfield[:,i]
         NAfield_lt = running_mean(NAfield_p, N_map)
         NAfield_st = NAfield_p[ci:-ci] - NAfield_lt 
         NAfieldstd_lt = (NAfield_lt - np.mean(NAfield_lt)) / (np.std(NAfield_lt))
         NAfieldstd_st = (NAfield_st - np.mean(NAfield_st)) / (np.std(NAfield_st))
         NAfieldlaggedcorr_lt_temp = np.correlate(NAfieldstd_lt, AMOstd_lt2, 'full')
         NAfieldlaggedcorr_st_temp = np.correlate(NAfieldstd_st, AMOstd_st2, 'full')
        
         lagzero = len(NAfieldlaggedcorr_lt_temp)/2
         NAfield_laggedcorr_lt[i,:] = NAfieldlaggedcorr_lt_temp[lagzero-lagmax:lagzero+lagmax+1]
         NAfield_laggedcorr_st[i,:] = NAfieldlaggedcorr_st_temp[lagzero-lagmax:lagzero+lagmax+1]
         
                  
    fieldcorrs = fieldcorrs.reshape(nplev, nlat, nlon)
    fieldcorrs_lt = fieldcorrs_lt.reshape(nplev, nlat, nlon)
    fieldcorrs_st = fieldcorrs_st.reshape(nplev, nlat, nlon)
         
    fieldlagcorrs = fieldlagcorrs.reshape(len(lags), nplev, nlat, nlon)
    fieldlagcorrs_lt = fieldlagcorrs_lt.reshape(len(lags), nplev, nlat, nlon)
    fieldlagcorrs_st = fieldlagcorrs_st.reshape(len(lags), nplev, nlat, nlon)
    
    ##Plot AMO
    fig=plt.figure(figsize=(16,14))
    fig.tight_layout()
    ax = fig.add_subplot(311)
    plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
    ax.plot(tyears, sstanom_globe_an)
    if detr:
        ax.set_ylim(-0.6,0.6)
    else:
        ax.set_ylim(-1,1)
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    #ax.fill_between(tyears, 0, sstanom_globe_an, where= sstanom_globe_an>0, color='red')
    #ax.fill_between(tyears, 0, sstanom_globe_an, where= sstanom_globe_an<0, color='blue')
    ax.set_title(r'global mean SST (base period: {:3.0f} to {:3.0f})'.format(tyears[baseti], tyears[basetf]))
    ax.axhline(0, color='black')
    ax = fig.add_subplot(312)
    ax.plot(tyears, sstanom_na_an)
    #ax.fill_between(tyears, 0, sstanom_na_an, where= sstanom_na_an>0, color='red')
    #ax.fill_between(tyears, 0, sstanom_na_an, where= sstanom_na_an<0, color='blue')
    if detr:
        ax.set_ylim(-0.6,0.6)
    else:
        ax.set_ylim(-1,1)
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    ax.axhline(0, color='black')
    ax.set_title(r'NA mean SST')
    #plt.savefig(fout + 'MERRA2_global_NA_SST_anomaly_timeseries.pdf')
    #plt.close()
    
    AMO_smooth = running_mean(AMO, N_map)
    ci = (N_map-1)/2
    AMO_st = AMO[ci:-ci] - AMO_smooth
    ax = fig.add_subplot(313)
    #plt.figure()
    #ax=plt.gcf().gca()
    ax.plot(tyears, AMO, label='AMO')
    ax.plot(tyears[ci:-ci],AMO_smooth,label='{:1.0f}-yr RM'.format(N_map))
    #plt.plot(tyears[ci:-ci], AMO_st, label='short-term residual')
    #ax.fill_between(tyears[ci:-ci], 0, AMO_smooth, where= AMO_smooth>0, color='red')
    #ax.fill_between(tyears[ci:-ci], 0, AMO_smooth, where= AMO_smooth<0, color='blue')
    ax.set_title(r'AMO (NA mean SST - global mean SST)'.format(latbounds[0], latbounds[1]))
    ax.axhline(0, color='black')
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    ax.legend()
    ax.set_xlabel('time (years)')
    plt.savefig(fout + 'MERRA2_AMO_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    #Plot horizontally averaged NA 3D field vs time.
     
    tt, pp = np.meshgrid(tyears, p)
    
    
    cflevels = np.arange(-3.0, 3.0+0.25, 0.25)
    cfanomlevels = np.arange(-1.5, 1.5+0.05, 0.05)
    if fsave == 'cldfrac3D':
        fieldlevels = cflevels
        anomlevels = cfanomlevels
    else:
        fieldlevels = 30
        anomlevels = 30
    
    fig=plt.figure(figsize=(16,18))
    plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
    ax = fig.add_subplot(411)
    h=ax.contourf(tt, pp, fieldanom_globe_an.T, levels=fieldlevels, cmap=plt.cm.RdBu_r, extend='both')
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    ax.set_ylabel('pressure (hPa)')
    ax.set_title(r'global mean {:s} (base period: {:3.0f} to {:3.0f})'.format(ftitle, tyears[baseti], tyears[basetf]))
    fig.colorbar(h, ax=ax, label=r'{:s} ({:s})'.format(ftitle, units))
    ax = fig.add_subplot(412)
    h=ax.contourf(tt, pp, fieldanom_na_an.T, 30, levels=fieldlevels, cmap=plt.cm.RdBu_r, extend='both')
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    ax.set_title(r'NA mean {:s}'.format(ftitle))
    fig.colorbar(h, ax=ax, label=r'{:s} ({:s})'.format(ftitle, units))
    
    NAfield_smooth = running_mean(NAfield2, N_map)
    
    tt_smooth, pp_smooth = np.meshgrid(tyears[ci:-ci], p)
    
    ax = fig.add_subplot(413)
    h = ax.contourf(tt, pp, NAfield2.T, levels=anomlevels, cmap=plt.cm.RdBu_r, extend='both')
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    plt.title(r'NA mean {:s} - global mean {:s}'.format(ftitle, ftitle))
    ax.set_ylabel('pressure (hPa)')
    #ax.set_xlabel('time (years)')
    fig.colorbar(h, ax=ax, label=r'{:s} ({:s})'.format(ftitle, units))
    #plt.savefig(fout + 'MERRA2_{:s}_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(fsave, latbounds[0], latbounds[1], str(detr)[0]))
    #plt.close()
    
    ax = fig.add_subplot(414)
    h = ax.contourf(tt_smooth, pp_smooth, NAfield_smooth.T, levels=anomlevels, cmap=plt.cm.RdBu_r, extend='both')
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    plt.title(r'NA mean {:s} - global mean {:s} ({:1.0f}-yr RM)'.format(ftitle, ftitle, N_map))
    ax.set_ylabel('pressure (hPa)')
    ax.set_xlabel('time (years)')
    fig.colorbar(h, ax=ax, label=r'{:s} ({:s})'.format(ftitle, units))
    plt.savefig(fout + 'MERRA2_{:s}_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
        
    weights = np.cos(np.deg2rad(lats))
    
    fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs[:,:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=-1)
    fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_lt[:,:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=-1)
    fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_st[:,:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=-1)
    
    fieldlagcorrs_z = np.ma.average(fieldlagcorrs_zonalave, axis=-1, weights=weights[NAminlati:NAmaxlati])
    fieldlagcorrs_lt_z = np.ma.average(fieldlagcorrs_lt_zonalave, axis=-1, weights=weights[NAminlati:NAmaxlati])
    fieldlagcorrs_st_z = np.ma.average(fieldlagcorrs_st_zonalave, axis=-1, weights=weights[NAminlati:NAmaxlati])
    
    lagg, pp = np.meshgrid(lags, p)
    
    #Plot lagged correlation between long-term AMO and horizontally averaged NA 3D field
    fig=plt.figure(figsize=(22,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(131)
    h = ax.pcolor(lagg, pp, fieldlagcorrs_z.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    ax.axvline(0, color='k')
    ax.set_title('correlation of AMO with NA {:s}'.format(ftitle))
    ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
    ax.set_ylabel('pressure (hPa)')
    #ax.set_ylim(0,60)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(132)
    h= ax.pcolor(lagg, pp, fieldlagcorrs_lt_z.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    ax.axvline(0, color='k')
    ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
    ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
    #ax.set_ylim(0,60)
    ax = fig.add_subplot(133)
    h = ax.pcolor(lagg, pp, fieldlagcorrs_st_z.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    ax.axvline(0, color='k')
    ax.set_title('short-term correlation'.format(N_map))
    ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
    #ax.set_ylim(0,60)
    fig.colorbar(h, ax=ax, orientation="vertical")
    plt.savefig(fout + 'MERRA2_AMO_{:s}_lagcorr_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()

    
    #Plot maps of SST and THF patterns associated with AMO
    #CHANGE THIS FOR MAP PROJECTION
    prj = cart.crs.PlateCarree()
    bnds = [-90, 0, 0, 60]
    
    #latitude/longitude labels
    par = np.arange(-90.,91.,15.)
    mer = np.arange(-180.,180.,15.)
    
    lstep = 0.01
    levels = np.arange(-1.0, 1.0+lstep, lstep)
    x, y = np.meshgrid(lons, lats)
    pstep = 0.2
    sststep = 0.02
    
    if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
        fieldmin=-5
        fieldmax=5
        fieldstep = 0.05
        cbstep = 1.0
    elif fsave == 'cldfrac3D':
        fieldmin = -2.0
        fieldmax = 2.0
        fieldstep = 0.05
        cbstep = 0.5
    else:
        fieldmin=-10
        fieldmax=10
        fieldstep =0.2
        cbstep=2.5
    
    sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
    fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
    
    
    plt.figure(figsize=(16,12))
    ax = plt.axes(projection=prj)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=99, edgecolor='k', facecolor='grey')
    ax.set_xticks(mer, crs=prj)
    ax.set_yticks(par, crs=prj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    #ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1)
    #ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
    plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'$^{\circ}$C')
    #ax.quiver(x, y, ucorrs, vcorrs, transform=prj)
    #ax.quiver(x[::12,::12], y[::12,::12], ucorrs[::12,::12], vcorrs[::12,::12], transform=prj, width=0.001, headwidth=16, headlength=10, minshaft=4)
    plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_sst_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    #Plot zonal averages of 3D fieldcorrs, fieldcorrs_lt, and fieldcorrs_st. plot as contourf (p, latitude)
    
    sstcorrs_zonalave = np.ma.average(sstcorrs[:,NAminloni:NAmaxloni], axis=-1)
    
    fieldcorrs_zonalave = np.ma.average(fieldcorrs[:,:,NAminloni:NAmaxloni], axis=-1)
    fieldcorrs_lt_zonalave = np.ma.average(fieldcorrs_lt[:,:,NAminloni:NAmaxloni], axis=-1)
    fieldcorrs_st_zonalave = np.ma.average(fieldcorrs_st[:,:,NAminloni:NAmaxloni], axis=-1)
    
    x, y = np.meshgrid(lats, p)

    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    ax.plot(lats, sstcorrs_zonalave)
    ax.axhline(0, color='k')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_ylabel(r'$^{\circ}$C')
    ax.set_xlabel(r'latitude ($^{\circ}$)')
    ax.set_xlim(0, 60)
    #ax.set_ylim(50,1000)
    #ax.invert_yaxis()
    #cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    #cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    #cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_sst_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    plot = ax.contourf(x, y, fieldcorrs_zonalave, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_ylabel('pressure (hPa)')
    ax.set_xlabel(r'latitude ($^{\circ}$)')
    ax.set_xlim(0, 60)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of {:s} on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(ftitle, latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_{:s}_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()

    
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    plot = ax.contourf(x, y, fieldcorrs_lt_zonalave, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_ylabel('pressure (hPa)')
    ax.set_xlabel(r'latitude ($^{\circ}$)')
    ax.set_xlim(0, 60)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of long-term {:s} on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(ftitle, latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_{:s}_ltcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    plot = ax.contourf(x, y, fieldcorrs_st_zonalave, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_ylabel('pressure (hPa)')
    ax.set_xlabel(r'latitude ($^{\circ}$)')
    ax.set_xlim(0, 60)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of short-term {:s} on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(ftitle, latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_{:s}_stcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()














































