#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import cdutil
import genutil
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
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, regressout_x, butter_lowpass_filter, corr2_coeff
import pandas as pd

fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/NA index/'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')


#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201612.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#dataname = 'ERAi'
dataname = 'MERRA2'

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'axes.titlesize': 22})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 18})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
matplotlib.rcParams.update({'axes.labelsize': 16})
matplotlib.rcParams.update({'ytick.labelsize': 16})
matplotlib.rcParams.update({'xtick.labelsize': 16})
minlat = -90
maxlat = 90

maxlon = 360
minlon = 0

tskip = 12

ps = fSLP('SLP')
ps = ps/1e2
ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
nt_ps = ps.shape[0]
ps = ps[tskip:,:]

#sst = fsst('skt')
sst = fsst('TSKINWTR')
lats = sst.getLatitude()[:]
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:nt_ps,:]

#cE = fcE('CDH')
#cD = fcD('CN')

lhf = fsst('EFLUXWTR')
shf = fsst('HFLUXWTR')
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

#lhf = fthf('slhf')
#lhf = lhf/(12*3600)
#shf = fthf('sshf')
#sshf is accumulated 
#shf = shf/(12*3600)
thf = lhf + shf
#thf is positive down in ERAi, convert to positive up
#thf = thf

#ps = fSLP('SLP')
#ps = ps/1e2
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#ps = ps[tskip:,:]/1e2

#u = fuv('U10M')
#u = fuv('u10')
#u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#u = u[tskip:,:]

v = fuv('V10M')
#v = fuv('v10')
#v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#v = v[tskip:,:]

#umag = np.sqrt(np.square(v) + np.square(u))

#qv10m = fRH('QV10M')
LW_net_surf = radfile['LWGNT']
#LW_net_surf_cs = radfile('LWGNTCLR')
SW_net_surf = radfile['SWGNT']
#SW_net_surf_cs = radfile('SWGNTCLR')
#LW_net_TOA = radfile['LWTUP']
#lwnettoa_cs = radfile('LWTUPCLR')
#swnettoa = radfile('SWTNT')
#swnettoa_cs = radfile('SWTNTCLR')

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

Q_net_surf = LW_net_surf + SW_net_surf

#CRE_surf = Q_net_surf - Q_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH AMO
#field = umag
#ftitle = r'$|\mathbf{u}_{10m}|$'
#fsave = 'umag'
#units = 'm/s'

#field = cD*1e3
#ftitle = r'$c_D$'
#fsave = 'cD'
#units = r'10$^{-3}$'

#field = cE*1e3
#ftitle = r'$c_{E,heat}$'
#fsave = 'cEheat'
#units = r'10$^{-3}$ kg m$^{-2}$ s$^{-1}$'

#field=qv10m
#ftitle=r'RH$_{10m}$'
#fsave = 'RH10m'
#units = '%'

#field = Q_net_surf
#ftitle = r'$Q_{net}$'
#fsave = 'Qnetsurf'
#units = r'W m$^{-2}$'

#field = LW_net_surf
#ftitle = r'$LW_{net}$'
#fsave = 'LWnetsurf'
#units = r'W m$^{-2}$'
#
#field = SW_net_surf
#ftitle = r'$SW_{net}$'
#fsave = 'SWnetsurf'
#units = r'W m$^{-2}$'

#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

field = thf
ftitle = r'THF'
fsave = 'thf'
units = r'W m$^{-2}$'

#field = lhf
#ftitle = r'LHF'
#fsave = 'lhf'
#units = r'W m$^{-2}$'

#field = shf
#ftitle = r'SHF'
#fsave = 'shf'
#units = r'W m$^{-2}$'


#cf = cffile['MDSCLDFRCTTL']
#cf = cf[tskip:,:]

#field = cf*100.
#ftitle = r'$f_{total}$'
#fsave = 'ftotal'
#units = '%'

#field = sst
#ftitle = r'SST'
#fsave = 'sst'
#units = 'K'

#field = sst
#ftitle = r'$\frac{\partial T_s}{\partial t}$'
#fsave = 'dsstdt'
#units = 'K/s'

#field = ps
#ftitle = 'SLP'
#fsave = 'SLP'
#units = 'hPa'

#field = u
#ftitle = 'U-Wind'
#fsave = 'u10m'
#units = r'm/s'

#field = v
#ftitle = 'V-Wind'
#fsave = 'v10m'
#units = r'm/s'


if not(ftitle == 'SST'):
    field = field.subRegion(latitude=(minlat, maxlat), longitude=(minlon,maxlon))
    field = field[tskip:nt_ps,:]

#field = field[:ps.shape[0],...]/qsat

sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
#field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)

#True for detrending data, False for raw data
detr=True
corr=True
rENSO=True
lterm=True

lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
 

#EDIT THIS FOR BOUNDS
lonbounds = [280.,360.]
latbounds = [0,70.]

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
 #ps_an, params = detrend_separate(ps_an)
 field = detrend(field)

#lats = sst.getLatitude()[:]
#lons = sst.getLongitude()[:]
#nt = sst.shape[0]
#lons[0] = 0
#nlat = len(lats)
#nlon = len(lons)


t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1980 + t/12.

tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(field)

field = cdutil.ANNUALCYCLE.departures(field)
sst = cdutil.ANNUALCYCLE.departures(sst)

CTIminlati = np.argmin(np.abs(lats - (-6)))
CTImaxlati = np.argmin(np.abs(lats - 6))
CTIminloni = np.argmin(np.abs(lons - 0))
CTImaxloni = np.argmin(np.abs(lons - 90))

CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])

#coarse grid lat/lon spacing
#cstep=1
#lats = np.arange(minlat,maxlat+cstep,cstep)
#lons = np.arange(0,360+cstep,cstep)
#
#
#cgrid = cdms2.createGenericGrid(lats,lons)
##regridfunc = Regridder(ingrid, cgrid)
#sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

sst = sst.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
field = field.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))

#NAminlati = 0
#NAmaxlati = -1
#NAminloni = 0
#NAmaxloni = -1

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]
nlat = len(lats)
nlon = len(lons)

#NAminlati = np.argmin(np.abs(lats - latbounds[0]))
#NAmaxlati = np.argmin(np.abs(lats - latbounds[1]))
#NAminloni = np.argmin(np.abs(lons - lonbounds[0]))
#NAmaxloni = np.argmin(np.abs(lons - lonbounds[1]))

#interpolate to coarser grid to speed up 
#sst= sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
 
#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5*12 + 1
ci = (N_map-1)/2
ltlag = 5
stlag = 1

#lagmax=3
lagmax = 12*10
lagstep = 12
lags = np.arange(-lagmax,lagmax+2*lagstep, lagstep)

if rENSO:
    CTIlag = 2
    sst = regressout_x(sst[CTIlag:], CTI[:-CTIlag])
    CTI = CTI[CTIlag:]
    field = field[CTIlag:]
    #field = regressout_x(field, CTI)
    
nt = field.shape[0]


#lags = np.arange(-3,0,6,3)

#sst_globe = spatial_ave(sst, lats)
#field_globe = spatial_ave(field, lats)

#subtract global annual mean to isolate local processes
#sstprime = sst.T - sst_globe
#sstprime = sstprime.T
sstprime = sst
#
#fieldprime = field.T - field_globe
#fieldprime = fieldprime.T
fieldprime = field

# Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12*7.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)


#need to fill missing field data by interpolation... THF blows up otherwise 
#sst_df = pd.DataFrame(sst.reshape(nt, nlat*nlon))
field_df = pd.DataFrame(field.reshape(nt, nlat*nlon))
#sst_df = sst_df.interpolate()
field_df = field_df.interpolate()
#sstprime = sst_df.values.reshape(nt, nlat, nlon)
fieldprime = field_df.values.reshape(nt, nlat, nlon)


# apply the filter
field_lt = butter_lowpass_filter(fieldprime, cutoff, fs, order)
sst_lt = butter_lowpass_filter(sstprime, cutoff, fs, order)

#sstprime = np.ma.masked_array(sstprime, mask=~np.isfinite(sstprime))
fieldprime = np.ma.masked_array(fieldprime, mask=~np.isfinite(fieldprime))
field_lt = np.ma.masked_array(field_lt, mask=~np.isfinite(field_lt))
sst_lt = np.ma.masked_array(sst_lt, mask=np.abs(sst_lt) > 1e4)
 
field_st =  fieldprime - field_lt
sst_st = sstprime - sst_lt

nt = sst.shape[0]
nt_lt = sst_lt.shape[0]
#nt_lt = field_lt.shape[0]

#lats = lats[NAminlati:NAmaxlati]
#lons = lons[NAminloni:NAmaxloni]
#
#nlat = len(lats)
#nlon = len(lons)



#EDIT BOUNDS FOR AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
#for latbounds in latboundar:
    
fieldlagcorrs = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_st = np.zeros((len(lags), nlat, nlon))

#sstcorrs = MV.zeros((nlatc,nlonc))
#sstpvals = MV.zeros((nlat,nlon))
fieldcorrs = MV.zeros((nlat, nlon))
fieldcorrs_lt = MV.zeros((nlat,nlon))
fieldcorrs_st = MV.zeros((nlat,nlon))

sstcorrs = MV.zeros((nlat, nlon))
sstcorrs_lt = MV.zeros((nlat, nlon))
sstcorrs_st = MV.zeros((nlat, nlon))


       #compute correlation between long-term/short-term index and 2D field
print r'calculating correlations between index and {:s}...'.format(ftitle)
for i in range(nlat):   
                print 'latitude', lats[i]
             
             #for j in range(nlon):
                 
                sstprime_g = sstprime[:,i,:]
                fieldprime_g = fieldprime[:,i,:]
          
                
                field_lt_g = field_lt[:,i,:]
                field_st_g = field_st[:,i,:]
                sst_lt_g = sst_lt[:,i,:]
                sst_st_g = sst_st[:,i,:]
                
        
                N = sstprime_g.shape[1]
            
    #            clf = linear_model.LinearRegression()
    #            clf.fit(indexstd.reshape(-1,1), sstprime_g)
    #            sstcorrs[i,:] = np.squeeze(clf.coef_)
                
    #            clf = linear_model.LinearRegression()
    #            clf.fit(indexstd.reshape(-1,1), fieldprime_g)
    #            fieldcorrs[i,:] = np.squeeze(clf.coef_)
    
                scaler = StandardScaler()
                sststd = scaler.fit_transform(sstprime_g)
                if lterm:
                    sststd_lt = scaler.fit_transform(sst_lt_g)
                    sststd_st = scaler.fit_transform(sst_st_g)
                
                sstcoefs = np.diag(np.ma.cov(sststd, sstprime_g, rowvar=False)[:N,N:])
                fieldcoefs = np.diag(np.ma.cov(sststd, fieldprime_g, rowvar=False)[:N,N:])
                
                sstcorrs[i,:] = sstcoefs
                fieldcorrs[i,:] = fieldcoefs
                
    
                if lterm:
    
    #                clf = linear_model.LinearRegression()
    #                clf.fit(indexstd_lt.reshape(-1,1), sst_lt_g)
    #                sstcorrs_lt[i,:] = np.squeeze(clf.coef_)
    #             
    #                clf = linear_model.LinearRegression()
    #                clf.fit(indexstd_st.reshape(-1,1), sst_st_g)
    #                sstcorrs_st[i,:] = np.squeeze(clf.coef_)
                    
                    sstcoefs_lt = np.diag(np.ma.cov(sststd_lt, sst_lt_g, rowvar=False)[:N,N:])
                    sstcoefs_st = np.diag(np.ma.cov(sststd_st, sst_st_g, rowvar=False)[:N,N:])
                    
                    fieldcoefs_lt = np.diag(np.ma.cov(sststd_lt, field_lt_g, rowvar=False)[:N,N:])
                    fieldcoefs_st = np.diag(np.ma.cov(sststd_st, field_st_g, rowvar=False)[:N,N:])
                
                    sstcorrs_lt[i,:] = sstcoefs_lt
                    sstcorrs_st[i,:] = sstcoefs_st
                    
                    fieldcorrs_lt[i,:] = fieldcoefs_lt
                    fieldcorrs_st[i,:] = fieldcoefs_st
                    
                    
                for j, lag in enumerate(lags):
             
                    #scaler = StandardScaler()
                 
                         #nt = fieldprime_g.shape[0]
                         #nt_lt = field_lt_g.shape[0]
                         #fieldstd = scaler.fit_transform(fieldprime_g)
                         #fieldstd_lt = scaler.fit_transform(field_lt_g)
                         #fieldstd_st = scaler.fit_transform(field_st_g)
    #                     fieldstd = (fieldprime_g - np.ma.mean(fieldprime_g, axis=0, keepdims=True))/(nt*np.ma.std(fieldprime_g, axis=0, keepdims=True))
    #                     fieldstd_lt = (field_lt_g - np.ma.mean(field_lt_g, axis=0, keepdims=True))/(nt_lt*np.ma.std(field_lt_g, axis=0, keepdims=True))
    #                     fieldstd_st = (field_st_g - np.ma.mean(field_st_g, axis=0, keepdims=True))/(nt_lt*np.ma.std(field_st_g, axis=0, keepdims=True))
                     #else:
                    fieldstd = fieldprime_g
                    fieldstd_lt = field_lt_g
                    fieldstd_st = field_st_g 
                     
    #                fieldclf = linear_model.LinearRegression()
    #                fieldclf_lt = linear_model.LinearRegression()
    #                fieldclf_st = linear_model.LinearRegression()
                     
                    if corr:
                        
                                    
                         #THF LAGS SST
                        if lag > 0:
                            #fieldlagcoefs = np.diag(np.ma.corrcoef(sststd[:-lag,...], fieldprime_g[lag:,...], rowvar=False)[:N,N:])
                            fieldlagcoefs = np.diag(corr2_coeff(sststd[:-lag,...].T, fieldprime_g[lag:,...].T))
                            if lterm:
                                 fieldlagcoefs_lt = np.diag(corr2_coeff(sststd_lt[:-lag,...].T, field_lt_g[lag:,...].T))
                                 fieldlagcoefs_st = np.diag(corr2_coeff(sststd_st[:-lag,...].T, field_st_g[lag:,...].T))
    #                            fieldlagcoefs_lt = np.diag(np.ma.corrcoef(sststd_lt[:-lag,...], field_lt_g[lag:,...], rowvar=False)[:N,N:])
    #                            fieldlagcoefs_st = np.diag(np.ma.corrcoef(sststd_st[:-lag,...], field_st_g[lag:,...], rowvar=False)[:N,N:])
        #                    fieldclf.fit(MMstd[:-lag], fieldstd[lag:,:])
        #                    fieldclf_lt.fit(MMstd_lt[:-lag], fieldstd_lt[lag:,:])
        #                    fieldclf_st.fit(MMstd_st[:-lag], fieldstd_st[lag:,:])
                    
                        #THF LEADS SST
                        elif lag < 0: 
                            #fieldlagcoefs = np.diag(np.ma.corrcoef(sststd[-lag:,...], fieldprime_g[:lag,...], rowvar=False)[:N,N:])
                            fieldlagcoefs = np.diag(corr2_coeff(sststd[-lag:,...].T, fieldprime_g[:lag,...].T))
                            if lterm:
                                fieldlagcoefs_lt = np.diag(corr2_coeff(sststd_lt[-lag:,...].T, field_lt_g[:lag,...].T))
                                fieldlagcoefs_st = np.diag(corr2_coeff(sststd_st[-lag:,...].T, field_st_g[:lag,...].T))
                                #fieldlagcoefs_lt = np.diag(np.ma.corrcoef(sststd_lt[-lag:,...], field_lt_g[:lag,...], rowvar=False)[:N,N:])
                                #fieldlagcoefs_st = np.diag(np.ma.corrcoef(sststd_st[-lag:,...], field_st_g[:lag,...], rowvar=False)[:N,N:])
        #                    fieldclf.fit(MMstd[-lag:], fieldstd[:lag,:])
        #                    fieldclf_lt.fit(MMstd_lt[-lag:], fieldstd_lt[:lag,:])
        #                    fieldclf_st.fit(MMstd_st[-lag:], fieldstd_st[:lag,:])
                    
                        else:
                            #fieldlagcoefs = np.diag(np.ma.corrcoef(sststd, fieldprime_g, rowvar=False)[:N,N:])
                            fieldlagcoefs = np.diag(corr2_coeff(sststd.T, fieldprime_g.T))
                            if lterm:
                                fieldlagcoefs_lt = np.diag(corr2_coeff(sststd_lt.T, field_lt_g.T))
                                fieldlagcoefs_st = np.diag(corr2_coeff(sststd_st.T, field_st_g.T))
                                #fieldlagcoefs_lt = np.diag(np.ma.corrcoef(sststd_lt, field_lt_g, rowvar=False)[:N,N:])
                                #fieldlagcoefs_st = np.diag(np.ma.corrcoef(sststd_st, field_st_g, rowvar=False)[:N,N:])
        #                    fieldclf.fit(MMstd, fieldstd)
        #                    fieldclf_lt.fit(MMstd_lt, fieldstd_lt)
        #                    fieldclf_st.fit(MMstd_st, fieldstd_st)
                        
                        
                    else:
                
                         #THF LAGS SST
                        if lag > 0:
                            fieldlagcoefs = np.diag(np.ma.cov(sststd[:-lag,...], fieldprime_g[lag:,...], rowvar=False)[:N,N:])
                            if lterm:
                                fieldlagcoefs_lt = np.diag(np.ma.cov(sststd_lt[:-lag,...], field_lt_g[lag:,...], rowvar=False)[:N,N:])
                                fieldlagcoefs_st = np.diag(np.ma.cov(sststd_st[:-lag,...], field_st_g[lag:,...], rowvar=False)[:N,N:])
        #                    fieldclf.fit(MMstd[:-lag], fieldstd[lag:,:])
        #                    fieldclf_lt.fit(MMstd_lt[:-lag], fieldstd_lt[lag:,:])
        #                    fieldclf_st.fit(MMstd_st[:-lag], fieldstd_st[lag:,:])
                    
                        #THF LEADS SST
                        elif lag < 0: 
                            fieldlagcoefs = np.diag(np.ma.cov(sststd[-lag:,...], fieldprime_g[:lag,...], rowvar=False)[:N,N:])
                            if lterm:
                                fieldlagcoefs_lt = np.diag(np.ma.cov(sststd_lt[-lag:,...], field_lt_g[:lag,...], rowvar=False)[:N,N:])
                                fieldlagcoefs_st = np.diag(np.ma.cov(sststd_st[-lag:,...], field_st_g[:lag,...], rowvar=False)[:N,N:])
        #                    fieldclf.fit(MMstd[-lag:], fieldstd[:lag,:])
        #                    fieldclf_lt.fit(MMstd_lt[-lag:], fieldstd_lt[:lag,:])
        #                    fieldclf_st.fit(MMstd_st[-lag:], fieldstd_st[:lag,:])
                    
                        else:
                            fieldlagcoefs = np.diag(np.ma.cov(sststd, fieldprime_g, rowvar=False)[:N,N:])
                            if lterm:
                                fieldlagcoefs_lt = np.diag(np.ma.cov(sststd_lt, field_lt_g, rowvar=False)[:N,N:])
                                fieldlagcoefs_st = np.diag(np.ma.cov(sststd_st, field_st_g, rowvar=False)[:N,N:])
        #                    fieldclf.fit(MMstd, fieldstd)
        #                    fieldclf_lt.fit(MMstd_lt, fieldstd_lt)
        #                    fieldclf_st.fit(MMstd_st, fieldstd_st)
            
                    
                    
                    fieldlagcorrs[j,i,:] = fieldlagcoefs
                    fieldlagcorrs_lt[j,i,:] = fieldlagcoefs_lt
                    fieldlagcorrs_st[j,i,:] = fieldlagcoefs_st
#    
#
        

#ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
#ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
#ticklbls[ticklbls == -0.00] = 0.00
      
#Plot maps of SST and THF patterns associated with AMO
#CHANGE THIS FOR MAP PROJECTION
cent=-(lonbounds[1]-lonbounds[0])/2.
prj = cart.crs.PlateCarree(central_longitude=cent)
#prj = cart.crs.Mollweide(central_longitude=cent)

bnds = [np.round(lonbounds[0]-359), np.round(lonbounds[1]-361), latbounds[0], latbounds[1]]

pardiff = 30.
merdiff = 60.
if lonbounds[1] - lonbounds[0] <= 180:
    pardiff = 15.
    merdiff = 15.
par = np.arange(-90.,91.,pardiff)
mer = np.arange(-180.,180.,merdiff)
lstep = 0.01
levels = np.arange(-1.0, 1.0+lstep, lstep)
x, y = np.meshgrid(lons, lats)
pstep = 0.2
sststep = 0.02

orient = 'horizontal'
if lonbounds[1] - lonbounds[0] <= 180:
    orient = 'vertical'

if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst':
    fieldmin=-0.6
    fieldmax=0.6
    fieldstep = 0.02
    cbstep = 0.2
elif fsave == 'umag':
    fieldmin=-0.5
    fieldmax=0.5
    fieldstep=0.01
    cbstep = 0.1
elif fsave == 'RH10m':
    fieldmin = -2
    fieldmax = 2
    fieldstep = 0.01
    cbstep = 0.5
elif fsave == 'qvdiff':
    fieldmin = -2
    fieldmax = 2
    fieldstep = 0.01
    cbstep = 0.5
elif fsave == 'cE':
    fieldmin = -0.5
    fieldmax = 0.5
    fieldstep = 0.01
    cbstep = 0.1
elif fsave == 'cD':
    fieldmin = -0.5
    fieldmax = 0.5
    fieldstep = 0.01
    cbstep = 0.1
else:
    fieldmin=-10
    fieldmax=10
    fieldstep =0.2
    cbstep=2.5

if fsave == 'sst':
    #cmap = plt.cm.cubehelix_r
    cmap = plt.cm.RdBu_r
else:
    cmap = plt.cm.RdBu_r
    
ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
ticklbls[ticklbls == -0.0] = 0.0
                 
sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

if not(corr):

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
    #ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
    #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, fieldcorrs, cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
    cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    if fsave == 'sst':
        plt.title(r'SST autoregression')
    else:
        plt.title(r'{:s}'.format(ftitle))
    plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    if lterm:
    
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
        ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
        #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
        plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        if fsave == 'sst':
            plt.title(r'Long-term SST autoregression')
        else:
            plt.title(r'Long-term {:s} ({:1.0f}-month RM)'.format(ftitle, N_map-1))
        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
        plt.close()
        
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
        ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
        #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
        plot = ax.contourf(x, y, fieldcorrs_st, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        if fsave == 'sst':
            plt.title(r'Short-term SST autoregression')
        else:
            plt.title(r'Short-term {:s} (residual)'.format(ftitle))
        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
        plt.close()
    
    
    #UNCOMMENT TO PLOT LEAD-LAG REGRESSION MAPS
    
    
    for leadi in range(len(lags)):
        #lagi=-(leadi+1)
    
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
        ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
        #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
        plot = ax.contourf(x, y, fieldlagcorrs[leadi,...], cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        if fsave == 'sst':
            plt.title(r'SST autoregression ({:1.0f} month lag)'.format(lags[leadi]))
        else:
            plt.title(r'{:s} ({:1.0f} month lag)'.format(ftitle, lags[leadi]))
        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lag{:1.0f}corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lags[leadi], latbounds[0], latbounds[1], str(detr)[0]))
        plt.close()
        
        
        if lterm:

            
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
            ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
            #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
            plot = ax.contourf(x, y, fieldlagcorrs_lt[leadi,...], cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
            cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklbls)
            if fsave == 'sst':
                plt.title(r'Long-term SST autoregression {:s} ({:1.0f} month lag)'.format( lags[leadi]))
            else:
                plt.title(r'Long-term {:s} ({:1.0f} month lag)'.format(ftitle, lags[leadi]))
            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lag{:1.0f}ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lags[leadi], latbounds[0], latbounds[1], str(detr)[0]))
            plt.close()
            
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
            ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
            #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
            plot = ax.contourf(x, y, fieldlagcorrs_st[leadi,...], cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
            cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklbls)
            if fsave == 'sst':
                plt.title(r'Short-term SST autoregression ({:1.0f} month lag)'.format(lags[leadi]))
            else:
                plt.title(r'Short-term {:s} ({:1.0f} month lag)'.format(ftitle, lags[leadi]))
            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lag{:1.0f}stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lags[leadi], latbounds[0], latbounds[1], str(detr)[0]))
            plt.close()


fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs, axis=2)
if lterm:
    fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_lt, axis=2)
    fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_st, axis=2)

#fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#if lterm:
#    fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#    fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)

 
if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst':
    fieldmin=-0.6
    fieldmax=0.6
    fieldstep = 0.02
    cbstep = 0.2
elif fsave == 'umag':
    fieldmin=-0.3
    fieldmax=0.3
    fieldstep=0.01
    cbstep = 0.1
elif fsave == 'RH10m':
    fieldmin = -2
    fieldmax = 2
    fieldstep = 0.01
    cbstep = 0.5
elif fsave == 'cE':
    fieldmin = -0.5
    fieldmax = 0.5
    fieldstep = 0.01
    cbstep = 0.1
elif fsave == 'cD':
    fieldmin = -0.5
    fieldmax = 0.5
    fieldstep = 0.01
    cbstep = 0.1
else:
    fieldmin=-4
    fieldmax=4
    fieldstep =0.1
    cbstep=1.0
    
if corr:
    fieldminlag = -0.4
    fieldmaxlag = 0.4
    cbsteplag = 0.2
    fieldunitslag = 'Correlation'

else:
    fieldminlag = fieldmin
    fieldmaxlag = fieldmax
    cbsteplag = cbstep 
    fieldunitslag = units
    
#if fsave == 'sst':
#    if not(lterm):
#        cmap = plt.cm.cubehelix_r
#    else:
#        cmap = plt.cm.PuOr_r
#else:
if fsave == 'sst':
    #cmap = plt.cm.cubehelix_r
    cmap = plt.cm.PRGn
else:
    cmap = plt.cm.PRGn
    
ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
ticklbls[ticklbls == -0.00] = 0.00
                 
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

weights = np.cos(np.deg2rad(lats))

lagoffset = np.diff(lags)[0]/2.
latoffset = np.diff(lats)[0]/2.

laglabels = np.round(np.arange(-lagmax,lagmax+2*lagstep,lagmax/2))

#lagg, latt = np.meshgrid(lags-lagoffset, NAlats-latoffset)
lagg, latt = np.meshgrid(lags-lagoffset, lats-latoffset)

#Plot zonally-averaged lagged correlation between SST and THF'

if lagmax > 12:
    lagg = lagg/12.
    laglabels = laglabels/12.
    lagunits = 'years'
else:
    lagunits = 'months'

if lterm:
    fig=plt.figure(figsize=(22,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(131)
    h = ax.pcolor(lagg, latt, fieldlagcorrs_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    ax.set_title('{:s}'.format(ftitle))
    ax.set_xlabel(r'{:s} Lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylabel('Latitude (degrees)')
    ax.set_ylim(lats[0],lats[-1])
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    
    ax = fig.add_subplot(132)
    ax.pcolor(lagg, latt, fieldlagcorrs_lt_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    ax.set_title('Long-term {:s} ({:1.0f}-yr LP)'.format(ftitle, Tn/12.))
    ax.set_xlabel(r'{:s} Lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylim(lats[0],lats[-1])
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    ax = fig.add_subplot(133)
    h = ax.pcolor(lagg, latt, fieldlagcorrs_st_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    ax.set_title('Short-term {:s}'.format(ftitle))
    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylim(lats[0],lats[-1])
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    
    cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    if corr:
        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorrlagmax{:2.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregrlagmax{:2.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close() 
else:
    #Plot zonally-averaged lagged correlation between long-term AMO and THF
    fig=plt.figure(figsize=(10,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(111)
    h = ax.pcolor(lagg, latt, fieldlagcorrs_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('{:s}'.format(ftitle))
    else:
        ax.set_title('{:s}'.format(ftitle))
    ax.set_xlabel(r'{:s} Lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylabel('Latitude (degrees)')
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
    cb.set_ticks(ticks)
    ax.axvline(0, color='k')
    if corr:
            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_lagmax{:3.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    else:
            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_lagmax{:3.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close() 
    
    
#latw = 15
#
#slats = np.array([5, 45])
#slati = slats[0]    
#nlati = slati+latw
#
#si = np.argmin(np.abs(lats - slati))
#ni = np.argmin(np.abs(lats - nlati))
#    
#NAfieldlagcorr_ave = spatial_ave(fieldlagcorrs[:,si:ni,:], lats[si:ni])
#NAfieldlagcorr_lt_ave = spatial_ave(fieldlagcorrs_lt[:,si:ni,:], lats[si:ni])
#NAfieldlagcorr_st_ave = spatial_ave(fieldlagcorrs_st[:,si:ni,:], lats[si:ni])
#
#        
#if lterm:
#
#    #Plot NA averaged lagged correlation between long-term AMO and THF
#    fig=plt.figure(figsize=(22,6))
#    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
#    ax = fig.add_subplot(131)
#    plt.plot(lagg[0,:], NAfieldlagcorr_ave)
#    ax.axvline(0, color='k')
#    ax.set_title('{:s}'.format(ftitle))
#    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
#    if corr:
#        ax.set_ylabel('Correlation')
#    else:
#        ax.set_ylabel('{:s} ({:s})'.format(ftitle, units))
#    #if corr:
#    #    ax.set_ylim((-0.5,0.5))
#    ax.set_ylim((fieldminlag, fieldmaxlag))
#    #ax.set_ylim(latbounds[0], latbounds[1])
#    ax.axhline(0, color='k', linewidth=1)
#    ax.set_xticks(laglabels)
#    ax.set_xticklabels(laglabels)
#    ax = fig.add_subplot(132)
#    plt.plot(lagg[0,:], NAfieldlagcorr_lt_ave)
#    ax.axvline(0, color='k')
#    ax.set_title('Long-term {:s}'.format(ftitle))
#    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
#    if corr:
#        ax.set_ylabel('Correlation')
#    else:
#        ax.set_ylabel('{:s} ({:s})'.format(ftitle, units))
#    ax.set_ylim((fieldminlag, fieldmaxlag))
#    ax.set_xticks(laglabels)
#    ax.axhline(0, color='k', linewidth=1)
#    ax.set_xticklabels(laglabels)
#    ax = fig.add_subplot(133)
#    plt.plot(lagg[0,:], NAfieldlagcorr_st_ave)
#    ax.axvline(0, color='k')
#    ax.set_title('Short-term {:s}'.format(ftitle))
#    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
#    if corr:
#        ax.set_ylabel('Correlation')
#    else:
#        ax.set_ylabel('{:s} ({:s})'.format(ftitle, units))
#    ax.set_ylim((fieldminlag, fieldmaxlag))
#    ax.set_xticks(laglabels)
#    ax.axhline(0, color='k', linewidth=1)
#    ax.set_xticklabels(laglabels)
#    if corr:
#            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_lagmax{:3.0f}_{:1.0f}to{:1.0f}ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lats[si], lats[ni], lagmax, latbounds[0], latbounds[1], str(detr)[0]))
#    else:
#            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_lagmax{:3.0f}_{:1.0f}to{:1.0f}ave__{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lats[si], lats[ni], lagmax, latbounds[0], latbounds[1], str(detr)[0]))
#    plt.close() 
#    
#else:
#    
#    #Plot zonally-averaged lagged correlation between long-term AMO and THF
#    fig=plt.figure(figsize=(10,6))
#    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
#    ax = fig.add_subplot(111)
#    plt.plot(lagg[0,:], NAfieldlagcorr_ave)
#    ax.axvline(0, color='k')
#    ax.set_title('{:s}'.format(ftitle))
#    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
#    if corr:
#        ax.set_ylabel('Correlation')
#    else:
#        ax.set_ylabel('{:s} ({:s})'.format(ftitle, units))
#    ax.set_ylim((fieldminlag, fieldmaxlag))
#    #ax.set_ylim(latbounds[0], latbounds[1])
#    ax.axhline(0, color='k', linewidth=1)
#    ax.set_xticks(laglabels)
#    ax.set_xticklabels(laglabels)
#    ax.axvline(0, color='k')
#    ax.set_title('{:s}'.format(ftitle))
#    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
#    if corr:
#            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_lagmax{:3.0f}_NAave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
#    else:
#            plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_lagmax{:3.0f}_NAave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
#    plt.close() 
















































