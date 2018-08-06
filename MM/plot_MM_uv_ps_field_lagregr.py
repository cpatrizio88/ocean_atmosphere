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
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, regressout_x

fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/MM/'

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

MMf = np.loadtxt(fin + 'AMM.txt', skiprows=1)

MM_dv = MMf[:,2]
MMyr = MMf[:,0]
MMmo = MMf[:,1]

MMwind = MMf[:,3]


MMt = MMyr + MMmo/12.


#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201612.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#dataname = 'ERAi'
dataname = 'MERRA2'

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'axes.titlesize': 22})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 18})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
matplotlib.rcParams.update({'axes.labelsize': 16})
matplotlib.rcParams.update({'ytick.labelsize': 16})
matplotlib.rcParams.update({'xtick.labelsize': 16})
maxlat = 90
minlat = -90

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
#lats = sst.getLatitude()[:]
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

u = fuv('U10M')
u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
u = u[tskip:nt_ps,:]
#
v = fuv('V10M')
v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
v = v[tskip:nt_ps,:]

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

##### EDIT FIELD FOR CORRELATION WITH MM
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

field = Q_net_surf
ftitle = r'$Q_{net}$'
fsave = 'Qnetsurf'
units = r'W m$^{-2}$'

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

#field = thf
#ftitle = r'THF'
#fsave = 'thf'
#units = r'W m$^{-2}$'

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


#NAmaxlati = np.where(lats > maxlat)[0][0]
#NAminlati = np.where(lats > minlat)[0][0]

#sst = sst.subRegion(longitude=(minlon, maxlon))
#sst = sst[tskip:,NAminlati:NAmaxlati,:]
#field = field[tskip:,NAminlati:NAmaxlati,:]

if not(ftitle == 'SST'):
    field = field.subRegion(latitude=(minlat, maxlat), longitude=(minlon,maxlon))
    field = field[tskip:nt_ps,:]

#field = field[:ps.shape[0],...]/qsat

sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
##field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)
ps_mask = np.ma.getmaskarray(ps)
ps_mask = np.ma.mask_or(sst_mask[:ps.shape[0],:,:], ps_mask)
ps = np.ma.array(ps, mask=ps_mask)

#True for detrending data, False for raw data
detr=True
corr=False

lterm=False

#EDIT THIS FOR BOUNDS
lonbounds = [290.,360.]
latbounds = [-30,70.]

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
 u = detrend(u)
 v = detrend(v)
 field = detrend(field)
 
 
lats = field.getLatitude()[:]
lons = field.getLongitude()[:]


#lats = sst.getLatitude()[:]
#lons = sst.getLongitude()[:]
#nt = sst.shape[0]
#lons[0] = 0
#nlat = len(lats)
#nlon = len(lons)


t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
tyears = 1980 + t/12.

MMstarti = np.where(MMt == tyears[0])[0][0]
MMendi = np.where(MMt == tyears[-1])[0][0]



#tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(field)
cdutil.setTimeBoundsMonthly(u)
cdutil.setTimeBoundsMonthly(v)
cdutil.setTimeBoundsMonthly(ps)

field = cdutil.ANNUALCYCLE.departures(field)
sst = cdutil.ANNUALCYCLE.departures(sst)
u = cdutil.ANNUALCYCLE.departures(u)
v = cdutil.ANNUALCYCLE.departures(v)
ps = cdutil.ANNUALCYCLE.departures(ps)

CTIminlati = np.argmin(np.abs(lats - (-6)))
CTImaxlati = np.argmin(np.abs(lats - 6))
CTIminloni = np.argmin(np.abs(lons - 0))
CTImaxloni = np.argmin(np.abs(lons - 90))
 
CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])


#SUBTRACT COLD TONGUE INDEX FROM SST

#coarse grid lat/lon spacing
cstep=1
lats = np.arange(minlat,maxlat+cstep,cstep)
lons = np.arange(0,360+cstep,cstep)


cgrid = cdms2.createGenericGrid(lats,lons)
#regridfunc = Regridder(ingrid, cgrid)
sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
u = u.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
v = v.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
ps = ps.regrid(cgrid, regridTool="esmf", regridMethod = "linear")



sst = sst.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
field = field.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
u = u.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
v = v.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
ps = ps.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))


#NAminlati = 0
#NAmaxlati = -1
#NAminloni = 0
#NAmaxloni = -1

lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
nlat = len(lats)
nlon = len(lons)
nt = field.shape[0]
#interpolate to coarser grid to speed up 
#sst= sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

#regress out CTI from all fields
sst_MM = regressout_x(sst, CTI)
#field = regressout_x(field, CTI)
#u = regressout_x(u, CTI)
#v = regressout_x(v, CTI)
#ps = regressout_x(ps, CTI)

#MMlatbounds = [-30,30]
#MMlonbounds = [290,360]
#
#MMlatboundmini = np.argmin(np.abs(MMlatbounds[0]-lats))
#eqi = np.argmin(np.abs(0 - lats))
#MMlatboundmaxi = np.argmin(np.abs(MMlatbounds[1]-lats))
#MMlonboundmini = np.argmin(np.abs(MMlonbounds[0]-lons))
#MMlonboundmaxi = np.argmin(np.abs(MMlonbounds[1]-lons))
#
#sst_MMNH = sst_MM[:,eqi:MMlatboundmaxi,MMlonboundmini:MMlonboundmaxi]
#sst_MMSH = sst_MM[:,MMlatboundmini:eqi,MMlonboundmini:MMlonboundmaxi]
#
#MMlatsNH = lats[eqi:MMlatboundmaxi]
#MMlatsSH = lats[MMlatboundmini:eqi]
#
#MM = spatial_ave(sst_MMNH, MMlatsNH) - spatial_ave(sst_MMSH, MMlatsSH)


MM = MM_dv[MMstarti:MMendi+1]
MM_dv = MM_dv[MMstarti:MMendi+1]
MMwind = MMwind[MMstarti:MMendi+1]

 
#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5*12 + 1
ci = (N_map-1)/2
ltlag = 5
stlag = 1

#lagmax=3
lagmax = 12
lagstep = 2
lags = np.arange(-lagmax,lagmax+2*lagstep, lagstep)


#lags = np.arange(-3,0,6,3)

#sst_globe = spatial_ave(sst, lats)
field_globe = spatial_ave(field, lats)

#subtract global annual mean to isolate local processes
#sstprime = sst.T - sst_globe
#sstprime = sstprime.T
sstprime = sst


#fieldprime = field.T - field_globe
#fieldprime = fieldprime.T
fieldprime = field

uprime = u
vprime = v

psprime = ps



field_lt = running_mean(fieldprime, N_map)
field_st =  fieldprime[ci:-ci,:] - field_lt

sst_lt = running_mean(sstprime, N_map)
sst_st = sstprime[ci:-ci,:] - sst_lt

u_lt = running_mean(uprime, N_map)
u_st = uprime[ci:-ci,:] - u_lt

v_lt = running_mean(vprime, N_map)
v_st = vprime[ci:-ci,:] - v_lt

ps_lt = running_mean(psprime, N_map)
ps_st = psprime[ci:-ci,:] - ps_lt


field_lt = np.ma.masked_array(field_lt, mask=np.abs(field_lt>1e4))
field_st = np.ma.masked_array(field_st, mask=np.abs(field_st>1e4))
sst_lt = np.ma.masked_array(sst_lt, mask=np.abs(sst_lt>1e4))
sst_st = np.ma.masked_array(sst_st, mask=np.abs(sst_st>1e4))


MM_lt = running_mean(MM, N_map)
MM_st = MM[ci:-ci] - MM_lt

MMwind_lt = running_mean(MMwind, N_map)
MMwind_st = MMwind[ci:-ci]- MMwind_lt

nt = field.shape[0]
nt_lt = sst_lt.shape[0]


#lats = lats[NAminlati:NAmaxlati]
#lons = lons[NAminloni:NAmaxloni]
#
#nlat = len(lats)
#nlon = len(lons)


scaler = StandardScaler()
MMstd = scaler.fit_transform(MM.reshape(-1,1))
MMstd_lt = scaler.fit_transform(MM_lt.reshape(-1,1))
MMstd_st = scaler.fit_transform(MM_st.reshape(-1,1))


MMwindstd = scaler.fit_transform(MMwind.reshape(-1,1))
MMwindstd_lt = scaler.fit_transform(MMwind_lt.reshape(-1,1))
MMwindstd_st = scaler.fit_transform(MMwind_st.reshape(-1,1))

sstcorrs = MV.zeros((nlat,nlon))
sstcorrs_lt = MV.zeros((nlat, nlon))
sstcorrs_st = MV.zeros((nlat, nlon))
fieldcorrs = MV.zeros((nlat, nlon))
fieldcorrs_lt = MV.zeros((nlat,nlon))
fieldcorrs_st = MV.zeros((nlat,nlon))

fieldlagcorrs = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_st = np.zeros((len(lags), nlat, nlon))

ulagcorrs = np.zeros((len(lags), nlat, nlon))
ulagcorrs_lt = np.zeros((len(lags), nlat, nlon))
ulagcorrs_st = np.zeros((len(lags), nlat, nlon))

vlagcorrs = np.zeros((len(lags), nlat, nlon))
vlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
vlagcorrs_st = np.zeros((len(lags), nlat, nlon))

pslagcorrs = np.zeros((len(lags), nlat, nlon))
pslagcorrs_lt = np.zeros((len(lags), nlat, nlon))
pslagcorrs_st = np.zeros((len(lags), nlat, nlon))



#compute correlation between long-term/short-term MM and 2D field
print r'calculating correlations between MM and {:s}...'.format(ftitle)
for i in range(nlat):   
        print 'latitude', lats[i]
     
     #for j in range(nlon):
         
        sstprime_g = sstprime[:,i,:]
        fieldprime_g = fieldprime[:,i,:]
        uprime_g = uprime[:,i,:]        
        vprime_g = vprime[:,i,:]
        psprime_g = psprime[:,i,:]

        
        field_lt_g = field_lt[:,i,:]
        field_st_g = field_st[:,i,:]
        sst_lt_g = sst_lt[:,i,:]
        sst_st_g = sst_st[:,i,:]
        
        u_lt_g = u_lt[:,i,:]
        u_st_g = u_st[:,i,:]
        
        v_lt_g = v_lt[:,i,:]
        v_st_g = v_st[:,i,:]
        
        ps_lt_g = ps_lt[:,i,:]
        ps_st_g = ps_st[:,i,:]
        
         
        clf = linear_model.LinearRegression()
        clf.fit(MMstd.reshape(-1,1), sstprime_g)
        sstcorrs[i,:] = np.squeeze(clf.coef_)
        
        clf = linear_model.LinearRegression()
        clf.fit(MMstd_lt.reshape(-1,1), sst_lt_g)
        sstcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        clf = linear_model.LinearRegression()
        clf.fit(MMstd_st.reshape(-1,1), sst_st_g)
        sstcorrs_st[i,:] = np.squeeze(clf.coef_)
        
     
        clf = linear_model.LinearRegression()
        clf.fit(MMstd.reshape(-1,1), fieldprime_g)
        fieldcorrs[i,:] = np.squeeze(clf.coef_)
     
        clf = linear_model.LinearRegression()
        clf.fit(MMstd_lt.reshape(-1,1), field_lt_g)
        fieldcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        clf = linear_model.LinearRegression()
        clf.fit(MMstd_st.reshape(-1,1), field_st_g)
        fieldcorrs_st[i,:] = np.squeeze(clf.coef_)
        
        
        for j, lag in enumerate(lags):
         
            scaler = StandardScaler()
            if corr:
                 fieldstd = scaler.fit_transform(fieldprime_g)
                 fieldstd_lt = scaler.fit_transform(field_lt_g)
                 fieldstd_st = scaler.fit_transform(field_st_g)
                 
                 ustd = scaler.fit_transform(uprime_g)
                 ustd_lt = scaler.fit_transform(u_lt_g)
                 ustd_st = scaler.fit_transform(u_st_g)
                 
                 vstd = scaler.fit_transform(vprime_g)
                 vstd_lt = scaler.fit_transform(v_lt_g)
                 vstd_st = scaler.fit_transform(v_st_g)
                 
                 psstd = scaler.fit_transform(psprime_g)
                 psstd_lt = scaler.fit_transform(ps_lt_g)
                 psstd_st = scaler.fit_transform(ps_st_g)
                 
            else:
                 fieldstd = fieldprime_g
                 fieldstd_lt = field_lt_g
                 fieldstd_st = field_st_g 
                 
                 ustd = uprime_g
                 ustd_lt = u_lt_g
                 ustd_st = u_st_g 
                 
                 vstd = vprime_g
                 vstd_lt = v_lt_g
                 vstd_st = v_st_g 
                 
                 psstd = psprime_g
                 psstd_lt = ps_lt_g
                 psstd_st = ps_st_g 
             
            fieldclf = linear_model.LinearRegression()
            fieldclf_lt = linear_model.LinearRegression()
            fieldclf_st = linear_model.LinearRegression()
            
            uclf = linear_model.LinearRegression()
            uclf_lt = linear_model.LinearRegression()
            uclf_st = linear_model.LinearRegression()
            
            vclf = linear_model.LinearRegression()
            vclf_lt = linear_model.LinearRegression()
            vclf_st = linear_model.LinearRegression()
            
            psclf = linear_model.LinearRegression()
            psclf_lt = linear_model.LinearRegression()
            psclf_st = linear_model.LinearRegression()
    
        
             #THF LAGS SST
            if lag > 0:
                fieldclf.fit(MMstd[:-lag], fieldstd[lag:,:])
                fieldclf_lt.fit(MMstd_lt[:-lag], fieldstd_lt[lag:,:])
                fieldclf_st.fit(MMstd_st[:-lag], fieldstd_st[lag:,:])
                
                uclf.fit(MMstd[:-lag], ustd[lag:,:])
                uclf_lt.fit(MMstd_lt[:-lag], ustd_lt[lag:,:])
                uclf_st.fit(MMstd_st[:-lag], ustd_st[lag:,:])
                
                vclf.fit(MMstd[:-lag], vstd[lag:,:])
                vclf_lt.fit(MMstd_lt[:-lag], vstd_lt[lag:,:])
                vclf_st.fit(MMstd_st[:-lag], vstd_st[lag:,:])
                
                psclf.fit(MMstd[:-lag], psstd[lag:,:])
                psclf_lt.fit(MMstd_lt[:-lag], psstd_lt[lag:,:])
                psclf_st.fit(MMstd_st[:-lag], psstd_st[lag:,:])

        
            #THF LEADS SST
            elif lag < 0: 
                fieldclf.fit(MMstd[-lag:], fieldstd[:lag,:])
                fieldclf_lt.fit(MMstd_lt[-lag:], fieldstd_lt[:lag,:])
                fieldclf_st.fit(MMstd_st[-lag:], fieldstd_st[:lag,:])
                
                uclf.fit(MMstd[-lag:], ustd[:lag,:])
                uclf_lt.fit(MMstd_lt[-lag:], ustd_lt[:lag,:])
                uclf_st.fit(MMstd_st[-lag:], ustd_st[:lag,:])
                
                vclf.fit(MMstd[-lag:], vstd[:lag,:])
                vclf_lt.fit(MMstd_lt[-lag:], vstd_lt[:lag,:])
                vclf_st.fit(MMstd_st[-lag:], vstd_st[:lag,:])
                
                psclf.fit(MMstd[-lag:], psstd[:lag,:])
                psclf_lt.fit(MMstd_lt[-lag:], psstd_lt[:lag,:])
                psclf_st.fit(MMstd_st[-lag:], psstd_st[:lag,:])
        
            else:
                fieldclf.fit(MMstd, fieldstd)
                fieldclf_lt.fit(MMstd_lt, fieldstd_lt)
                fieldclf_st.fit(MMstd_st, fieldstd_st)
                
                uclf.fit(MMstd, ustd)
                uclf_lt.fit(MMstd_lt, ustd_lt)
                uclf_st.fit(MMstd_st, ustd_st)
                
                vclf.fit(MMstd, vstd)
                vclf_lt.fit(MMstd_lt, vstd_lt)
                vclf_st.fit(MMstd_st, vstd_st)
                
                psclf.fit(MMstd, psstd)
                psclf_lt.fit(MMstd_lt, psstd_lt)
                psclf_st.fit(MMstd_st, psstd_st)
        
                
                
            fieldlagcorrs[j,i,:] = np.squeeze(fieldclf.coef_)
            fieldlagcorrs_lt[j,i,:] = np.squeeze(fieldclf_lt.coef_)
            fieldlagcorrs_st[j,i,:] = np.squeeze(fieldclf_st.coef_)
            
            ulagcorrs[j,i,:] = np.squeeze(uclf.coef_)
            ulagcorrs_lt[j,i,:] = np.squeeze(uclf_lt.coef_)
            ulagcorrs_st[j,i,:] = np.squeeze(uclf_st.coef_)
            
            vlagcorrs[j,i,:] = np.squeeze(vclf.coef_)
            vlagcorrs_lt[j,i,:] = np.squeeze(vclf_lt.coef_)
            vlagcorrs_st[j,i,:] = np.squeeze(vclf_st.coef_)
            
            pslagcorrs[j,i,:] = np.squeeze(psclf.coef_)
            pslagcorrs_lt[j,i,:] = np.squeeze(psclf_lt.coef_)
            pslagcorrs_st[j,i,:] = np.squeeze(psclf_st.coef_)
       


#Plot maps of SST and THF patterns associated with MM
#CHANGE THIS FOR MAP PROJECTION
cent=-(lonbounds[1]-lonbounds[0])/2.
prj = cart.crs.PlateCarree(central_longitude=cent)
#prj = cart.crs.Mollweide(central_longitude=cent)

bnds = [np.round(lonbounds[0]-359), np.round(lonbounds[1]-361), latbounds[0], latbounds[1]]

#latitude/longitude labels

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

if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst':
    fieldmin=-0.5
    fieldmax=0.5
    fieldstep = 0.02
    cbstep = 0.25
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
elif fsave == 'u10m' or fsave == 'v10m':
    fieldmin = -1
    fieldmax = 1
    fieldstep = 0.5
    cbstep = 0.25

else:
    fieldmin=-10
    fieldmax=10
    fieldstep =0.2
    cbstep=2.5
    

#NAlats = lats[NAminlati:NAmaxlati]
#NEED TO AVERAGE OVER NA LONGITUDES
NAsst = spatial_ave(sst, lats)
NAsst_lt = spatial_ave(sst_lt, lats)

NAfield = spatial_ave(field, lats)
NAfield_lt = spatial_ave(field_lt, lats)



#PLOT MM INDEX AND FIELD
ci = (N_map-1)/2
fig=plt.figure(figsize=(14,22))
ax = fig.add_subplot(311)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, MM)
ax.plot(tyears[ci:-ci], MM_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'Atlantic Meridional Mode')
ax.axhline(0, color='black')
ax.set_ylabel(r'SST ($^{{\circ}}$C)')
ax.set_xlabel('Time (years)')
ax.legend(loc='upper right')
ax = fig.add_subplot(312)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, MMwind)
ax.plot(tyears[ci:-ci], MMwind_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'Atlantic Meridional Mode')
ax.legend(loc='upper right')
ax.axhline(0, color='black')
ax.set_ylabel(r'Wind (m/s)')
ax.set_xlabel('Time (years)')
ax = fig.add_subplot(313)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, NAfield)
ax.plot(tyears[ci:-ci], NAfield_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
ax.legend(loc='upper right')
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'{:s}'.format(ftitle))
ax.axhline(0, color='black')
ax.set_ylabel(r'{:s} ({:s})'.format(ftitle, units))
#ax.legend()
ax.set_xlabel('Time (years)')
plt.savefig(fout + '{:s}_MM{:s}_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
    
ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
ticklbls[ticklbls == -0.0] = 0.0
                 
sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

orient = 'horizontal'
if lonbounds[1] - lonbounds[0] <= 180:
    orient = 'vertical'

    

#plt.figure(figsize=(16,12))
#ax = plt.axes(projection=prj)
#ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=99, edgecolor='k', facecolor='grey')
#ax.set_xticks(mer, crs=prj)
#ax.set_yticks(par, crs=prj)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)
#ax.get_yaxis().set_tick_params(direction='out')
#ax.get_xaxis().set_tick_params(direction='out')
#ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
##ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
#plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())    
#cb = plt.colorbar(plot, orientation = orient, label='K')
#cb.set_ticks(sstticks)
#cb.set_ticklabels(sstticklbls)
#plt.title(r'regression of SST on AMM'.format(ftitle))
#plt.savefig(fout + '{:s}_AMM_sst_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()    
    
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
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
    #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, fieldcorrs, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())    
    cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    plt.title(r'{:s}'.format(ftitle))
    plt.savefig(fout + '{:s}_AMM_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
    plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
    cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    plt.title(r'Long-term {:s} ({:1.0f}-month RM)'.format(ftitle, N_map-1))
    plt.savefig(fout + '{:s}_AMM_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
    plt.title(r'Short-term {:s} (residual)'.format(ftitle))
    plt.savefig(fout + '{:s}_AMM_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close() 
    
if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst':
    fieldmin=-0.48
    fieldmax=0.48
    fieldstep = 0.02
    cbstep = 0.12
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
elif fsave == 'u10m' or fsave == 'v10m':
    fieldmin = -1
    fieldmax = 1
    fieldstep = 0.5
    cbstep = 0.25

else:
    fieldmin=-10
    fieldmax=10
    fieldstep =0.5
    cbstep=2.5
    
ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),2)
ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 2)
ticklbls[ticklbls == -0.0] = 0.0
    
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
    
cmap = plt.cm.RdBu_r
uskip=6

psmax=0.5
psmin = -0.5
psstep = 0.1
pscbstep = 0.1
SLPlevels = np.arange(psmin, psmax+psstep, psstep)

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
        ct = ax.contour(x, y, pslagcorrs[leadi,...], levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
        ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
        if np.any(np.round(ct.levels, 5) == 0):
            ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
        #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
        plot = ax.contourf(x, y, fieldlagcorrs[leadi,...], cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
        qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ulagcorrs[leadi, ::uskip,::uskip], vlagcorrs[leadi, ::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
        ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        plt.title(r'{:s}, SLP, and 10-m Winds ({:1.0f} month lag)'.format(ftitle, lags[leadi]))
        plt.savefig(fout + '{:s}_AMM_uvSLP{:s}_lag{:1.0f}corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lags[leadi], latbounds[0], latbounds[1], str(detr)[0]))
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
            ct = ax.contour(x, y, pslagcorrs_lt[leadi,...], levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
            ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
            if np.any(np.round(ct.levels, 5) == 0):
                ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
            #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
            plot = ax.contourf(x, y, fieldlagcorrs_lt[leadi,...], cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
            qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ulagcorrs_lt[leadi, ::uskip,::uskip], vlagcorrs_lt[leadi, ::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
            ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
            cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklbls)
            plt.title(r'Long-term {:s} and 10-m Winds ({:1.0f} month lag)'.format(ftitle, lags[leadi]))
            plt.savefig(fout + '{:s}_AMM_uvSLP{:s}_lag{:1.0f}ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lags[leadi], latbounds[0], latbounds[1], str(detr)[0]))
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
            ct = ax.contour(x, y, pslagcorrs_st[leadi,...], levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
            ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
            if np.any(np.round(ct.levels, 5) == 0):
                ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
            #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
            plot = ax.contourf(x, y, fieldlagcorrs_st[leadi,...], cmap=cmap, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
            qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ulagcorrs_st[leadi, ::uskip,::uskip], vlagcorrs_st[leadi, ::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
            ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
            cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklbls)
            plt.title(r'Short-term {:s}, SLP and 10-m Winds ({:1.0f} month lag)'.format(ftitle, lags[leadi]))
            plt.savefig(fout + '{:s}_AMM_uvSLP{:s}_lag{:1.0f}stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lags[leadi], latbounds[0], latbounds[1], str(detr)[0]))
            plt.close()


if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst':
    fieldmin=-0.3
    fieldmax=0.3
    fieldstep = 0.01
    cbstep = 0.1
else:
    fieldmin=-4
    fieldmax=4
    fieldstep =0.1
    cbstep=1.0
    
if corr:
    fieldminlag = -0.6 
    fieldmaxlag = 0.6
    cbsteplag = 0.2
    fieldunitslag = 'Correlation'

else:
    fieldminlag = fieldmin
    fieldmaxlag = fieldmax
    cbsteplag = cbstep 
    fieldunitslag = units
    
    
lagoffset = np.diff(lags)[0]/2.
latoffset = np.diff(lats)[0]/2.
    
ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
ticklbls[ticklbls == -0.00] = 0.00

laglabels = np.round(np.arange(-lagmax,lagmax+2*lagstep,2*lagstep))
                 
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

weights = np.cos(np.deg2rad(lats))
#CRE_surflagcorrs = np.ma.array(CRE_surflagcorrs, mask=~np.isfinite(CRE_surflagcorrs))
#CRE_surflagcorrs_lt = np.ma.array(CRE_surflagcorrs_lt, mask=~np.isfinite(CRE_surflagcorrs_lt))
#CRE_surflagcorrs_st = np.ma.array(CRE_surflagcorrs_st, mask=~np.isfinite(CRE_surflagcorrs_st))
fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs[:,...], axis=2)
fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_lt[:,...], axis=2)
fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_st[:,...], axis=2)

#pslagcorrs_zonalave = np.ma.average(pslagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#pslagcorrs_lt_zonalave = np.ma.average(pslagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#pslagcorrs_st_zonalave = np.ma.average(pslagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)#SHOULDN'T THIS BE EQUIVALENT TO THE CORRELATION BETWEEN SMOOTHED AMO AND NA THF? i.e. NACRE_surf_laggedcorr_lt[i,:]
#thflagcorrs_test = np.ma.average(thflagcorrs_lt_zonalave, axis=1, weights=weights[NAminlati:NAmaxlati])

cmap = plt.cm.PRGn

lagg, latt = np.meshgrid(lags-lagoffset, lats-latoffset)

if lagmax > 12:
    lagg = lagg/12.
    laglabels = laglabels/12.
    laglabels = laglabels.astype(int)
    lagunits = 'years'
else:
    lagunits = 'months'
    
if lterm:

    #Plot zonally-averaged lagged correlation between long-term AMO and THF
    fig=plt.figure(figsize=(22,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(131)
    h = ax.pcolor(lagg, latt, fieldlagcorrs_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('{:s}'.format(ftitle))
    else:
        ax.set_title('{:s}'.format(ftitle))
    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylabel('Latitude (degrees)')
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    ax = fig.add_subplot(132)
    ax.pcolor(lagg, latt, fieldlagcorrs_lt_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('Long-term {:s} ({:1.0f}-yr RM)'.format(ftitle, N_map/12.))
    else:
        ax.set_title('Long-term {:s} ({:1.0f}-yr RM)'.format(ftitle, N_map/12.))
    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.set_xticks(laglabels)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticklabels(laglabels)
    ax = fig.add_subplot(133)
    h = ax.pcolor(lagg, latt, fieldlagcorrs_st_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=cmap)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('Short-term {:s}'.format(ftitle))
    else:
        ax.set_title('Short-term {:s}'.format(ftitle))
    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.set_xticks(laglabels)
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticklabels(laglabels)
    cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    if corr:
            plt.savefig(fout + '{:s}_MM_{:s}_lagcorr_lagmax{:3.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    else:
            plt.savefig(fout + '{:s}_MM_{:s}_lagregr_lagmax{:3.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
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
    ax.set_xlabel(r'{:s} lag ({:s})'.format(ftitle, lagunits))
    ax.set_ylabel('Latitude (degrees)')
    ax.set_ylim(latbounds[0], latbounds[1])
    ax.axhline(0, color='k', linewidth=1)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
    cb.set_ticks(ticks)
    ax.axvline(0, color='k')
    if corr:
            plt.savefig(fout + '{:s}_MM_{:s}_lagcorr_lagmax{:3.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    else:
            plt.savefig(fout + '{:s}_MM_{:s}_lagregr_lagmax{:3.0f}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lagmax, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close() 















































