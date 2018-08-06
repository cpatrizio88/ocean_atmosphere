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
from sklearn.preprocessing import StandardScaler
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, calcsatspechum

fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
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

#v = fuv('V10M')
#v = fuv('v10')
#v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#v = v[tskip:,:]

#umag = np.sqrt(np.square(v) + np.square(u))

#qv10m = fRH('QV10M')
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

#field = ps
#ftitle = 'SLP'
#fsave = 'SLP'
#units = 'hPa'



#NAmaxlati = np.where(lats > maxlat)[0][0]
#NAminlati = np.where(lats > minlat)[0][0]

#sst = sst.subRegion(longitude=(minlon, maxlon))
#sst = sst[tskip:,NAminlati:NAmaxlati,:]
#field = field[tskip:,NAminlati:NAmaxlati,:]

field = field.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
field = field[tskip:nt_ps,:]
#field = field[tskip:-1,:]

#qsat = calcsatspechum(sst[:ps.shape[0],...], ps)
#qsat = qsat/1e3

#field = field[:ps.shape[0],...]/qsat

sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
#field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)

#True for detrending data, False for raw data
detr=True
corr=False

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
 #ps_an, params = detrend_separate(ps_an)
 field = detrend(field)

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]
nt = sst.shape[0]
lons[0] = 0
nlat = len(lats)
nlon = len(lons)

t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1980 + t/12.

tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(field)


#field = cdutil.ANNUALCYCLE.departures(field)
#sst = cdutil.ANNUALCYCLE.departures(sst)

#coarse grid lat/lon spacing
cstep=1.5
lats = np.arange(minlat,maxlat+cstep,cstep)
lons = np.arange(0,360+cstep,cstep)

nlat = len(lats)
nlon = len(lons)

cgrid = cdms2.createGenericGrid(lats,lons)
#regridfunc = Regridder(ingrid, cgrid)
sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#interpolate to coarser grid to speed up 
#sst= sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
tmonthinyear = t % 12

#cold and warm season months for NH... reverse for SH
coldmonths = [9,10,11,0,1,2]
warmmonths = [3,4,5,6,7,8]
twarmi = np.in1d(tmonthinyear, warmmonths)
twarmi = np.where(twarmi == True)
tcoldi = np.in1d(tmonthinyear, coldmonths)
tcoldi = np.where(tcoldi == True)
t = 1980 + t/12.

sstcold = np.squeeze(np.take(sst, tcoldi, axis=0))
sstwarm = np.squeeze(np.take(sst, twarmi, axis=0))

fieldcold = np.squeeze(np.take(field, tcoldi, axis=0))
fieldwarm= np.squeeze(np.take(field, twarmi, axis=0))
 
#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5
ci = (N_map-1)/2
ltlag = 5
stlag = 1

lagmax=3
lags = np.arange(-lagmax,lagmax+1)

#lags = np.arange(-3,0,6,3)

#sst_globe = spatial_ave(sst, lats)
sstcold_globe = spatial_ave(sstcold, lats)
sstwarm_globe = spatial_ave(sstwarm, lats)
fieldcold_globe = spatial_ave(fieldcold, lats)
fieldwarm_globe = spatial_ave(fieldwarm, lats)

#subtract global annual mean to isolate local processes
#sstcoldprime = sstcold.T - sstcold_globe
#sstcoldprime = sstcoldprime.T
sstcoldprime = sstcold

#sstwarmprime = sstwarm.T - sstwarm_globe
#sstwarmprime = sstwarmprime.T
sstwarmprime = sstwarm

#sstprime = sst

#fieldcoldprime = fieldcold.T - fieldcold_globe
#fieldcoldprime = fieldcoldprime.T
fieldcoldprime = fieldcold

#fieldwarmprime = fieldwarm.T - fieldwarm_globe
#fieldwarmprime = fieldwarmprime.T
fieldwarmprime = fieldwarm 

#fieldprime = field

#field_lt = running_mean(fieldprime, N_map)
#field_st = fieldprime[ci:-ci,:] - field_lt

nt = sst.shape[0]
#nt_lt = field_lt.shape[0]


#EDIT BOUNDS FOR AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
#for latbounds in latboundar:
    
fieldlagcorrs_cold = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_warm = np.zeros((len(lags), nlat, nlon))
#fieldlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
#fieldlagcorrs_st = np.zeros((len(lags), nlat, nlon))

#sstcorrs = MV.zeros((nlatc,nlonc))
#sstpvals = MV.zeros((nlat,nlon))
fieldcorrs_cold = MV.zeros((nlat, nlon))
fieldcorrs_warm = MV.zeros((nlat, nlon))
#fieldcorrs_lt = MV.zeros((nlat,nlon))
#fieldcorrs_st = MV.zeros((nlat,nlon))


#compute correlation between long-term/short-term AMO and 2D field
print r'calculating correlations between AMO and {:s}...'.format(ftitle)
for i in range(nlat):   
        print 'latitude', lats[i]
     
     #for j in range(nlon):

        sstcoldprime_g = sstcoldprime[:,i,:]
        sstwarmprime_g = sstwarmprime[:,i,:]
        fieldcoldprime_g = fieldcoldprime[:,i,:]
        fieldwarmprime_g = fieldwarmprime[:,i,:]
#        else:
#            sstcoldprime_g = sstwarmprime[:,i,:]
#            sstwarmprime_g = sstcoldprime[:,i,:]
#            fieldcoldprime_g = fieldwarmprime[:,i,:]
#            fieldwarmprime_g = fieldcoldprime[:,i,:]
        #field_lt_g = field_lt[:,i,:]
        #field_st_g = field_st[:,i,:]
         
        #sst_lt_g = running_mean(sstprime_g, N_map)
        #sst_st_g = sstprime_g[ci:-ci] - sst_lt_g
         
        scaler = StandardScaler()
         
        sstcoldstd = scaler.fit_transform(sstcoldprime_g)
        sstwarmstd = scaler.fit_transform(sstwarmprime_g)
        #sststd_lt = scaler.fit_transform(sst_lt_g)
        #sststd_st = scaler.fit_transform(sst_st_g)
        
        M_cold = sstcoldprime_g.shape[0]
        M_warm = sstwarmprime_g.shape[0]
        N = fieldcoldprime_g.shape[1]
        
        scaler = StandardScaler()
        if corr:
             fieldcoldstd = scaler.fit_transform(fieldcoldprime_g)
             fieldwarmstd = scaler.fit_transform(fieldwarmprime_g)
             #fieldstd_lt = scaler.fit_transform(field_lt_g)
             #fieldstd_st = scaler.fit_transform(field_st_g)
        else:
             fieldcoldstd = fieldcoldprime_g
             fieldwarmstd = fieldwarmprime_g
             #fieldstd_lt = field_lt_g
             #fieldstd_st = field_st_g 
        
        #M_lt = sststd_lt.shape[0]
        #M_st = sststd_st.shape[0]
         
        #coefs = np.diag(np.ma.matmul(sststd.T, fieldprime_g))/M
        
        coefscold = np.diag(np.ma.cov(sstcoldstd, fieldcoldprime_g, rowvar=False)[:N,N:])
        coefswarm= np.diag(np.ma.cov(sstwarmstd, fieldwarmprime_g, rowvar=False)[:N,N:])


        #coefs_lt = np.diag(np.matmul(sststd_lt.T, field_lt_g))/M_lt
        #coefs_st = np.diag(np.matmul(sststd_st.T, field_st_g))/M_st
         
        fieldcorrs_cold[i,:] = coefscold
        fieldcorrs_warm[i,:] = coefswarm
        #fieldcorrs_lt[i,:] = coefs_lt
        #fieldcorrs_st[i,:] = coefs_st
                   
lonbounds = [0.1,359.99]
latbounds = [minlat, maxlat]
    
NAminlati = np.where(lats >= latbounds[0])[0][0]
NAmaxlati = np.where(lats >= latbounds[1])[0][0]
NAminloni = np.where(lons >= lonbounds[0])[0][0]
NAmaxloni = np.where(lons >= lonbounds[1])[0][0]


#fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)

lagoffset = np.diff(lags)[0]/2.
latoffset = np.diff(lats)[0]/2.

laglabels = np.round(np.arange(-10,15,5))

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
elif fsave == 'Qnetsurf':
    fieldmin=-4
    fieldmax=4
    fieldstep = 0.1
    cbstep=1.0
elif fsave == 'thf':
    fieldmin=-4
    fieldmax=4
    fieldstep=0.1
    cbstep=1.0
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
    fieldminlag = -1 
    fieldmaxlag = 1
    cbsteplag = 0.2
    fieldunitslag = ''

else:
    fieldminlag = fieldmin
    fieldmaxlag = fieldmax
    cbsteplag = cbstep 
    fieldunitslag = units

ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
ticklbls[ticklbls == -0.00] = 0.00
      
#Plot maps of SST and THF patterns associated with AMO
#CHANGE THIS FOR MAP PROJECTION
cent=-180
prj = cart.crs.PlateCarree(central_longitude=cent)
#prj = cart.crs.Mollweide(central_longitude=cent)
bnds = [-179, 179, minlat, maxlat]

#latitude/longitude labels
par = np.arange(-90.,91.,30.)
mer = np.arange(-180.,180.,60.)

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
    fieldmin=-0.8
    fieldmax=0.8
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
    fieldmin=-60
    fieldmax=60
    fieldstep =0.5
    cbstep=10
    
ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
ticklbls[ticklbls == -0.0] = 0.0
                 
sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

#plt.figure(figsize=(16,12))
#ax = plt.axes(projection=prj)
fig, ax = plt.subplots(2, 1, subplot_kw=dict(projection=prj), figsize=(12,16))
ax[0,].coastlines()
ax[0,].add_feature(cart.feature.LAND, zorder=99, edgecolor='k', facecolor='grey')
ax[0,].set_xticks(mer, crs=prj)
ax[0,].set_yticks(par, crs=prj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[0,].xaxis.set_major_formatter(lon_formatter)
ax[0,].yaxis.set_major_formatter(lat_formatter)
ax[0,].get_yaxis().set_tick_params(direction='out')
ax[0,].get_xaxis().set_tick_params(direction='out')
ax[0,].set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
#ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
plot = ax[0,].contourf(x, y, fieldcorrs_cold, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
#ax.set_aspect('auto')
#cb = plt.colorbar(plot, orientation = 'horizontal', label=r'{:s}'.format(units))
#cb.set_ticks(ticks)
#cb.set_ticklabels(ticklbls)
ax[0,].set_title(r'regression of {:s} on SST$_{{cold}}$'.format(ftitle))
#ax = plt.subplot(2,1,2, projection=prj)
ax[1,].coastlines()
ax[1,].add_feature(cart.feature.LAND, zorder=99, edgecolor='k', facecolor='grey')
ax[1,].set_xticks(mer, crs=prj)
ax[1,].set_yticks(par, crs=prj)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax[1,].xaxis.set_major_formatter(lon_formatter)
ax[1,].yaxis.set_major_formatter(lat_formatter)
ax[1,].get_yaxis().set_tick_params(direction='out')
ax[1,].get_xaxis().set_tick_params(direction='out')
ax[1,].set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
#ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
plot = ax[1,].contourf(x, y, fieldcorrs_warm, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
#ax.set_aspect('auto')
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(plot, ax=ax.ravel().tolist(), fraction=0.046, pad=0.07, orientation = 'horizontal', label=r'{:s}'.format(units))
cb.set_ticks(ticks)
cb.set_ticklabels(ticklbls)
ax[1,].set_title(r'regression of {:s} on SST$_{{warm}}$'.format(ftitle))
#plt.tight_layout()
plt.savefig(fout + '{:s}_monthlylocalSST_seasons_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

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
#ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
##ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', inline=1, linewidths=1)
#plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
#cb = plt.colorbar(plot, orientation = 'horizontal', label=r'{:s}'.format(units))
#cb.set_ticks(ticks)
#cb.set_ticklabels(ticklbls)
#plt.title(r'regression of long-term {:s} on SST ({:1.0f}-yr RM)'.format(ftitle, N_map))
#plt.savefig(fout + '{:s}_localSST_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
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
#ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
##ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', inline=1, linewidths=1)
#plot = ax.contourf(x, y, fieldcorrs_st, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
#cb = plt.colorbar(plot, orientation = 'horizontal', label=r'{:s}'.format(units))
#cb.set_ticks(ticks)
#cb.set_ticklabels(ticklbls)
#plt.title(r'regression of short-term {:s} on SST ({:1.0f}-yr RM residual)'.format(ftitle, N_map))
#plt.savefig(fout + '{:s}_localSST_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#i = np.where(windows>N_map)[0][0]-1
#    
#    
#fieldcorrs_zonalave = np.ma.average(fieldcorrs[:,NAminloni:NAmaxloni], axis=-1)
#fig = plt.figure(figsize=(12,8))
#ax = fig.gca()
#ax.plot(fieldcorrs_zonalave, lats)
#ax.axvline(0, color='k')
#ax.get_yaxis().set_tick_params(direction='out')
#ax.get_xaxis().set_tick_params(direction='out')
#ax.set_xlabel(r'{:s}'.format(units))
#ax.set_ylabel(r'latitude ($^{\circ}$)')
#ax.set_ylim(0, 60)
#if fsave == 'thf' or fsave == 'Qnetsurf':
#    ax.set_xlim(-5.5,5.5)
##ax.set_ylim(50,1000)
##ax.invert_yaxis()
##cb = plt.colorbar(plot, label=r'{:s}'.format(units))
##cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
##cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
#plt.title(r'regression of {:s} on SST'.format(ftitle))
#plt.savefig(fout + '{:s}_localSST_{:s}_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#    
#
#lonbounds = [280,359.99]
#latbounds = [0,60]
    
#NAminlati = np.where(lats >= latbounds[0])[0][0]
#NAmaxlati = np.where(lats >= latbounds[1])[0][0]
#NAminloni = np.where(lons >= lonbounds[0])[0][0]
#NAmaxloni = np.where(lons >= lonbounds[1])[0][0]
#
#NAlats = lats[NAminlati:NAmaxlati]
#
#    
#if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
#    fieldmin=-5
#    fieldmax=5
#    fieldstep = 0.05
#    cbstep = 1.0
#elif fsave == 'sst':
#    fieldmin=-0.3
#    fieldmax=0.3
#    fieldstep = 0.02
#    cbstep = 0.1
#elif fsave == 'umag':
#    fieldmin=-0.3
#    fieldmax=0.3
#    fieldstep=0.01
#    cbstep = 0.1
#elif fsave == 'RH10m':
#    fieldmin = -2
#    fieldmax = 2
#    fieldstep = 0.01
#    cbstep = 0.5
#elif fsave == 'cE':
#    fieldmin = -0.5
#    fieldmax = 0.5
#    fieldstep = 0.01
#    cbstep = 0.1
#elif fsave == 'cD':
#    fieldmin = -0.5
#    fieldmax = 0.5
#    fieldstep = 0.01
#    cbstep = 0.1
#else:
#    fieldmin=-3
#    fieldmax=3
#    fieldstep =0.1
#    cbstep=1.0
#    
#if corr:
#    fieldminlag = -1.0 
#    fieldmaxlag = 1.0
#    cbsteplag = 0.2
#    fieldunitslag = ''
#
#else:
#    fieldminlag = fieldmin
#    fieldmaxlag = fieldmax
#    cbsteplag = cbstep 
#    fieldunitslag = units
#    
#ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
#ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
#ticklbls[ticklbls == -0.00] = 0.00
#                 
#fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
#
#weights = np.cos(np.deg2rad(lats))
#
#lagg, latt = np.meshgrid(lags-lagoffset, NAlats-latoffset)
#
##Plot zonally-averaged lagged correlation between SST and THF
#fig=plt.figure(figsize=(12,6))
##plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
#ax = fig.add_subplot(111)
#h = ax.pcolor(lagg, latt, fieldlagcorrs_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#ax.axvline(0, color='k')
#if corr:
#    ax.set_title('correlation of {:s} with SST'.format(ftitle))
#else:
#    ax.set_title('regression of {:s} on SST'.format(ftitle))
#ax.set_xlabel(r'{:s} lag (months)'.format(ftitle))
#ax.set_ylabel('latitude (degrees)')
#ax.set_ylim(NAlats[0],NAlats[-1])
#ax.set_xticks(laglabels)
#ax.set_xticklabels(laglabels)
#if len(latboundar) > 1:
#    ax.axhline(20, color='grey', linewidth=1)
#    ax.axhline(45, color='grey', linewidth=1)
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
#ax = fig.add_subplot(132)
#ax.pcolor(lagg, latt, fieldlagcorrs_lt_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#ax.axvline(0, color='k')
#if corr:
#    ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
#else:
#    ax.set_title('long-term regression ({:1.0f}-yr RM)'.format(N_map))
#ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
#ax.set_ylim(NAlats[0],NAlats[-1])
#ax.set_xticks(laglabels)
#ax.set_xticklabels(laglabels)
#if len(latboundar) > 1:
#    ax.axhline(latboundar[0][1], color='grey', linewidth=1)
#    ax.axhline(latboundar[1][1], color='grey', linewidth=1)
#ax = fig.add_subplot(133)
#h = ax.pcolor(lagg, latt, fieldlagcorrs_st_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#ax.axvline(0, color='k')
#if corr:
#    ax.set_title('short-term correlation')
#else:
#    ax.set_title('short-term regression')
#ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
#ax.set_ylim(NAlats[0],NAlats[-1])
#ax.set_xticks(laglabels)
#ax.set_xticklabels(laglabels)
#if len(latboundar) > 1:
#    ax.axhline(latboundar[0][1], color='grey', linewidth=1)
#    ax.axhline(latboundar[1][1], color='grey', linewidth=1)
#cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
#cb.set_ticks(ticks)
#cb.set_ticklabels(ticklbls)
#if corr:
#    if len(latboundar) > 1:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_zonalavelocal_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#    else:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#else:
#    if len(latboundar) > 1:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_zonalavelocal_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#    else:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close() 
##
##calculate average of lagged correlation within defined latbounds
#lats = lats[NAminlati:NAmaxlati]
#weights = np.cos(np.deg2rad(lats))
#tli = np.where(lats > latboundar[0][0])[0][0]
#tui = np.where(lats > latboundar[0][1]-0.001)[0][0]
#
#tfieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
#tfieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
#
#fig=plt.figure(figsize=(18,7))
#ax = fig.add_subplot(121)
#ax.plot(lags, tfieldlagcorrs_lt_ave)
#ax.axhline(0, color='black')
#ax.axvline(0, color='black')
#ax.set_ylim(fieldminlag,fieldmaxlag)
#ax.set_ylabel(r'{:s}'.format(fieldunitslag))
#if corr:
#    ax.set_title('long-term correlation of {:s} with SST ({:1.0f}-yr RM)'.format(ftitle, N_map))
#else:
#    ax.set_title('long-term regression of {:s} on SST ({:1.0f}-yr RM)'.format(ftitle, N_map))
#ax.set_xlabel('{:s} lag (years)'.format(ftitle))
#ax = fig.add_subplot(122)
#ax.plot(lags, tfieldlagcorrs_st_ave)
#ax.set_ylim(fieldminlag,fieldmaxlag)
#ax.set_ylabel(r'{:s}'.format(fieldunitslag))
#ax.axhline(0, color='black')
#ax.axvline(0, color='black')
#if corr:
#    ax.set_title('short-term correlation')
#else:
#    ax.set_title('short-term regression')
#ax.set_xlabel('{:s} lag (years)'.format(ftitle))
#if corr:
#    plt.savefig(fout + '{:s}_localSST_{:s}_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
#else:
#    plt.savefig(fout + '{:s}_localSST_{:s}_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
#plt.close() 

#













































