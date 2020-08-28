#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:40:34 2019

@author: cpatrizio
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
#import cdms2 as cdms2
#import cdutil
import matplotlib
#import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from scipy.stats.stats import pearsonr, linregress
#from sklearn import linear_model
#import pandas as pd
#from sklearn.preprocessing import StandardScaler, Imputer
#from ocean_atmosphere.llcmapping import LLCMapper
#from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave \
#  
#                                      calc_NA_globeanom, detrend_separate, detrend_common, butter_lowpass_filter, corr2_coeff, cov2_coeff, cov, cor
import xarray as xr
import xesmf as xe
import pandas as pd
import ocean_atmosphere.stats as st
import ocean_atmosphere.misc_fns as st2
import cmocean
import colorcet as cc
#import seaborn as sns
#import proplot as plot


fin = '/Volumes/GoogleDrive/My Drive/data_drive/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
#fin = '/Users/cpatrizio/data/OAFlux/'
#fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/ocean-atmosphere/localSST_global/'

fout = '/Users/cpatrizio/figures_arc/'


#MERRA-2
fsst =  xr.open_dataset(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = xr.open_dataset(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = xr.open_dataset(fin + 'MERRA2_rad_monthly1980to2017.nc')
cffile = xr.open_dataset(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
#fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
#fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')
ftau = xr.open_dataset(fin + 'MERRA2_tau_monthly1980to2019.nc')
fssh = xr.open_dataset(fin + 'ncep.ssh.198001-201912.nc')

fh = xr.open_dataset(fin + 'ncep.mixedlayerdepth.198001-201712.nc')
#fh = cdms2.open(fin + 'ECCO_mxldepth_monthly1992to2015.nc')

ssh = fssh.sshg

#OAFlux 
fin = '/Volumes/GoogleDrive/My Drive/data_drive/OAFlux/'
fsst =  xr.open_dataset(fin + 'oaflux_ts_1980to2017.nc')
fthf =   xr.open_dataset(fin + 'oaflux_thf_1980to2017.nc')




#h = fh('MXLDEPTH')
#
#h = np.ma.array(h, mask=np.isnan(h))
#
#hlats = h.getLatitude()[:,0]
#hlons = h.getLongitude()[0,:]
#hgrid = cdms2.createGenericGrid(hlats.getValue(),hlons.getValue())
#h.setGrid(hgrid)
#
#months = range(h.shape[0])
#ta = cdms2.createAxis(months)
#ta.id = 'time'
#ta.units = 'months since 1992-01-01'
#h.setAxis(0,ta)

h = fh.dbss_obml

#h = h.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
hbar = h.mean(dim='time')
#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201712.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#dataname = 'ERAi'
#dataname = 'MERRA2'
dataname = 'OAFlux'

matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'axes.titlesize': 36})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 24})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
matplotlib.rcParams.update({'axes.labelsize': 22})
matplotlib.rcParams.update({'ytick.labelsize': 22})
matplotlib.rcParams.update({'xtick.labelsize': 22})

#EDIT THIS FOR BOUNDS
lonbounds = [0,360]
latbounds = [-65.,65.]



#longitude bounds are in degrees East
#lonbounds = [140,180]
#latbounds = [30,50]


#lonbounds = [120,180]
#latbounds = [20,60]

#latw=5
#slats = np.array([35])

#KO extension    
#lonbounds = [153,168]
#latbounds = [34,43]

#lonbounds = [142,165]
#latbounds = [35,42]

#KO extension (long)
#lonbounds=[145,205]
#latbounds=[30,45]

#KOE (west)
#lonbounds=[145,175]
#latbounds=[30,45]

#KOE  (east)
#lonbounds=[175,205]
#latbounds=[30,45]

#ENA
#lonbounds = [315,345]
#latbounds = [30,50]


#ENP
#lonbounds = [210,230]
#latbounds = [38,53]

#SP
#latbounds=[-52,-38]
#lonbounds=[190,250]

#SubEP
#latbounds=[12,25]
#lonbounds=[215,235]

#North Atlantic
latbounds=[0,65]
lonbounds=[270,360]

##O'Reilly region (AMOmid)
#lonbounds=[300,340]
#latbounds=[40,60]

#Extratropical North Atlantic
#lonbounds=[315,345]
#latbounds=[47,60]

#Extropical East North Pacific
#lonbounds=[210,245]
#latbounds=[30,60]

#Equatorial Pacific
#lonbounds=[175,285]
#latbounds=[-8,5]

#Southern Oceans
#lonbounds=[0,360]
#latbounds=[-60,-18]





#NP
#lonbounds = [120,290]
#latbounds = [-10,60]


#NH
#lonbounds = [0,360]
#latbounds = [0,65]


#NEW REGIONS
#Southern Indian Ocean
#latbounds = [-40,-28]
#lonbounds = [60,115]

#Eastern South Pacific
#latbounds = [-50,-34]
#lonbounds = [220,280]


#Subtropical North Pacific
#latbounds=[12,25]
#lonbounds=[215,235]

#Subtropical South Atlantic
#latbounds=[-25,-10]
#lonbounds=[340,355]

#Subtropical North Atlantic
#latbounds = [10,30]
#lonbounds = [300,360]

#Subtropical South Pacific
#latbounds=[-22,-8]
#lonbounds=[260,280]

##Equatorial West Pacific
#latbounds=[-10,10]
#lonbounds=[165,205]

#Equatorial East Pacific
#latbounds=[-7,7]
#lonbounds=[230,270]

#Tropical Atlantic
#latbounds=[5,25]
#lonbounds=[300,360]



minlon=lonbounds[0]
maxlon=lonbounds[1]
minlat = latbounds[0]
maxlat= latbounds[1]

ps = fSLP.SLP
# ps = ps/1e2
# ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
# nt_ps = ps.shape[0]
# ps = ps[tskip:,:]

#ps = fsst('msl')
ps = ps/1e2
ps = ps.assign_coords(lon=(ps.lon % 360)).roll(lon=((ps.shape[2] // 2)-1))
ps = ps.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))




tskip = 12
#cf = cffile('MDSCLDFRCLO')
#cf = cffile('MDSCLDFRCHI')
#ps = fSLP('SLP')
#ps = ps/1e2
##ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#nt_ps = cf.shape[0]

#ps = fsst('msl')
#ps = ps/1e2
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#nt_ps = ps.shape[0]
#ps = ps[tskip:,:]


#sst = fsst('sst')
sst = fsst.tmpsf
#sst = fsst('TSKINWTR')
lats = sst.lat
sst = sst.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
#sst = sst[tskip:nt_ps,:]

#cE = fcE('CDH')
#cD = fcD('CN')

#lhf = fsst('EFLUXWTR')
#shf = fsst('HFLUXWTR')
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

lhf = fthf.lhtfl
shf = fthf.shtfl
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

#lhf = fthf('slhf')
#lhf = lhf/(12*3600)
#shf = fthf('sshf')
#sshf is accumulated 
#shf = shf/(12*3600)
thf = lhf + shf

taux = ftau.TAUXWTR
tauy = ftau.TAUYWTR
taux = taux.assign_coords(lon=(taux.lon % 360)).roll(lon=((taux.shape[2] // 2)-1))
tauy = tauy.assign_coords(lon=(tauy.lon % 360)).roll(lon=((tauy.shape[2] // 2)-1))
taux = taux.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
tauy = tauy.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

#thf = -thf
#thf is positive down in ERAi, convert to positive up
#thf = thf
#lhf = -lhf
#shf = -shf

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
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT

# Change longitude from -180 - 180 to 0 - 360
LW_net_surf = LW_net_surf.assign_coords(lon=(LW_net_surf.lon % 360)).roll(lon=((LW_net_surf.shape[2] // 2)-1))
SW_net_surf = SW_net_surf.assign_coords(lon=(SW_net_surf.lon % 360)).roll(lon=((SW_net_surf.shape[2] // 2)-1))

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs
#
Q_net_surf = LW_net_surf + SW_net_surf


thf = thf[:Q_net_surf.shape[0],:,:]
sst = sst[:Q_net_surf.shape[0],:,:]
h = h[:Q_net_surf.shape[0],:,:]
ssh = ssh[:Q_net_surf.shape[0],:,:]
taux = taux[:Q_net_surf.shape[0],:,:]
tauy = tauy[:Q_net_surf.shape[0],:,:]
ps = ps[:Q_net_surf.shape[0],:,:]
#SW_net_surf = SW_net_surf[:Q_net_surf.shape[0],:,:]


months = np.arange(sst.shape[0])
tyears = 1980 + months/12.

dates = pd.date_range('1980-01-01', periods=len(months), freq='MS')

sst.time.values = dates
thf.time.values = dates
h.time.values = dates
Q_net_surf.time.values = dates
SW_net_surf.time.values = dates
ssh.time.values = dates
taux.time.values = dates
tauy.time.values = dates
ps.time.values = dates

ii=-1

#if dataname == 'MERRA2':

sst = sst[tskip:ii,:,:]
h = h[tskip:ii,:,:]
ssh = ssh[tskip:ii,:,:]
Q_net_surf = Q_net_surf[tskip:ii,:,:]
SW_net_surf = SW_net_surf[tskip:ii,:,:]
thf = thf[tskip:ii,:,:]
taux = taux[tskip:ii,:,:]
tauy = tauy[tskip:ii,:,:]
ps = ps[tskip:ii,:,:]

    
#True for low-pass filtering 
lowpass = False
anom_flag = True
timetend=False
detr=True
rENSO=False
corr=True
lterm=True
drawmaps=True

#coarse grid lat/lon spacingt
cstep_lat=1.0
cstep_lon=1.0
lats = np.arange(minlat,maxlat+cstep_lat,cstep_lat)
lons = np.arange(minlon,maxlon+cstep_lon,cstep_lon)


ds_out = xr.Dataset({'lat': (['lat'], lats),
                     'lon': (['lon'], lons)})

regridder = xe.Regridder(sst, ds_out, 'bilinear', reuse_weights=True)
sst = regridder(sst)  # print basic regridder information.

regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=True)
h = regridder(h)

regridder = xe.Regridder(ssh, ds_out, 'bilinear', reuse_weights=True)
ssh = regridder(ssh)

regridder = xe.Regridder(thf, ds_out, 'bilinear', reuse_weights=True)
thf = regridder(thf)

regridder = xe.Regridder(hbar, ds_out, 'bilinear', reuse_weights=True)
hbar = regridder(hbar)

regridder = xe.Regridder(Q_net_surf, ds_out, 'bilinear', reuse_weights=True)
Q_net_surf  = regridder(Q_net_surf)

regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
taux  = regridder(taux)

regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
tauy  = regridder(tauy)

regridder = xe.Regridder(ps, ds_out, 'bilinear', reuse_weights=True)
ps  = regridder(ps)


#Ekman transport is c/f*(-tauy*dSST/dx + taux*dSST/dy)
#c is specific heat capacity of seawater == 3850 J/(kg C)

rho = 1000
c_p = 3850
dt = 30*3600*24
C = rho*c_p*h
Cbar = C.mean(dim='time')

# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)

# Remove linear trend
if detr: 
 sst = sst.fillna(0.)    
 sst = xr.DataArray(signal.detrend(sst, axis=0), dims=sst.dims, coords=sst.coords)   

 h = h.fillna(0.)    
 h = xr.DataArray(signal.detrend(h, axis=0), dims=h.dims, coords=h.coords)   
 
 thf = thf.fillna(0.)    
 thf = xr.DataArray(signal.detrend(thf, axis=0), dims=thf.dims, coords=thf.coords) 
  
 Q_net_surf = Q_net_surf.fillna(0.)    
 Q_net_surf = xr.DataArray(signal.detrend(Q_net_surf, axis=0), dims=Q_net_surf.dims, coords=Q_net_surf.coords) 
 
 SW_net_surf = SW_net_surf.fillna(0.)    
 SW_net_surf = xr.DataArray(signal.detrend(SW_net_surf, axis=0), dims=SW_net_surf.dims, coords=SW_net_surf.coords) 

 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12.*5
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

if not(lowpass):
    Tn = 0.
    
# Butterworth low-pass filter
if lowpass:
    #sst = xr.DataArray(st.butter_lowpass_filter(sst, cutoff, fs, order),dims=sst.dims,coords=sst.coords)
    #thf = xr.DataArray(st.butter_lowpass_filter(thf, cutoff, fs, order),dims=thf.dims,coords=thf.coords)
    #Q_net_surf = xr.DataArray(st.butter_lowpass_filter(Q_net_surf, cutoff, fs, order),dims=Q_net_surf.dims,coords=Q_net_surf.coords)
    sst = st.butter_lowpass_filter_xr(sst, cutoff, fs, order)
    thf = st.butter_lowpass_filter_xr(thf, cutoff, fs, order)
    Q_net_surf = st.butter_lowpass_filter_xr(Q_net_surf, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)

# Mask zero values (continents)    
sst = sst.where(~(sst==0))
thf = thf.where(~(thf==0))
Q_net_surf = Q_net_surf.where(~(thf==0))

# Compute SST tendency
tendsst = (sst.shift(time=-2)-sst)[:-2]
tendsst = tendsst/(2*dt)

nt = sst.shape[0]

thf = thf.isel(time=slice(1,nt-1))
Q_net_surf = Q_net_surf.isel(time=slice(1,nt-1))
sst = sst.isel(time=slice(1,nt-1))

nt = sst.shape[0]

# Make sure sst tendency times match up with other fields
tendsst.time.values = thf.time.values

Qr = Cbar*tendsst -(-thf + Q_net_surf)
Qr = Qr.transpose('time','lat','lon')

Q_s = -thf + Q_net_surf

nlat = len(lats)
nlon = len(lons)

##
lagmax = 6
lagstep = 1
lags = np.arange(-lagmax,lagmax+lagstep, lagstep)
#lags=-lags

lagcorrs = np.zeros((len(lags), nlat, nlon))

# Edit this to change field
field = Q_s
fieldname = 'Qs'
fieldtitle = r'$Q_s$'
if corr:
    units = ''
else:
    units = r'W/m$^2$'

for ll, lag in enumerate(lags):
    
    print('lag',lag)
    
    print 
    
    if corr:
       lagcorrs[ll,:] = st2.cor(field,sst,lagx=lag)
    else:
       # Standardize SST
       sst = (sst - sst.mean(dim='time'))/sst.std(dim='time')
       lagcorrs[ll,:], intercept = st2.reg(field,sst,lagx=lag)
    

# Plotting
bnds = [np.round(lonbounds[0]-359), np.round(lonbounds[1]-361), latbounds[0], latbounds[1]]

cent = (bnds[0]+bnds[1])/2.
prj = cart.crs.PlateCarree(central_longitude=cent)

pardiff = 30.
merdiff = 60.
if lonbounds[1] - lonbounds[0] <= 180:
    merdiff = 15.
if lonbounds[1] - lonbounds[0] <= 60:
    merdiff = 5.
if np.abs(latbounds[1]-latbounds[0]) <= 30:
    pardiff=5.
par = np.arange(-90.,91.,pardiff)
mer = np.arange(-180.,180.,merdiff)
x, y = np.meshgrid(lons, lats)


orient = 'horizontal'
if np.abs(latbounds[1] - latbounds[0]) > np.abs(lonbounds[1] - lonbounds[0]):
    orient = 'vertical'
    
    
#sstvmax = 1.6
#vmin=-1.0
#vmax=1.0
#if lowpass:
#    sstvmax = 0.5
#    vmin=-0.5
#    vmax=0.5
    
vmin=-1.0
vmax=1.0
if not(corr):
   vmin=-15.0
   vmax=15.0
    

    
#sstcmap = cmocean.cm.plasma
#sstcmap = plt.cm.cubehelix_r
#fieldcmap = cmocean.cm.balance
fieldcmap = plt.cm.RdBu_r

#sstcmap = cc.cm.CET_L17
sstcmap = cc.cm.CET_L19
#sstcmap = cc.cm.isolum
#sstcmap = cc.cm.fire
#fieldcmap = cc.cm.CET_D1A
#fieldcmap = cc.cm.coolwarm

for ll, lag in enumerate(lags):
              
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
    hh=ax.pcolormesh(x, y, lagcorrs[ll,:], cmap=fieldcmap, vmin=vmin, vmax=vmax, transform=cart.crs.PlateCarree())
    cb = plt.colorbar(hh, orientation = orient, label=r'{:s}'.format(units))
    plt.title(r'{:s}'.format(fieldtitle))
    if corr:
        plt.savefig(fout + '{:s}_sst{:s}corr_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, lag, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_sst{:s}regr_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, lag, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()






#
#f, axs = plot.subplots(nrows=2, axwidth=2.5, proj='pcarree', proj_kw={'lon_0':cent})
#
##levels = np.linspace(0,sstvmax,10)
#cblabels = np.arange(0,sstvmax+0.2,0.2)
#axs[0].set_title(r'$\sigma_T^2$')
#hh=axs[0].pcolormesh(x, y, T_var, levels=20,cmap='sunrise', vmin=0,vmax=sstvmax)
#axs[0].add_feature(cart.feature.LAND, zorder=99, linewidth=0.5, edgecolor='k',facecolor='grey')
#axs[0].set_xticks(mer, crs=prj)
#axs[0].set_yticks(par, crs=prj)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#axs[0].xaxis.set_major_formatter(lon_formatter)
#axs[0].yaxis.set_major_formatter(lat_formatter)
##axs[0].get_yaxis().set_tick_params(direction='out')
##axs[0].get_xaxis().set_tick_params(direction='out')
#axs[0].set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
#axs.format(suptitle='SST Variance', coast=True, latlines=pardiff, lonlines=merdiff)
#
#axs[1].set_title(r'SUM')
#hh=axs[1].pcolormesh(x, y, T_var_sum, levels=20, cmap='sunrise', vmin=0, vmax=sstvmax)
#axs[1].add_feature(cart.feature.LAND, zorder=99, linewidth=0.5, edgecolor='k',facecolor='grey')
#axs[1].set_xticks(mer, crs=prj)
#axs[1].set_yticks(par, crs=prj)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#axs[1].xaxis.set_major_formatter(lon_formatter)
#axs[1].yaxis.set_major_formatter(lat_formatter)
##axs[0].get_yaxis().set_tick_params(direction='out')
##axs[0].get_xaxis().set_tick_params(direction='out')
#axs[1].set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
##cb = plot.colorbar(hh, orientation = orient, label=r'K$^{2}$')
#cb=f.colorbar(hh, label=r'K$^{2}$', ticks=cblabels)
#
##cb.ax.set_yticklabels(['{:2.1f}'.format(i) for i in cblabels])
##plot.show()
#plt.savefig(fout + '{:s}_Tvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()













