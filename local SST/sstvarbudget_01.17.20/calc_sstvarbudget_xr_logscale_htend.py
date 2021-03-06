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
import matplotlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import xarray as xr
import xesmf as xe
import pandas as pd
import ocean_atmosphere.stats as st
import ocean_atmosphere.misc_fns as st2
import colorcet as cc
import importlib
import ocean_atmosphere.map
importlib.reload(ocean_atmosphere.map)
from ocean_atmosphere.map import Mapper
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import cmocean
#import proplot as plot
#import seaborn as sns
#import proplot as plot


#fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
#fin = '/Users/cpatrizio/data/OAFlux/'
#fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/ocean-atmosphere/localSST_global/'

fin = '/Volumes/GoogleDrive/My Drive/data_drive/MERRA2/'

fout = '/Users/cpatrizio/figures_arc/'


#MERRA-2
fsst =  xr.open_dataset(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fthf = xr.open_dataset(fin + 'MERRA2_thf_monthly1980to2017.nc')
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
fseaice = xr.open_dataset(fin + 'MERRA2_seaice_monthly1980to2019.nc')

#fh = xr.open_dataset(fin + 'ncep.mixedlayerdepth.198001-201712.nc')

#dataname = 'ERAi'
#dataname = 'MERRA2'
#dataname = 'OAFlux'
#dataname = 'ERA5'
dataname = 'ECCO'

#ECCO
fin = '/Volumes/GoogleDrive/My Drive/data_drive/ECCO/'
#ft= xr.open_dataset(fin + 'ECCO_theta_monthly1992to2015.nc')
fh = xr.open_dataset(fin + 'ECCO_mxldepth_monthly1992to2015.nc')
#fTmxlfrac = xr.open_dataset(fin + 'ECCO_Tmxlfrac.nc')

fsst = xr.open_dataset(fin + 'ecco_SST.nc')
fTmxl = xr.open_dataset(fin + 'ecco_T_mxl.nc')

#fsst = fsst.rename({'__xarray_dataarray_variable__':'Ts'})
#fTmxl = fTmxl.rename({'__xarray_dataarray_variable__':'Tmxl'})

sst_ECCO = fsst.Ts
Tmxl_ECCO = fTmxl.Tmxl

#Tmxlfrac = Tmxl_ECCO/sst_ECCO

#Tmxlfrac = fTmxlfrac.Tmxlfrac

ssh = fssh.sshg


#OAFlux 
fin = '/Volumes/GoogleDrive/My Drive/data_drive/OAFlux/'
fsstoa =  xr.open_dataset(fin + 'oaflux_ts_1980to2017.nc')
fthf =   xr.open_dataset(fin + 'oaflux_thf_1980to2017.nc')

#ISCCP 
#fin_rad = '/Users/cpatrizio/data/ISCCP/'
#lwfile = xr.open_dataset(fin_rad + 'ISCCP_lw_1983to2009.nc')
#swfile = xr.open_dataset(fin_rad + 'ISCCP_sw_1983to2009.nc')

#ERA5
#fin = '/Users/cpatrizio/data/ERA5/'
#fsst = xr.open_dataset(fin + 'ERA5_sst_monthly1979to2019.nc')
#fthf = xr.open_dataset(fin + 'ERA5_thf_monthly1979to2019.nc')
#frad = xr.open_dataset(fin + 'ERA5_rad_monthly1979to2019.nc')

h = fh.MXLDEPTH
#theta = ft.THETA

time = h.tim
lats = h.lat[:,0]
lons = h.lon[0,:]
#z = theta.dep
#z = z.rename({'i2':'k'})

h.i1.values = h.tim.values[:]
h.i2.values = h.lat.values[:,0]
h.i3.values = h.lon.values[0,:]

h = h.drop('lat')
h = h.drop('lon')
h = h.drop('tim')

h = h.rename({'i1':'time','i2': 'lat', 'i3':'lon'})

h = h[1:,:]


#theta.i1.values = theta.tim.values[:]
#theta.i2.values = theta.dep.values[:]
#theta.i3.values = theta.lat.values[:,0]
#theta.i4.values = theta.lon.values[0,:]
#
#theta = theta.drop('lat')
#theta = theta.drop('lon')
#theta = theta.drop('tim')
#theta = theta.drop('dep')
#
#theta = theta.rename({'i1':'time','i2': 'k', 'i3':'lat', 'i4':'lon'})
#
#delz = z.diff(dim='k')
#
#mxldepth_clim = h.mean(dim='time')
#
#mxlpoints = theta.k < mxldepth_clim
#
#delz_sum = delz.where(mxlpoints).sum(dim='k')
#
#weights = delz/delz_sum
#
##sst = theta.isel(k=0)
#
#theta_mxl = (weights*theta).where(mxlpoints).sum(dim='k')

#"
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

#h = fh.dbss_obml

#h = h.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
hbar = h.mean(dim='time')
#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201712.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#plot.rc.update({'mathtext.fontset': 'cm'})


matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'axes.titlesize': 30})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 24})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
#matplotlib.rcParams.update({'axes.labelsize': 22})
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
#latbounds=[0,65]
#lonbounds=[270,360]

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
#latbounds = [15,60]





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




#ERA5
#sst = fsst.sst
#lhf = fthf.mslhf
#shf = fthf.msshf
#LW_net_surf = frad.msnlwrf
#SW_net_surf = frad.msnswrf

#MERRA-2
#sst = fsst.TSKINWTR
#lhf = fthf.EFLUX
#shf = fthf.HFLUX


#OAFlux
#sst = fsstoa.tmpsf
lhf = fthf.lhtfl
shf = fthf.shtfl
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT

#ECCO
sst = fsst.Ts
#sst = fTmxl.Tmxl
#sst_interp = sst
#sst_interp= theta.isel(k=0)
#sst = theta_mxl


thf = lhf + shf
Q_net_surf = LW_net_surf + SW_net_surf

if dataname == 'ERA5':
    sst = sst.rename({'latitude':'lat', 'longitude':'lon'})
    sst = sst.sortby('lat',ascending=True)


lats = sst.lat
lons = sst.lon

#lons_interp = sst_interp.lon
#lats_interp = sst_interp.lat
if lons.max() <= 180:
    sst = sst.assign_coords(lon=(sst.lon % 360)).roll(lon=((len(lons) // 2)))
    
#if lons_interp.max() <= 180:
#   sst_interp = sst_interp.assign_coords(lon=(sst_interp.lon % 360)).roll(lon=((len(lons_interp) // 2)))
 
sst = sst.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

sst = sst.transpose('time', 'lat', 'lon')

#sst_interp = sst_interp.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

#sst_interp = sst_interp.transpose('time', 'lat', 'lon')
#f_seaice = fseaice.FRSEAICE
#f_seaice = f_seaice.rename({'TIME':'time', 'XDim': 'lon', 'YDim': 'lat'})
#f_seaice = f_seaice.assign_coords(lon=(f_seaice.lon % 360)).roll(lon=((f_seaice.shape[2] // 2)-1))
#f_ocean = 1 - f_seaice
#
#f_oceanbar = f_ocean.mean(dim='time')

#thf = f_oceanbar*thf
#lhf = f_oceanbar*lhf
#shf = f_oceanbar*shf

if dataname == 'ERA5':
    lhf = lhf.rename({'latitude':'lat', 'longitude':'lon'})
    thf=thf.rename({'latitude':'lat', 'longitude':'lon'})
    shf = shf.rename({'latitude':'lat', 'longitude':'lon'})
    thf = thf.sortby('lat',ascending=True)
    lhf = lhf.sortby('lat',ascending=True)
    shf = shf.sortby('lat',ascending=True)
    

#In ERA5 the turbulent heat fluxes are defined positive down
if dataname == 'ERA5':
    thf = -thf
    lhf = -lhf
    shf = -shf
if thf.lon.max() <= 180:
    thf = thf.assign_coords(lon=(thf.lon % 360)).roll(lon=((thf.shape[2] // 2)-1))
    
thf = thf.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
     
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



if dataname == 'ERA5':
    Q_net_surf=Q_net_surf.rename({'latitude':'lat', 'longitude':'lon'})
    Q_net_surf=Q_net_surf.sortby('lat',ascending=True)


if Q_net_surf.lon.max() <=180:
    Q_net_surf = Q_net_surf.assign_coords(lon=(Q_net_surf.lon % 360)).roll(lon=((Q_net_surf.shape[2] // 2)-1))
    #W_net_surf = LW_net_surf.assign_coords(lon=(LW_net_surf.lon % 360)).roll(lon=((LW_net_surf.shape[2] // 2)-1))
    #SW_net_surf = SW_net_surf.assign_coords(lon=(SW_net_surf.lon % 360)).roll(lon=((SW_net_surf.shape[2] // 2)-1))

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs
#




Q_net_surf = Q_net_surf.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

if not(dataname == 'ECCO'):
    sst = sst[:Q_net_surf.shape[0],:,:]
    #sst_interp = sst_interp[:Q_net_surf.shape[0],:,:]
thf = thf[:Q_net_surf.shape[0],:,:]
ssh = ssh[:Q_net_surf.shape[0],:,:]
taux = taux[:Q_net_surf.shape[0],:,:]
tauy = tauy[:Q_net_surf.shape[0],:,:]
ps = ps[:Q_net_surf.shape[0],:,:]
#SW_net_surf = SW_net_surf[:Q_net_surf.shape[0],:,:]


months_sst = np.arange(sst.shape[0])
#months_sst_interp = np.arange(sst_interp.shape[0])
months = np.arange(thf.shape[0])
tyears = 1980 + months/12.
if dataname == 'ERA5':
    dates = pd.date_range('1979-01-01', periods=len(months), freq='MS')
else:
    dates = pd.date_range('1980-01-01', periods=len(months), freq='MS')
    
if dataname == 'ECCO':
    dates_sst = pd.date_range('1992-01-01', periods=len(months_sst), freq='MS')
#    dates_sst_interp = pd.date_range('1992-01-01', periods=len(months_sst_interp), freq='MS')
else:
    dates_sst = dates

sst.time.values = dates_sst
#sst_interp.time.values = dates_sst_interp

thf.time.values = dates
h.time.values = dates_sst
Q_net_surf.time.values = dates
#SW_net_surf.time.values = dates
#ssh.time.values = dates
taux.time.values = dates
tauy.time.values = dates
if dataname == 'ECCO':
#   sst_interp = sst_interp.sel(time=slice('1992-02-01','2015-12-01'))
    h = h.sel(time=slice('1992-02-01','2015-11-01'))
    sst = sst.sel(time=slice('1992-02-01','2015-11-01'))
    thf = thf.sel(time=slice('1992-02-01','2015-11-01'))
    Q_net_surf = Q_net_surf.sel(time=slice('1992-02-01','2015-11-01'))
    taux = taux.sel(time=slice('1992-02-01','2015-11-01'))
    tauy = tauy.sel(time=slice('1992-02-01','2015-11-01'))
    #h = h.sel(time=slice('1992-02-01','2015-12-01'))



ii=-1

#if dataname == 'MERRA2':

#sst = sst[tskip:ii,:,:]
##h = h[tskip:ii,:,:]
#ssh = ssh[tskip:ii,:,:]
#Q_net_surf = Q_net_surf[tskip:ii,:,:]
#SW_net_surf = SW_net_surf[tskip:ii,:,:]
#thf = thf[tskip:ii,:,:]
#taux = taux[tskip:ii,:,:]
#tauy = tauy[tskip:ii,:,:]
#ps = ps[tskip:ii,:,:]

    
#True for low-pass filtering 
lowpass = True
anom_flag = True
timetend=False
detr=True
rENSO=False
corr=False
lterm=True
drawmaps=True
drawbox=False
Qekplot = True

#interpolate to same grid
cstep_lat=1.0
cstep_lon=1.0
if dataname == 'ERA5':
    cstep_lat=0.5
    cstep_lon=0.5
if dataname == 'MERRA2':
    cstep_lat = 0.5
    cstep_lon= 0.625
if dataname == 'OAFlux':
    cstep_lat=1.0
    cstep_lon=1.0

#cstep_lat = 2.0
#cstep_lon = 2.0
lats = np.arange(minlat,maxlat+cstep_lat,cstep_lat)
lons = np.arange(minlon,maxlon+cstep_lon,cstep_lon)

thf = thf.transpose('time','lat','lon')
lhf = lhf.transpose('time', 'lat','lon')
shf = shf.transpose('time', 'lat','lon')
Q_net_surf = Q_net_surf.transpose('time', 'lat','lon')


ds_out = xr.Dataset({'lat': (['lat'], lats),
                     'lon': (['lon'], lons)})

regridder = xe.Regridder(sst, ds_out, 'bilinear', reuse_weights=True)
sst = regridder(sst)  # print basic regridder information.

regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=True)
h = regridder(h)

#regridder = xe.Regridder(Tmxlfrac, ds_out, 'bilinear', reuse_weights=True)
#Tmxlfrac = regridder(Tmxlfrac)

regridder = xe.Regridder(thf, ds_out, 'bilinear', reuse_weights=True)
thf = regridder(thf)
#
regridder = xe.Regridder(hbar, ds_out, 'bilinear', reuse_weights=True)
hbar = regridder(hbar)

regridder = xe.Regridder(Q_net_surf, ds_out, 'bilinear', reuse_weights=True)
Q_net_surf  = regridder(Q_net_surf)

regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
taux  = regridder(taux)
#
regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
tauy  = regridder(tauy)

regridder = xe.Regridder(sst_ECCO, ds_out, 'bilinear', reuse_weights=True)
sst_ECCO  = regridder(sst_ECCO)
#
regridder = xe.Regridder(Tmxl_ECCO, ds_out, 'bilinear', reuse_weights=True)
Tmxl_ECCO  = regridder(Tmxl_ECCO)


##cstep_lat = 2.0
##cstep_lon = 2.0
#lats_in = sst.lat.values
#lons_in = sst.lon.values
#
#lats_in = lats
#lons_in = lons
#
#dlat = np.diff(lats)[0]
#dlon = np.diff(lons)[0]
#
#lats_in_b = 0.5*(lats_in[1:] + lats_in[:-1])
#
#lats_in_b = np.insert(lats_in_b,0,lats_in_b[0]-dlat)
#lats_in_b = np.append(lats_in_b,lats_in_b[-1]+dlat)
#
#lons_in_b =  0.5*(lons_in[1:] + lons_in[:-1])
#
#lons_in_b = np.insert(lons_in_b,0,lons_in_b[0]-dlat)
#lons_in_b = np.append(lons_in_b,lons_in_b[-1]+dlat)
#    
#lats_out_b = np.arange(minlat,maxlat+cstep_lat,cstep_lat)
#lons_out_b = np.arange(minlon,maxlon+cstep_lon,cstep_lon)
#
#lons_out = 0.5*(lons_out_b[1:]+lons_out_b[:-1]) 
#lats_out = 0.5*(lats_out_b[1:]+lats_out_b[:-1])
#
#grid_in = {'lon': lons_in, 'lat': lats_in, 'lon_b': lons_in_b, 'lat_b': lats_in_b}
#grid_out = {'lon': lons_out, 'lat': lats_out, 'lon_b': lons_out_b, 'lat_b': lats_out_b}
#
#
#thf = thf.transpose('time','lat','lon')
#lhf = lhf.transpose('time', 'lat','lon')
#shf = shf.transpose('time', 'lat','lon')
#Q_net_surf = Q_net_surf.transpose('time', 'lat','lon')
#
#
#ds_out = xr.Dataset({'lat': (['lat'], lats),
#                     'lon': (['lon'], lons)})
#
#
#
#regridder = xe.Regridder(grid_in, grid_out, 'conservative')
#regridder.clean_weight_file()

taux_clim = taux.mean(dim='time')
tauy_clim = tauy.mean(dim='time')

sst_mean = sst.mean(dim='time')

lats = sst.lat
lons = sst.lon


# Scale the SST anomalies by the ratio of the mixed layer temperature variance to the SST in ECCO 
# This is to account for lower variability when averaged over the entire mixed layer
#sst = sst*Tmxlfrac


rho = 1000
c_p = 3850
dt = 30*3600*24
C = rho*c_p*h


Cbar = C.mean(dim='time')
Tbar = sst.mean(dim='time')

#if dataname == 'ECCO':
#    Cbar = rho*c_p*delz_sum


#Ekman transport is c/f*(-tauy*dSST/dx + taux*dSST/dy)
#c is specific heat capacity of seawater == 3850 J/(kg C)
omega = 7.2921e-5
rho = 1000
f = 2*omega*np.sin(np.deg2rad(lats))
r = 6.371e6
g=9.81

dphi = np.diff(lats)[0]*(2*np.pi/360.)
dpsi = np.diff(lons)[0]*(2*np.pi/360.)

dx = r*np.cos(np.deg2rad(lats))*dpsi
dy = r*dphi

nt = sst.shape[0]
nlat = len(lats)
nlon = len(lons)

dx2D = np.zeros((nt, nlat))
dx2D[:,:] = dx
dx3D = np.repeat(dx2D[:,:,np.newaxis],nlon-1,axis=2)
#
f2D = np.zeros((nt, nlat))
f2D[:,:] = f
f3D = np.repeat(f2D[:,:,np.newaxis],nlon,axis=2)

hbar3D = np.ma.zeros((nt, nlat, nlon))
hbar3D[:,:,:] = hbar[:,:]

dSSTdx_temp = sst.diff(dim='lon',n=1)/(dx3D)

dSSTdy_temp = sst.diff(dim='lat',n=1)/(dy)

times = sst.time
lats = sst.lat
lons = sst.lon

dSSTdx = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

dSSTdy = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

dSSTdx.values[:,:,1:] = dSSTdx_temp
dSSTdy.values[:,1:,:] = dSSTdy_temp

dSSTdy_mean = dSSTdy.mean(dim='time')
dSSTdx_mean = dSSTdx.mean(dim='time')

u_ek = (1/(rho*f3D*hbar3D))*tauy
v_ek = -(1/(rho*f3D*hbar3D))*taux

u_ek =  xr.DataArray(u_ek, dims=sst.dims, coords=sst.coords) 
v_ek =  xr.DataArray(v_ek, dims=sst.dims, coords=sst.coords) 

Q_ek = -Cbar*(u_ek*dSSTdx + v_ek*dSSTdy)

#eqi = np.where(lats==0)
#Q_ek = Q_ek.transpose('time','lat','lon')
#Q_ek.values[:,eqi,:] = 0

Q_ek_mean = Q_ek.mean(dim='time')

Q_ek = Q_ek.where(np.abs(lats) > 0)


# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    Q_ek,Q_ek_clim= st.anom(Q_ek)
    sst_ECCO,sst_ECCO_clim= st.anom(sst_ECCO)
    Tmxl_ECCO,Tmxl_ECCO_clim= st.anom(Tmxl_ECCO)
    h,h_clim = st.anom(h)



# Remove linear trend
if detr: 
 sst = sst.fillna(0.)    
 sst = xr.DataArray(signal.detrend(sst, axis=0), dims=sst.dims, coords=sst.coords)   

# h = h.fillna(0.)    
# h = xr.DataArray(signal.detrend(h, axis=0), dims=h.dims, coords=h.coords)   
 
 thf = thf.fillna(0.)    
 thf = xr.DataArray(signal.detrend(thf, axis=0), dims=thf.dims, coords=thf.coords) 
  
 Q_net_surf = Q_net_surf.fillna(0.)    
 Q_net_surf = xr.DataArray(signal.detrend(Q_net_surf, axis=0), dims=Q_net_surf.dims, coords=Q_net_surf.coords) 
 
 h = h.fillna(0.)
 h = xr.DataArray(signal.detrend(h, axis=0), dims=h.dims, coords=h.coords)   
 
# SW_net_surf = SW_net_surf.fillna(0.)    
# SW_net_surf = xr.DataArray(signal.detrend(SW_net_surf, axis=0), dims=SW_net_surf.dims, coords=SW_net_surf.coords) 
 
 Q_ek = Q_ek.fillna(0.)    
 Q_ek = xr.DataArray(signal.detrend(Q_ek, axis=0), dims=Q_ek.dims, coords=Q_ek.coords) 
# 
#  
# Q_ek_f = Q_ek_f.fillna(0.)    
# Q_ek_f = xr.DataArray(signal.detrend(Q_ek_f, axis=0), dims=Q_ek_f.dims, coords=Q_ek_f.coords) 
 


#eqi = np.where(lats[1:-1]==0)
#Q_ek[:,eqi,:] = 0
#Q_ek_test = Q_ek
h_var = h.var(dim='time')

#adv_hs = -(adv_hDivH['X'] + adv_hDivH['Y'])/vol
 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12.*3
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
    Q_ek = st.butter_lowpass_filter_xr(Q_ek, cutoff, fs, order)
    h = st.butter_lowpass_filter_xr(h, cutoff, fs, order)
    #Q_ek_f = st.butter_lowpass_filter_xr(Q_ek_f, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)

# Mask zero values (continents) 
ocean_points1 = ~(sst==0)
ocean_points2 = ~(xr.ufuncs.isnan(sst))
ocean_points = xr.ufuncs.logical_or(ocean_points1, ocean_points2)
sst = sst.where(ocean_points)
thf = thf.where(ocean_points)
Q_net_surf = Q_net_surf.where(ocean_points)
Q_ek = Q_ek.where(ocean_points)
#Q_ek = Q_ek.where(~(Q_ek==0))
#Q_ek = Q_ek.where(np.abs(lats) > 0)
# Compute SST tendency


sst_var = sst.var(dim='time')


Tmxl_var_ECCO = Tmxl_ECCO.var(dim='time')
sst_var_ECCO = sst_ECCO.var(dim='time')

# Scale the SST tendency by the ratio of MXL temp tendency / SST tendency from ECCO (as an esimate for a more physical result)
#Tmxlfrac = Tmxl_var_ECCO/sst_var_ECCO
#sst = sst*np.sqrt(Tmxlfrac)
#sst = sst*Tmxlfrac

tendh = (h.shift(time=-2)-h)[:-2]

tendh = tendh/(2*dt)

tendH = rho*c_p*Tbar*tendh

tendH = tendH.transpose('time','lat','lon')

tendsst = (sst.shift(time=-2)-sst)[:-2]

tendsst = tendsst/(2*dt)
#tendsst = tendsst/dt

nt = sst.shape[0]

thf = thf.isel(time=slice(1,nt-1))
Q_net_surf = Q_net_surf.isel(time=slice(1,nt-1))
sst = sst.isel(time=slice(1,nt-1))


nt = sst.shape[0]

# Make sure sst tendency times match up with other fields
tendsst.time.values = thf.time.values

Qr = Cbar*tendsst - (-thf + Q_net_surf) - Q_ek

Qr = Qr - tendH

#Qr = Cbar*tendsst -(-thf + Q_net_surf)
Qr = Qr.transpose('time','lat','lon')

Qr_mean = Qr.mean(dim='time')

Q_s = -thf + Q_net_surf + Q_ek

#Q_s = -thf + Q_net_surf 

Qr_var = Qr.var(dim='time')
Q_s_var = Q_s.var(dim='time')

covQsQo = st2.cov(Qr,Q_s)

nlat = len(lats)
nlon = len(lons)




# Compute covariance betwen SST tendency and fields
cov_Qr = st2.cov(tendsst, Qr)
cov_Qs = st2.cov(tendsst, Q_s)
cov_Qek = st2.cov(tendsst, Q_ek)
cov_tendH = st2.cov(tendsst, tendH)
#cov_Qek_f = st2.cov(tendsst,Q_ek_f)

#cov_Rnet = st2.cov(tendsst,Q_net_surf)
#cov_thf = st2.cov(tendsst,-thf)
#
#cov_RnetRnet = st2.cov(Q_net_surf,Q_net_surf)
#cov_thfRnet = st2.cov(-thf,Q_net_surf)
#cov_QrRnet = st2.cov(Qr, Q_net_surf)
#
#cov_thfthf = st2.cov(-thf,-thf)
#cov_Qrthf = st2.cov(Qr,-thf)
#
#cov_QrQr = st2.cov(Qr,Qr)


#cov_QrRnet = st2.cov(Qr,Q_net_surf)
#cov_QsRnet = st2.cov(Q_s,Q_net_surf)
#cov_QrQs = st2.cov(Qr,Q_s)

#cov_ssttend = st2.cov(tendsst,tendsst)





# Compute lagged sst autocorrelations
r1corrs = st2.cor(sst,sst,lagx=1)
r2corrs = st2.cor(sst,sst,lagx=2)



# Scaling factor (to convert from units of W*K/(s*m^2) to K^2)
fac = (2*dt**2/(Cbar*(1-r2corrs)))

#fac=(dt**2)/(2*Cbar*(1-r1corrs))

G = fac/Cbar

# Compute observed SST variance
T_var = sst.var(dim='time')

H_var = (Cbar*tendsst).var(dim='time')

# Compute contributions to SST variance
T_var_Qr = fac*cov_Qr
T_var_Qs = fac*cov_Qs
#T_var_thf = fac*cov_thf
T_var_Qek = fac*cov_Qek

T_var_tendH = fac*cov_tendH
#T_var_Qek_f = fac*cov_Qek_f
#T_var_Rnet = fac*cov_Rnet
#T_var_sum = T_var_Qr + T_var_Qs + T_var_Qek

#T_var_sum = T_var_Qr + T_var_Qs

T_var_sum = T_var_Qr + T_var_Qs + T_var_tendH

#T_var_RnetRnet = G*cov_RnetRnet
#T_var_thfRnet = G*cov_thfRnet
#T_var_QrRnet = G*cov_QrRnet
#
#T_var_thfthf = G*cov_thfthf
#T_var_Qrthf = G*cov_Qrthf
#T_var_QrQr = G*cov_QrQr


# Approximation of SST variance by the covariance of the SST tendency
#T_var_test = (fac*Cbar)*cov_ssttend


# Plotting
bnds = [np.round(lonbounds[0]-359), np.round(lonbounds[1]-361), latbounds[0], latbounds[1]]

cent = (bnds[0]+bnds[1])/2.
prj = cart.crs.PlateCarree(central_longitude=cent)

bnds[0] = bnds[0] + 1
bnds[2] = bnds[2] + 2

pardiff = 30.
merdiff = 60.
if lonbounds[1] - lonbounds[0] <= 180:
    merdiff = 15.
if lonbounds[1] - lonbounds[0] <= 60:
    merdiff = 5.
if np.abs(latbounds[1]-latbounds[0]) <= 30:
    pardiff=5.
par = np.arange(-90.,91.,pardiff)
mer = np.arange(-180.,180.+merdiff,merdiff)
x, y = np.meshgrid(lons, lats)


orient = 'horizontal'
if np.abs(latbounds[1] - latbounds[0]) > np.abs(lonbounds[1] - lonbounds[0]):
    orient = 'vertical'
    
    
sstvmax = 5.0
vmin=-1.0
vmax=1.0
if lowpass:
    sstvmax = 1.0
    vmin=-1.0
    vmax=1.0
#    

    
#sstcmap = cmocean.cm.plasma
#sstcmap = plt.cm.cubehelix_r
#fieldcmap = cmocean.cm.balance
fieldcmap = plt.cm.RdBu_r

sstcmap = cc.cm.CET_L17
#fieldcmap = cc.cm.CET_D1A
#fieldcmap = cc.cm.coolwarm

#varmin=10**2
#varmax=10**4

varmin=10**2
varmax=10**4


latbounds = [30,50]
lonbounds = [310,345]

lonbounds_box = [305,335]
latbounds_box = [38,50]

x1 = lonbounds_box[0]
x2 = lonbounds_box[1]
y1 = latbounds_box[0]
y2 = latbounds_box[1]




#if Qekplot:
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)

#fieldcmap = plot.Colormap('ColdHot')
cbfrac=0.11

mapper = Mapper()
mapper(T_var, bnds=bnds, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0, vmax=sstvmax)
plt.savefig(fout + '{:s}_Tvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(H_var, bnds=bnds, log=True, title=r'Mixed Layer Heat Storage Variance', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Hvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Q_s_var, bnds=bnds, log=True, title=r'$Q_s$ Variance', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qsvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Qr_var, bnds=bnds, log=True, title=r'$Q_o$ Variance', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qrvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

lvmin=-varmax
lvmax=-varmin
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=lvmin, vmax=lvmax)

mapper = Mapper()
mapper(covQsQo, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$Q_s, Q_o$ Covariance', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17_r, vmin=lvmin, vmax=lvmax)
plt.savefig(fout + '{:s}_covQsQo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

sstlognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0, vmax=sstvmax)
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)

mapper = Mapper()
m,ax=mapper(T_var, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(h_var, bnds=bnds, log=True, title=r'$\sigma_h^2$', units=r'm$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=10e5)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_hvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(T_var_sum,  log=True, bnds=bnds, title='SUM', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvarSUM__{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(T_var_Qs, logscale=False, bnds=bnds, title='$Q_s$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvarQs_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(T_var_Qr, logscale=False, bnds=bnds, title='$Q_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvarQo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


m,ax=mapper(T_var_tendH, logscale=False, bnds=bnds, title='Mixed Layer Depth Tendency', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvartendH_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#T_var_Qr_SAVE = T_var_Qr

#T_var_Qr_error = T_var_Qr - T_var_Qr_SAVE 
#
#m,ax=mapper(T_var_Qr_error, norm=lognorm, bnds=bnds, title='$Q_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
#if drawbox:
#    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
#plt.savefig(fout + '{:s}_TvarQo_error_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper(T_var_Rnet, bnds=bnds, title='$R_{net}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
#plt.savefig(fout + '{:s}_TvarRnet_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper(T_var_thf, bnds=bnds, title='$-THF$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
#plt.savefig(fout + '{:s}_TvarRnet_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

mapper(T_var_Qek, norm=lognorm, bnds=bnds, title='$Q_{ek}$', cbfrac=cbfrac, units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
plt.savefig(fout + '{:s}_TvarQek_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#mapper(T_var_Qek_v, bnds=bnds, title='$Q_{ek}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
#plt.savefig(fout + '{:s}_TvarQek_v_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#
#mapper(T_var_Qek_f, bnds=bnds, title='$Q_{ek}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
#plt.savefig(fout + '{:s}_TvarQek_f_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()


## Covariance terms
#mapper(T_var_RnetRnet, bnds=bnds, title='$R_{net},R_{net}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-2, vmax=2)
#plt.savefig(fout + '{:s}_TvarRnetRnet_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(T_var_QrRnet, bnds=bnds, title='$Q_{o},R_{net}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-2, vmax=2)
#plt.savefig(fout + '{:s}_TvarQoRnet_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(T_var_thfRnet, bnds=bnds, title='$-THF,R_{net}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-2, vmax=2)
#plt.savefig(fout + '{:s}_TvarthfRnet_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(T_var_thfthf, bnds=bnds, title='$-THF,-THF$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-2, vmax=2)
#plt.savefig(fout + '{:s}_Tvarthfthf_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(T_var_Qrthf, bnds=bnds, title='$Q_{o},-THF$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-2, vmax=2)
#plt.savefig(fout + '{:s}_TvarQrthf_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(T_var_QrQr, bnds=bnds, title='$Q_{o},Q_{o}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-2, vmax=2)
#plt.savefig(fout + '{:s}_TvarQrQr_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper(taux_clim, bnds=bnds, title='Zonal Wind Stress', units=r'N/m$^{2}$', cmap=fieldcmap, vmin=-0.2, vmax=0.2)
#plt.savefig(fout + '{:s}_meantaux_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(tauy_clim, bnds=bnds, title='Meridional Wind Stress', units=r'N/m$^{2}$', cmap=fieldcmap, vmin=-0.2, vmax=0.2)
#plt.savefig(fout + '{:s}_meantauy_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(u_ek_clim, bnds=bnds, title='$u_{ek}$', units=r'm/s', cmap=fieldcmap, vmin=-0.2, vmax=0.2)
#plt.savefig(fout + '{:s}_meanuek_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper(w_ek_mean, bnds=bnds, title='Mean $w_{ek}$', units=r'm/s', cmap=fieldcmap, vmin=-1e-5, vmax=1e-5)
#plt.savefig(fout + '{:s}_meanwek_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()


#mapper(v_ek_clim, bnds=bnds, title='$v_{ek}$', units=r'm/s', cmap=fieldcmap, vmin=-0.2, vmax=0.2)
#plt.savefig(fout + '{:s}_meanvek_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
##
mapper(Q_ek_mean, logscale=False, bnds=bnds, title='$Q_{ek}$', units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-200, vmax=200)
plt.savefig(fout + '{:s}_meanQek_h_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

##
mapper(sst_mean, logscale=False, bnds=bnds, title='$Q_{ek}$', units=r'W/m$^{2}$', cmap=sstcmap, vmin=-10, vmax=30)
plt.savefig(fout + '{:s}_sst_mean_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#sst_interp_mean = sst_interp.mean(dim='time')

#mapper(sst_interp_mean, bnds=bnds, title='$Q_{ek}$', units=r'W/m$^{2}$', cmap=sstcmap, vmin=-10, vmax=30)
#plt.savefig(fout + '{:s}_sstoa_mean_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper(Q_ek_v_mean, bnds=bnds, title='Mean $Q_{ek}$', units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-200, vmax=200)
#plt.savefig(fout + '{:s}_meanQek_v_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(Q_ek_f_mean, bnds=bnds, title='Mean $Q_{ek}$', units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-200, vmax=200)
#plt.savefig(fout + '{:s}_meanQek_f_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

fracvmin=-3.0
fracvmax=3.0
if not(lowpass):
    fracvmin = -1.0
    fracvmax = 1.0

logfrac = np.log(T_var_Qs/T_var_Qr) 
    
mapper(T_var_Qs/T_var_sum, logscale=False, bnds=bnds, title='$Q_s$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=fracvmin, vmax=fracvmax)
plt.savefig(fout + '{:s}_TvarQsFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(T_var_Qr/T_var_sum, logscale=False, bnds=bnds, title='$Q_o$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=fracvmin, vmax=fracvmax)
plt.savefig(fout + '{:s}_TvarQoFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

logfracmin=-3.0
logfracmax=3.0

#mapper(T_var_Qs/T_var_Qr, bnds=bnds, title='$(\sigma_{Q_s} / \sigma_{Q_o})$', units=r'', cmap=fieldcmap, vmin=logfracmin, vmax=logfracmax)
#plt.savefig(fout + '{:s}_TvarQsQoFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

logcmap = plt.cm.Spectral_r

mapper(logfrac, logscale=False, bnds=bnds, title='$log(\sigma^2_{T,Q_s} / \sigma^2_{T,Q_o})$', cbfrac=cbfrac, units=r'', cmap=logcmap, vmin=logfracmin, vmax=logfracmax)
plt.savefig(fout + '{:s}_TvarQsQoLOGFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(-(Tmxlfrac-1), logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} / \sigma^2_{T,mxl}$ - 1', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
plt.savefig(fout + '{:s}_TmxlTsratio_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(-(T_var/sst_var-1), logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} / \sigma^2_{T,mxl}$ - 1', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
plt.savefig(fout + '{:s}_TmxlTsratio_TEST_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(sst_var - T_var, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} - \sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
plt.savefig(fout + '{:s}_TmxlTsDIFF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(Qr_var, bnds=bnds, logscale=False, log=True, title='$Q_o$ Variance', cbfrac=cbfrac, units=r'W$^{2}$ m$^{-4}$', cmap=sstcmap, vmin=10**2, vmax=10**4)
plt.savefig(fout + '{:s}_Qovar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#mapper(sst_var_ECCO, bnds=bnds, logscale=False, log=True, title='$\sigma^2_T$', cbfrac=cbfrac, units=r'W$^{2}$ m$^{-4}$', cmap=sstcmap, vmin=0.01, vmax=sstvmax)
#plt.savefig(fout + '{:s}_sstECCOvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(Tmxl_var_ECCO, bnds=bnds, logscale=False, log=True, title='$\sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'W$^{2}$ m$^{-4}$', cmap=sstcmap, vmin=0.01, vmax=sstvmax)
#plt.savefig(fout + '{:s}_TmxlECCOvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper(sst_var_ECCO - Tmxl_var_ECCO, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} - \sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
#plt.savefig(fout + '{:s}_TmxlTsDIFFECCO_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper((np.sqrt(sst_var_ECCO/Tmxl_var_ECCO))-1, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} / \sigma^2_{T,mxl}$ - 1', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
#plt.savefig(fout + '{:s}_TmxlTsratioECCO_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#


#mapper(mxldepth_clim, bnds=bnds, title='Mixed Layer Depth', cbfrac=cbfrac, units=r'm', cmap=cmocean.cm.deep, vmin=20, vmax=300)
#plt.savefig(fout + '{:s}_mxldepth_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper(T_var_Qek/T_var_sum, bnds=bnds, title='$Q_{ek}$', units=r'', cmap=fieldcmap, vmin=fracvmin, vmax=fracvmax)
#plt.savefig(fout + '{:s}_TvarQekFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()













