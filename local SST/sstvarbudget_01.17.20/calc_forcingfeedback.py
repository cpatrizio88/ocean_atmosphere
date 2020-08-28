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
import scipy.signal as sig
import numpy.fft as fft
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
fsstM2 =  xr.open_dataset(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
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
dataname = 'OAFlux'
#dataname = 'ERA5'
#dataname = 'ECCO'

#ECCO
fin = '/Volumes/GoogleDrive/My Drive/data_drive/ECCO/'
#ft= xr.open_dataset(fin + 'ECCO_theta_monthly1992to2015.nc')
fh = xr.open_dataset(fin + 'ECCO_mxldepth_interp_1992to2015.nc')
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
# fin = '/Volumes/GoogleDrive/My Drive/data_drive/ERA5/'
# fsst = xr.open_dataset(fin + 'ERA5_sst_monthly1979to2019.nc')
# fthf = xr.open_dataset(fin + 'ERA5_thf_monthly1979to2019.nc')
# frad = xr.open_dataset(fin + 'ERA5_rad_monthly1979to2019.nc')

#ERSST
#fsst = xr.open_dataset('/Users/cpatrizio/data/ERSST/sst.mnmean.nc')

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

hmean = h.mean(dim='time')
hbar=hmean
#h = fh.dbss_obml
#h_anom, h_clim = st.anom(h)
#h = h.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#hbar = h_clim.max(dim='month')
#hbar = h.mean(dim='time')
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
# latbounds=[0,65]
# lonbounds=[270,360]

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
# sst = fsst.sst
# lhf = fthf.mslhf
# shf = fthf.msshf
# LW_net_surf = frad.msnlwrf
# SW_net_surf = frad.msnswrf

#MERRA-2
# sst = fsstM2.TSKINWTR
# lhf = fthf.EFLUX
# shf = fthf.HFLUX
# LW_net_surf = radfile.LWGNT
# SW_net_surf = radfile.SWGNT


#OAFlux
sst = fsstoa.tmpsf
lhf = fthf.lhtfl
shf = fthf.shtfl
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT

#ECCO
#sst = fsst.Ts
#sst = fTmxl.Tmxl
#sst_interp = sst
#sst_interp= theta.isel(k=0)
#sst = theta_mxl

#ERSST
#sst = fsst.sst


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
    sst = sst.assign_coords(lon=(sst.lon % 360)).roll(lon=((len(lons) // 2)-1))
    
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
     
# taux = ftau.TAUXWTR
# tauy = ftau.TAUYWTR
# taux = taux.assign_coords(lon=(taux.lon % 360)).roll(lon=((taux.shape[2] // 2)-1))
# tauy = tauy.assign_coords(lon=(tauy.lon % 360)).roll(lon=((tauy.shape[2] // 2)-1))
# taux = taux.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
# tauy = tauy.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

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
# taux = taux[:Q_net_surf.shape[0],:,:]
# tauy = tauy[:Q_net_surf.shape[0],:,:]
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
#h.time.values = dates
Q_net_surf.time.values = dates
#SW_net_surf.time.values = dates
#ssh.time.values = dates
# taux.time.values = dates
# tauy.time.values = dates
if dataname == 'ECCO':
#   sst_interp = sst_interp.sel(time=slice('1992-02-01','2015-12-01'))
    sst = sst.sel(time=slice('1992-02-01','2015-12-01'))
    thf = thf.sel(time=slice('1992-02-01','2015-12-01'))
    Q_net_surf = Q_net_surf.sel(time=slice('1992-02-01','2015-12-01'))
    # taux = taux.sel(time=slice('1992-02-01','2015-12-01'))
    # tauy = tauy.sel(time=slice('1992-02-01','2015-12-01'))
    #h = h.sel(time=slice('1992-02-01','2015-12-01'))

# sst = sst.sel(time=slice('1992-01-01','2015-01-01'))
# thf = thf.sel(time=slice('1992-01-01','2015-01-01'))
# Q_net_surf = Q_net_surf.sel(time=slice('1992-01-01','2015-01-01'))
# h = h.sel(time=slice('1992-01-01','2015-01-01'))
# taux = taux.sel(time=slice('1992-01-01','2015-01-01'))
# tauy = tauy.sel(time=slice('1992-01-01','2015-01-01'))

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
lowpass = False
highpass = False
anom_flag = True
timetend=False
detr=True
rENSO=True
corr=False
lterm=True
drawmaps=False
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

regridder = xe.Regridder(hmean, ds_out, 'bilinear', reuse_weights=True)
hmean= regridder(hmean)

regridder = xe.Regridder(Q_net_surf, ds_out, 'bilinear', reuse_weights=True)
Q_net_surf  = regridder(Q_net_surf)

# regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
# taux  = regridder(taux)
# #
# regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
# tauy  = regridder(tauy)

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

# taux_clim = taux.mean(dim='time')
# tauy_clim = tauy.mean(dim='time')

sst_mean = sst.mean(dim='time')

lats = sst.lat
lons = sst.lon


# Scale the SST anomalies by the ratio of the mixed layer temperature variance to the SST in ECCO 
# This is to account for lower variability when averaged over the entire mixed layer
#sst = sst*Tmxlfrac


rho = 1000
c_p = 3850
dt = 30*3600*24
#C = rho*c_p*h

#C_anom, C_clim = st.anom(C)

#Cbar = C_clim.mean(dim='month')

Cbar = rho*c_p*hbar

#Cbar = C.mean(dim='time')

#if dataname == 'ECCO':
#    Cbar = rho*c_p*delz_sum



h_anom, h_clim = st.anom(h)

h_clim_std = h_clim.std(dim='month')
h_bar = h_clim.mean(dim='month')


# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    #Q_ek,Q_ek_clim= st.anom(Q_ek)
    sst_ECCO,sst_ECCO_clim= st.anom(sst_ECCO)
    Tmxl_ECCO,Tmxl_ECCO_clim= st.anom(Tmxl_ECCO)



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
 


# Mask zero values (continents) 
ocean_points1 = ~(sst==0)
ocean_points2 = ~(xr.ufuncs.isnan(sst))
ocean_points = xr.ufuncs.logical_or(ocean_points1, ocean_points2)
sst = sst.where(ocean_points)
thf = thf.where(ocean_points)
# Q_net_surf = Q_net_surf.where(ocean_points)
# Q_ek = Q_ek.where(ocean_points)
#Q_ek = Q_ek.where(~(Q_ek==0))
#Q_ek = Q_ek.where(np.abs(lats) > 0)
# Compute SST tendency


Tmxl_var_ECCO = Tmxl_ECCO.var(dim='time')
sst_var_ECCO = sst_ECCO.var(dim='time')

# # Scale the SST tendency by the ratio of MXL temp tendency / SST tendency from ECCO (as an esimate for a more physical result)
Tmxlfrac = Tmxl_var_ECCO/sst_var_ECCO
sst = sst*np.sqrt(Tmxlfrac)


                
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

#Qr = Cbar*tendsst - (-thf + Q_net_surf) - Q_ek

Q_r = Cbar*tendsst -(-thf + Q_net_surf)
Q_r = Q_r.transpose('time','lat','lon')

Q_r_mean = Q_r.mean(dim='time')


#Q_s = -thf + Q_net_surf + Q_ek

Q_s = -thf + Q_net_surf 

Q_tot = Q_r + Q_s

nt = sst.shape[0]
#timeslice = slice(0,nt)
#timeslice = slice(int(Tn),nt-int(Tn))

# Q_s = Q_s.isel(time=timeslice)
# Q_r = Q_r.isel(time=timeslice)
# tendsst = tendsst.isel(time=timeslice)
# sst = sst.isel(time=timeslice)

order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 4.*12.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)


if lowpass:
    sst_lp = st.butter_lowpass_filter_xr(sst, cutoff, fs, order)
    r2corrs = st2.cor(sst_lp,sst_lp,lagx=2)
else:
    r2corrs = st2.cor(sst,sst,lagx=2)
    
#Niño 3.4 (5N-5S, 170W-120W

order = 5
fs = 1     # sample rate, (cycles per month)
Tn_enso = 6.
cutoff_enso = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
enso = st2.spatial_ave_xr(sst.sel(lon=slice(190,240)), lats=lats.sel(lat=slice(-5,5)))

enso = st.butter_lowpass_filter_xr(enso,cutoff_enso,fs,order) 

if not(lowpass):
     Tn = 0.
    

if rENSO:
   sste = st2.regressout_x(enso, sst)
   Q_se = st2.regressout_x(enso, Q_s)
   Q_re = st2.regressout_x(enso, Q_r)
   Q_tote = st2.regressout_x(enso, Q_tot)
   


# Scaling factor (to convert from units of W*K/(s*m^2) to K^2)
#G = (2*dt**2/(Cbar**2*(1-r2corrs)))



    
   
#lambdaQ_tot = st.cov(sste, Q_tote, lagx=-1)/st.cov(sste,sste, lagx=-1)

lambdaQ_totlag0 = st.cov(sste, Q_tote, lagx=0)/st.cov(sste,sste, lagx=0)
lambdaQ_totlag1 = st.cov(sste, Q_tote, lagx=-1)/st.cov(sste,sste, lagx=-1)
lambdaQ_totlag2 = st.cov(sste, Q_tote, lagx=-2)/st.cov(sste,sste, lagx=-2)
lambdaQ_totlag3 = st.cov(sste, Q_tote, lagx=-3)/st.cov(sste,sste, lagx=-3)

lambdaQ_tot = (lambdaQ_totlag1 + lambdaQ_totlag2 + lambdaQ_totlag3)/3.
#lambdaQ_tot = lambdaQ_totlag0


#lambdaQ_tot = lambdaQ_totlag0

lambdaQ_slag0 = st.cov(sste, Q_se, lagx=0)/st.cov(sste,sste, lagx=0)
lambdaQ_slag1 = st.cov(sste, Q_se, lagx=-1)/st.cov(sste,sste, lagx=-1)
lambdaQ_slag2 = st.cov(sste, Q_se, lagx=-2)/st.cov(sste,sste, lagx=-2)
lambdaQ_slag3 = st.cov(sste, Q_se, lagx=-3)/st.cov(sste,sste, lagx=-3)

lambdaQ_rlag0 = st.cov(sste, Q_re, lagx=0)/st.cov(sste,sste, lagx=0)
lambdaQ_rlag1 = st.cov(sste, Q_re, lagx=-1)/st.cov(sste,sste, lagx=-1)
lambdaQ_rlag2 = st.cov(sste, Q_re, lagx=-2)/st.cov(sste,sste, lagx=-2)
lambdaQ_rlag3 = st.cov(sste, Q_re, lagx=-3)/st.cov(sste,sste, lagx=-3)

lambdaQ_s = (lambdaQ_slag1 + lambdaQ_slag2 + lambdaQ_slag3)/3.
lambdaQ_r = (lambdaQ_rlag1 + lambdaQ_rlag2 + lambdaQ_rlag3)/3.

#lambdaQ_s = lambdaQ_slag0
#lambdaQ_r = lambdaQ_rlag0

gamma = ((dt*lambdaQ_tot)/Cbar)**2

Gstar = (2*dt**2)/(Cbar**2*(1-r2corrs+2*gamma))

G = (2*dt**2)/(Cbar**2*(1-r2corrs))


# lambdaQ_s = (lambdaQ_s_lag1)
# lambdaQ_r = (lambdaQ_r_lag1)

lambdaQ_totT = lambdaQ_tot*sst
lambdaQ_sT = lambdaQ_s*sst
lambdaQ_rT = lambdaQ_r*sst

Q_totstar = Q_tot - lambdaQ_totT
Q_sstar = Q_s - lambdaQ_sT
Q_rstar = Q_r - lambdaQ_rT


lambdaQ_tot_var = (lambdaQ_totT).var(dim='time')
lambdaQ_s_var = (lambdaQ_sT).var(dim='time')
lambdaQ_r_var = (lambdaQ_rT).var(dim='time')

#Q_totstar_var = Q_totstar.var(dim='time')
#Q_sstar_var = Q_sstar.var(dim='time')
#Q_rstar_var = Q_rstar.var(dim='time')

T_var = sst.var(dim='time')
Qtot_var = (Q_totstar).var(dim='time')
Q_s_var = Q_sstar.var(dim='time')
Q_r_var = Q_rstar.var(dim='time')
covQsQr = st.cov(Q_sstar,Q_rstar)

covQsQr = 0.5*(Qtot_var - Q_s_var - Q_r_var)


lats = sst.lat
lons = sst.lon
nlat = len(lats)
nlon = len(lons)


#f_temp, Tpower_temp= sig.periodogram(sst,axis=0,return_onesided=True)
#f_temp, Qtotpower_temp = sig.periodogram(Q_s + Q_r,axis=0,return_onesided=True)

# T_A = np.fft.fft(sst,axis=0)
# Qtot_A = np.fft.fft(Q_s + Q_r,axis=0)

#norm ortho means normalize by 1/sqrt(nt)

window = np.hanning(nt)

# sst = window[:,None,None]*sst
# tendsst =  window[:,None,None]*tendsst
# Qtot =  window[:,None,None]*(Q_s+Q_r)
# Q_s =  window[:,None,None]*Q_s
# Q_r = window[:,None,None]*Q_r

T_A = fft.fft(sst,axis=0, norm='ortho')
tendT_A = fft.fft(tendsst,axis=0, norm='ortho')
Qtot_A = fft.fft(Q_totstar,axis=0, norm='ortho')
Qs_A = fft.fft(Q_sstar,axis=0,norm='ortho')
Qr_A = fft.fft(Q_rstar,axis=0,norm='ortho')
freqs_csd, QsQr_A = sig.csd(Q_sstar,Q_rstar,axis=0,scaling='density',detrend=False,nfft=nt)


# N = nt
# if N%2:
#   derivative_operator = np.concatenate((np.arange(0,N//2,1),[0],np.arange(-N//2+1,0,1)))*1j
# else:
#   derivative_operator = np.concatenate((np.arange(0,N/2,1),np.arange(-N/2+1,0,1)))*1j

# #T_fft = fft.fft(sst,axis=0)
  

# # work on this...  
# dT_A = np.fft.ifft(derivative_operator[:,None,None]*T_A)


# halfnt = int(nt/2)

# what = 1j*np.zeros(nt)
# what[0:halfnt] = 1j*np.arange(0,halfnt, 1)
# what[halfnt+1:] = 1j*np.arange(-halfnt + 1, 1, 1)
# what = what*T_fft

# tendT_fft = np.fft.ifft(what,axis=0)

QsQr_A = QsQr_A[:-1,:]

#QsQr_A = QsQr_A/(np.sqrt(nt))

freqs =  fft.fftfreq(nt)
freqs = freqs[:int(nt/2)]

tendTpower_temp = np.abs(tendT_A)**2
#dTpower_temp = np.abs(dT_A)**2
Tpower_temp = np.abs(T_A)**2
Qtotpower_temp = np.abs(Qtot_A)**2
Qspower_temp = np.abs(Qs_A)**2
Qrpower_temp = np.abs(Qr_A)**2
#Why do I need 0.5 scaling?
QsQrpower_temp = np.real(QsQr_A)
#QsQrpower_temp = np.abs(QsQr_A)**2

#Double the power to account for negative frequencies (by symmetry)

tendTpower_temp = 2*tendTpower_temp[:int(nt/2)]
#dTpower_temp = 2*dTpower_temp[:int(nt/2)]
Tpower_temp = 2*Tpower_temp[:int(nt/2)]
Qtotpower_temp = 2*Qtotpower_temp[:int(nt/2)]
Qspower_temp = 2*Qspower_temp[:int(nt/2)]
Qrpower_temp = 2*Qrpower_temp[:int(nt/2)]

#scale by 1/nt again to make sure the sum of the power equals the variance (not sure why I need this...)
tendTpower_temp = tendTpower_temp/nt
#dTpower_temp = dTpower_temp/nt
Tpower_temp = Tpower_temp/nt
Qtotpower_temp = Qtotpower_temp/nt
Qspower_temp = Qspower_temp/nt
Qrpower_temp = Qrpower_temp/nt


nfreq = len(freqs)

f = xr.DataArray(np.nan*np.zeros([nfreq]), coords={'freq': freqs},dims=['freq'])
f.values = freqs

tendTpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

#dTpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
#                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Tpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qtotpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qspower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qrpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

QsQrpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

tendTpower.values = tendTpower_temp
#dTpower.values = dTpower_temp
Tpower.values = Tpower_temp
Qtotpower.values = Qtotpower_temp
Qspower.values = Qspower_temp
Qrpower.values = Qrpower_temp
QsQrpower.values = QsQrpower_temp

#QsQrpower = QsQrpower*f.diff('freq')

monthtosec=3600*24*30

# if lowpass:
#     cutoffi = np.where(f < cutoff)[0][-1]
# else:
#     cutoffi = -1
    
if highpass:
    cutoffi = np.where(f >= cutoff)[0][0]
    f = f[cutoffi:]
    Tpower = Tpower[cutoffi:,:]
    Qtotpower = Qtotpower[cutoffi:,:]
    Qspower = Qspower[cutoffi:,:]
    Qrpower = Qrpower[cutoffi:,:]
elif lowpass:
    cutoffi = np.where(f <= cutoff)[0][-1]
    f = f[3:cutoffi]
    Tpower = Tpower[3:cutoffi,:]
    Qtotpower = Qtotpower[3:cutoffi,:]
    Qspower = Qspower[3:cutoffi,:]
    Qrpower = Qrpower[3:cutoffi,:]
else:
    #filter out periodic signals that are greater than 15 yrs when looking at 'raw' data (not really physical)
    f = f[3:]
    Tpower = Tpower[3:]
    Qtotpower = Qtotpower[3:]
    Qspower = Qspower[3:]
    Qrpower = Qrpower[3:]

# f = f[3:cutoffi]
# Tpower = Tpower[3:cutoffi,:]
# Qtotpower = Qtotpower[3:cutoffi,:]
# Qspower = Qspower[3:cutoffi,:]
# Qrpower = Qrpower[3:cutoffi,:]


freq_sec = f/monthtosec

fac1 = 1/(((2  - 2*np.cos(2*np.pi*f))*Cbar**2)/(dt**2))

fac2 = 1/((lambdaQ_tot)**2+(2*np.pi*Cbar*freq_sec)**2)

fac3 = 1/((Cbar**2*np.sin((2*np.pi*f)))/(dt**2))

#fac = (fac1+fac2)/2.

Gf = Tpower/Qtotpower

fac = Gf
#fac = fac2

#calculate cross spectrum as a residual!
QsQrpower = 0.5*(Qtotpower - Qspower - Qrpower)

#Tpower_test = (Qtotpower*dt**2)/(conv_fac*Cbar**2)

Tpower_sum = fac*(Qspower + Qrpower + 2*QsQrpower)
Tpower_Qtot = fac*(Qtotpower)
#Tpower_test3 = (Qtotpower)/(freq_sec*Cbar)

Tpower_Qronly = fac*(Qrpower)
Tpower_Qsonly = fac*(Qspower)
Tpower_Qs = fac*(Qspower + QsQrpower)
Tpower_Qr = fac*(Qrpower + QsQrpower)
Tpower_QsQr = fac*(QsQrpower)


#Tvar = T_var
Qtot_var_Tpow = Qtotpower.sum('freq')
Qs_var_Tpow = Qspower.sum('freq')
Qr_var_Tpow = Qrpower.sum('freq')
QsQr_cov_Tpow = QsQrpower.sum('freq')

Tpower_plot = Tpower.sum('freq')
Tpower_Qtot_plot = Tpower_Qtot.sum('freq')
Tpower_Qs_plot = Tpower_Qs.sum('freq')
Tpower_Qr_plot = Tpower_Qr.sum('freq')
Tpower_Qsonly_plot = Tpower_Qsonly.sum('freq')
Tpower_Qronly_plot = Tpower_Qronly.sum('freq')
Tpower_QsQr_plot = Tpower_QsQr.sum('freq')
Tpower_sum_plot = Tpower_Qs_plot + Tpower_Qr_plot



Tvar_Qtot = Gstar*(Qtot_var_Tpow)
Tvar_Qs = Gstar*(Qs_var_Tpow + QsQr_cov_Tpow)
Tvar_Qr = Gstar*(Qr_var_Tpow + QsQr_cov_Tpow)
Tvar_QsQr = Gstar*(QsQr_cov_Tpow)

Tvar_sum = Tvar_Qs + Tvar_Qr

latp= 45
lonp= 190

ratio = Tpower.sel(lat=latp,lon=lonp)/Tpower_Qtot.sel(lat=latp, lon=lonp)

print('ratio mean', ratio.mean())


fig,ax=plt.subplots(nrows=4,ncols=1,sharex=True)
ax[0].plot(f,Tpower.sel(lat=latp,lon=lonp))
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xlim(0.006,0.5)
ax[0].set_title('$|T|^2$')
ax[0].set_ylabel('Power (K$^2$)')

ax[1].plot(f,Tpower_Qtot.sel(lat=latp, lon=lonp))
ax[1].set_yscale('log')
ax[1].set_title('$|T|^2$ from $Q_{tot}$')
#ax[1].set_xlabel('Frequency (month$^{-1}$)')
ax[1].set_ylabel('Power (K$^2$)')
#ax[2].plot(f,Tpower_sum.sel(lat=latp, lon=lonp))
#ax[2].set_yscale('log')
#ax[2].set_title('$|T|^2$ from sum')
ax[2].set_title('G(f)')
#ax[2].set_ylim(0,0.1)
ax[2].set_yscale('log')
ax[2].plot(f,Gf.sel(lat=latp, lon=lonp))
ax[3].set_title('fac')
#ax[3].set_ylim(0,0.1)
ax[3].set_yscale('log')
ax[3].plot(f,fac.sel(lat=latp, lon=lonp))
ax[3].set_xlabel('Frequency (month$^{-1}$)')
plt.savefig(fout + 'T_powerspectrum.pdf')
#ax[3].set_ylim(0,15)
#ax[4].set_title('1/fac')
#ax[4].plot(f,1/fac.sel(lat=latp, lon=lonp))



fig,ax=plt.subplots(figsize=(12,16),nrows=5,ncols=1,sharex=True)
ax[0].set_xscale('log')
ax[0].set_xlim(0.006,0.5)
ax[0].plot(f,Qtotpower.sel(lat=latp,lon=lonp))
ax[0].set_title('$|Q^{*}_{tot}|^2$')
ax[0].axhline(0,color='k', linewidth=1)
ax[1].plot(f,Qspower.sel(lat=latp,lon=lonp))
ax[1].set_title('$|Q^{*}_{s}|^2$')
ax[1].axhline(0,color='k', linewidth=1)
ax[2].set_title('$|Q^{*}_{o}|^2$')
ax[2].plot(f,Qrpower.sel(lat=latp,lon=lonp))
ax[2].axhline(0,color='k', linewidth=1)
ax[3].set_title('$|Q^{*}_{s} Q^{*}_{o}|$')
ax[3].plot(f,QsQrpower.sel(lat=latp,lon=lonp))
ax[3].axhline(0,color='k', linewidth=1)
ax[4].plot(f,(Qrpower + QsQrpower).sel(lat=latp,lon=lonp))
ax[4].axhline(0,color='k', linewidth=1)
ax[4].set_title('$|Q^{*}_o|^2 + |Q^{*}_{s} Q^{*}_{o}|$')
#ax[4].set_title('sum')
#ax[4].plot(f,(Qspower + Qrpower + 2*QsQrpower).sel(lat=latp,lon=lonp))
#ax[4].axhline(0,color='k', linewidth=1)


ax[0].set_yscale('symlog')
ax[1].set_yscale('symlog')
ax[2].set_yscale('symlog')
ax[3].set_yscale('symlog')
ax[4].set_yscale('symlog')

ax[0].set_yticks([-10,-1,1,10])
ax[1].set_yticks([-10,-1,1,10])
ax[2].set_yticks([-10,-1,1,10])
ax[3].set_yticks([-10,-1,1,10])
ax[4].set_yticks([-10,-1,1,10])

ax[0].set_ylim(-20,50)
ax[1].set_ylim(-20,50)
ax[2].set_ylim(-20,50)
ax[3].set_ylim(-20,50)
ax[4].set_ylim(-20,50)
ax[4].set_xlabel(r'Frequency (month$^{-1}$)')
plt.savefig(fout + 'QsQr_powerspectra_forcing.pdf')

delf = f.diff('freq')

T_var_actual = T_var.sel(lat=latp,lon=lonp)
Qtot_var_actual = Qtot_var.sel(lat=latp,lon=lonp)
Qs_var_actual = Q_s_var.sel(lat=latp,lon=lonp)
Qr_var_actual = Q_r_var.sel(lat=latp,lon=lonp)
covQsQr_actual = covQsQr.sel(lat=latp,lon=lonp)

#this is Parseval's identity: the sum of the power spectrum coefficients 
T_var_test = (Tpower.sel(lat=latp,lon=lonp)).sum('freq')
T_var_test_Qtot = Tpower_Qtot.sel(lat=latp,lon=lonp).sum('freq')
T_var_test_sum = Tpower_sum.sel(lat=latp,lon=lonp).sum('freq')
Qtot_var_test = (Qtotpower.sel(lat=latp,lon=lonp)).sum('freq')
Qs_var_test = (Qspower.sel(lat=latp,lon=lonp)).sum('freq')
Qr_var_test = (Qrpower.sel(lat=latp,lon=lonp)).sum('freq')
covQsQr_test = (QsQrpower.sel(lat=latp,lon=lonp)).sum('freq')

print('T_var', T_var_actual)
print('T_var_test', T_var_test)
print ('T_var_test_Qtot', T_var_test_Qtot)
print ('T_var_test_sum', T_var_test_sum)
#print('T_var_test2', T_var_test2)

print('Qtot_var', Qtot_var_actual)
print('Qtot_var_test', Qtot_var_test)
print('Qtot_var_test_sum', Qs_var_test + Qr_var_test + 2*covQsQr_test)
#print('T_var_test2', T_var_test2)
print('Qs_var', Qs_var_actual)
print('Qs_var_test', Qs_var_test)


print('Qr_var', Qr_var_actual)
print('Qr_var_test', Qr_var_test)

print('covQsQr_var', covQsQr_actual)
print('covQsQr_test', covQsQr_test)


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
vmin_pow = -1.0
vmax_pow = 1.0
if lowpass:
    sstvmax = 1.0
    vmin=-1.0
    vmax=1.0
    
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

# latbounds = [30,50]
# lonbounds = [310,345]

# latbounds = [48,60]
# lonbounds = [310,330]

# lonbounds_box = [305,335]
# latbounds_box = [38,50]

# lonbounds_box = [wlon,elon]
# latbounds_box = [slat,nlat]

# x1 = lonbounds_box[0]
# x2 = lonbounds_box[1]
# y1 = latbounds_box[0]
# y2 = latbounds_box[1]


#if Qekplot:
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)

#fieldcmap = plot.Colormap('ColdHot')
cbfrac=0.11

sstlognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0, vmax=sstvmax)
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)

lambdamax=100

mapper = Mapper()
m,ax=mapper(lambdaQ_tot, logscale=False, bnds=bnds, title=r'$\lambda_{tot}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_lambdaQ_tot_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
m,ax=mapper(lambdaQ_s, logscale=False, bnds=bnds, title=r'$\lambda_{s}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_lambdaQ_s_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()



mapper = Mapper()
m,ax=mapper(lambdaQ_r, logscale=False, bnds=bnds, title=r'$\lambda_{o}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_lambdaQ_o_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()




m,ax=mapper(Tpower_plot, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tpow_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
m,ax=mapper(Tpower_Qtot_plot, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tpow_Qtotstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
m,ax=mapper(Tpower_sum_plot, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tpow_sumstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(Tpower_Qs_plot, norm=lognorm, bnds=bnds, title='$\widetilde{Q}^{*}_s$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin_pow, vmax=vmax_pow)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tpow_Qsstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


m,ax=mapper(Tpower_Qr_plot, norm=lognorm, bnds=bnds, title='$\widetilde{Q}^{*}_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin_pow, vmax=vmax_pow)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tpow_Qostar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


m,ax=mapper(Tpower_QsQr_plot, norm=lognorm, bnds=bnds, title='$\overline{Q^{*}_sQ^{*}_o}$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin_pow, vmax=vmax_pow)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tpow_QsQrstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

# m,ax=mapper(Tpower_Qronly_plot, log=True, bnds=bnds, title='$\widetilde{Q}_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tpow_Qostaronly_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(Tpower_Qsonly_plot,log=True, bnds=bnds, title='$\widetilde{Q}_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tpow_Qsstaronly_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()



mapper = Mapper()
m,ax=mapper(T_var, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

# mapper = Mapper()
# m,ax=mapper(Tpower_plot - Tpower_Qtot_plot, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tvarerrorstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

mapper = Mapper()
mapper(Qtot_var, bnds=bnds, logscale=False, log=True,  title=r'$\overline{Q^{\prime 2}_{tot}}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qtotstarvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
mapper(Q_s_var, bnds=bnds, logscale=False, log=True,  title=r'$\overline{Q^{\prime 2}_s}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qsstarvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Q_r_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{\prime 2}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qrstarvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

lvmin=-varmax
lvmax=-varmin
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=lvmin, vmax=lvmax)

mapper = Mapper()
mapper(covQsQr, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$\overline{Q^\prime_sQ^\prime_o}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17_r, vmin=lvmin, vmax=lvmax)
plt.savefig(fout + '{:s}_covQsQostar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
m,ax=mapper(Tvar_Qtot, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_Qtotstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(Tvar_Qs, norm=lognorm, bnds=bnds, title='$\widetilde{Q}^{*}_s$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_Qsstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


m,ax=mapper(Tvar_Qr, norm=lognorm, bnds=bnds, title='$\widetilde{Q}^{*}_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_Qostar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(Tvar_QsQr, norm=lognorm, bnds=bnds, title='$\widetilde{Q}^{*}_s \widetilde{Q}^{*}_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_QsQrstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(Tvar_sum, log=True, bnds=bnds, title='sum', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_sum_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(Tvar_sum - T_var, log=True, bnds=bnds, title='error', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_error_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
mapper(Qs_var_Tpow, bnds=bnds, logscale=False, log=True,  title=r'$\overline{Q^{* 2}_s}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qsstarpow_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Qr_var_Tpow, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{* 2}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qrstarpow_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Qtot_var_Tpow, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{* 2}_{tot}}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qtotstarpow_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Q_s_var + Q_r_var + 2*covQsQr, bnds=bnds, logscale=False, log=True, title=r'sum',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_SUMstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#Why are the last two plots not the same??


# mapper = Mapper()
# mapper(Qtot_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{\prime 2}_{tot}}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_Qtotvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

lvmin=-varmax
lvmax=-varmin
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=lvmin, vmax=lvmax)

mapper = Mapper()
mapper(QsQr_cov_Tpow, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$\overline{Q^*_sQ^*_o}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17_r, vmin=lvmin, vmax=lvmax)
plt.savefig(fout + '{:s}_QsQostarpow_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Qs_var_Tpow + Qr_var_Tpow + 2*QsQr_cov_Tpow, bnds=bnds, logscale=False, log=True, title=r'sum',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_SUMstarpow_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

fracvmin = -1.0
fracvmax = 1.0


logfracabs = np.log(np.abs(Tvar_Qs)/np.abs(Tvar_Qr)) 

mapper(Tvar_Qs/Tvar_sum, logscale=False, bnds=bnds, title='$Q^{*}_s$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=fracvmin, vmax=fracvmax)
plt.savefig(fout + '{:s}_TvarQsFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(Tvar_Qr/Tvar_sum, logscale=False, bnds=bnds, title='$Q^{*}_o$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=fracvmin, vmax=fracvmax)
plt.savefig(fout + '{:s}_TvarQrFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

logfracmin=-3.0
logfracmax=3.0

logcmap = plt.cm.Spectral_r

mapper(logfracabs, logscale=False, bnds=bnds, title='$log(|\widetilde{Q}^{*}_s| / |\widetilde{Q}^{*}_o|)$', cbfrac=cbfrac, units=r'', cmap=logcmap, vmin=logfracmin, vmax=logfracmax)
plt.savefig(fout + '{:s}_TvarQsQoLOGFRACABS_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(Gstar, logscale=False, log=True, bnds=bnds, title='$G^{*}$', cbfrac=cbfrac, units=r'$K^2/(W m^{-2})$', cmap=logcmap, vmin=1e-5, vmax=1e-2)
plt.savefig(fout + '{:s}_Gstar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(G, logscale=False, log=True, bnds=bnds, title='$G$', cbfrac=cbfrac, units=r'$K^2/(W m^{-2})$', cmap=logcmap, vmin=1e-5, vmax=1e-2)
plt.savefig(fout + '{:s}_G_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
# mapper = Mapper()
# m,ax=mapper(Tvar_sum, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tvar_sum_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()


#lambdamax=60

# m,ax=mapper(lambdaQ_s, logscale=False, bnds=bnds, title=r'$\lambda_{Q_s}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambdaQ_s_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(lambdaQ_r, logscale=False, bnds=bnds, title=r'$\lambda_{Q_o}$',  units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambdaQ_o_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(Q_sstar_var, bnds=bnds, logscale=False, log=True,  title=r'$\overline{Q^{* \prime 2}_s}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_Qsstarvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(Q_rstar_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{* \prime 2}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_Qostarvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(Q_s_var, bnds=bnds, logscale=False, log=True,  title=r'$\overline{Q^{\prime 2}_s}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_Qsvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(Q_r_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{\prime 2}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_Qovar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(lambdaQ_s_var, bnds=bnds, logscale=False, log=True,  title=r'$\lambda^2_{Q_s} \overline{T^{\prime 2}_s}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_lambdaQsvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(lambdaQ_r_var, bnds=bnds, logscale=False, log=True, title=r'$\lambda^2_{Q_o}\overline{T^{\prime 2}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_lambdaQovar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# mapper(testQ_r_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{T^{\prime}Q^{* \prime}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# plt.savefig(fout + '{:s}_TESTQRVAR_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()












