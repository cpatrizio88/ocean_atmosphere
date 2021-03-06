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
latbounds=[0,80]
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
#latbounds = [15,60]

#NP
# lonbounds = [120,250]
# latbounds = [0,75]

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
#ps = ps/1e2
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

#lons_inters = sst_interp.lon
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
#h.time.values = dates
ps.time.values = dates
Q_net_surf.time.values = dates
#SW_net_surf.time.values = dates
#ssh.time.values = dates
taux.time.values = dates
tauy.time.values = dates
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

if dataname == 'MERRA2':

    sst = sst[tskip:ii,:,:]
    ps = ps[tskip:ii,:,:]
    Q_net_surf = Q_net_surf[tskip:ii,:,:]
    thf = thf[tskip:ii,:,:]
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
anom_flag = True
timetend=False
detr=True
rENSO=False
corr=False
lterm=True
drawmaps=False
drawbox=True
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
ps = ps.transpose('time', 'lat', 'lon')
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

#
regridder = xe.Regridder(ps, ds_out, 'bilinear', reuse_weights=True)
ps = regridder(ps)

regridder = xe.Regridder(hmean, ds_out, 'bilinear', reuse_weights=True)
hmean= regridder(hmean)

regridder = xe.Regridder(Q_net_surf, ds_out, 'bilinear', reuse_weights=True)
Q_net_surf  = regridder(Q_net_surf)

regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
taux  = regridder(taux)
# #
regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
tauy  = regridder(tauy)

regridder = xe.Regridder(sst_ECCO, ds_out, 'bilinear', reuse_weights=True)
sst_ECCO  = regridder(sst_ECCO)
#
regridder = xe.Regridder(Tmxl_ECCO, ds_out, 'bilinear', reuse_weights=True)
Tmxl_ECCO  = regridder(Tmxl_ECCO)


#ps = ps.transpose('time', 'lat', 'lon')

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


#Ekman transport is c/f*(-tauy*dSST/dx + taux*dSST/dy)
#c is specific heat capacity of seawater == 3850 J/(kg C)
# omega = 7.2921e-5
# rho = 1000
# f = 2*omega*np.sin(np.deg2rad(lats))
# r = 6.371e6
# g=9.81

# dphi = np.diff(lats)[0]*(2*np.pi/360.)
# dpsi = np.diff(lons)[0]*(2*np.pi/360.)

# dx = r*np.cos(np.deg2rad(lats))*dpsi
# dy = r*dphi

# nt = sst.shape[0]
# nlat = len(lats)
# nlon = len(lons)

# dx2D = np.zeros((nt, nlat))
# dx2D[:,:] = dx
# dx3D = np.repeat(dx2D[:,:,np.newaxis],nlon-1,axis=2)
# #
# f2D = np.zeros((nt, nlat))
# f2D[:,:] = f
# f3D = np.repeat(f2D[:,:,np.newaxis],nlon,axis=2)

# hbar3D = np.ma.zeros((nt, nlat, nlon))
# hbar3D[:,:,:] = hbar[:,:]

# dSSTdx_temp = sst.diff(dim='lon',n=1)/(dx3D)

# dSSTdy_temp = sst.diff(dim='lat',n=1)/(dy)

# times = sst.time
# lats = sst.lat
# lons = sst.lon

# dSSTdx = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                       coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

# dSSTdy = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                       coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

# dSSTdx.values[:,:,1:] = dSSTdx_temp
# dSSTdy.values[:,1:,:] = dSSTdy_temp

# dSSTdy_mean = dSSTdy.mean(dim='time')
# dSSTdx_mean = dSSTdx.mean(dim='time')

# u_ek = (1/(rho*f3D*hbar3D))*tauy
# v_ek = -(1/(rho*f3D*hbar3D))*taux

# u_ek =  xr.DataArray(u_ek, dims=sst.dims, coords=sst.coords) 
# v_ek =  xr.DataArray(v_ek, dims=sst.dims, coords=sst.coords) 

# Q_ek = -Cbar*(u_ek*dSSTdx + v_ek*dSSTdy)

# #eqi = np.where(lats==0)
# #Q_ek = Q_ek.transpose('time','lat','lon')
# #Q_ek.values[:,eqi,:] = 0

# Q_ek_mean = Q_ek.mean(dim='time')

# Q_ek = Q_ek.where(np.abs(lats) > 0)

# h_anom, h_clim = st.anom(h)

# h_clim_std = h_clim.std(dim='month')
# h_bar = h_clim.mean(dim='month')


# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    #Q_ek,Q_ek_clim= st.anom(Q_ek)
    ps,ps_clim= st.anom(ps)
    sst_ECCO,sst_ECCO_clim= st.anom(sst_ECCO)
    Tmxl_ECCO,Tmxl_ECCO_clim= st.anom(Tmxl_ECCO)

# Remove linear trend
if detr: 
 sst = sst.fillna(0.)    
 sst = xr.DataArray(signal.detrend(sst, axis=0), dims=sst.dims, coords=sst.coords)   

 #ps = ps.fillna(0.)    
 #ps = xr.DataArray(signal.detrend(ps, axis=0), dims=ps.dims, coords=ps.coords)   
 
 thf = thf.fillna(0.)    
 thf = xr.DataArray(signal.detrend(thf, axis=0), dims=thf.dims, coords=thf.coords) 
  
 Q_net_surf = Q_net_surf.fillna(0.)    
 Q_net_surf = xr.DataArray(signal.detrend(Q_net_surf, axis=0), dims=Q_net_surf.dims, coords=Q_net_surf.coords) 
 
# SW_net_surf = SW_net_surf.fillna(0.)    
# SW_net_surf = xr.DataArray(signal.detrend(SW_net_surf, axis=0), dims=SW_net_surf.dims, coords=SW_net_surf.coords) 
 
 # Q_ek = Q_ek.fillna(0.)    
 # Q_ek = xr.DataArray(signal.detrend(Q_ek, axis=0), dims=Q_ek.dims, coords=Q_ek.coords) 
# 
#  
# Q_ek_f = Q_ek_f.fillna(0.)    
# Q_ek_f = xr.DataArray(signal.detrend(Q_ek_f, axis=0), dims=Q_ek_f.dims, coords=Q_ek_f.coords) 


#eqi = np.where(lats[1:-1]==0)
#Q_ek[:,eqi,:] = 0
#Q_ek_test = Q_ek
 

#adv_hs = -(adv_hDivH['X'] + adv_hDivH['Y'])/vol
 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 10.*12
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
    #Q_ek = st.butter_lowpass_filter_xr(Q_ek, cutoff, fs, order)
    sst_ECCO = st.butter_lowpass_filter_xr(sst_ECCO, cutoff, fs, order)
    Tmxl_ECCO = st.butter_lowpass_filter_xr(Tmxl_ECCO, cutoff, fs, order)
    #Q_ek_f = st.butter_lowpass_filter_xr(Q_ek_f, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)

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




#sst_var = sst.var(dim='time')


# Tmxl_var_ECCO = Tmxl_ECCO.var(dim='time')
# sst_var_ECCO = sst_ECCO.var(dim='time')

# # Scale the SST tendency by the ratio of MXL temp tendency / SST tendency from ECCO (as an esimate for a more physical result)
# Tmxlfrac = Tmxl_var_ECCO/sst_var_ECCO
# sst = sst*np.sqrt(Tmxlfrac)

#NINO 3.4
#Niño 3.4 (5N-5S, 170W-120W

tendsst = (sst.shift(time=-2)-sst)[:-2]

tendsst = tendsst/(2*dt)

tendps = (ps.shift(time=-2)-ps)[:-2]

tendps = tendps/(2*dt)
#tendsst = tendsst/dt

nt = sst.shape[0]

thf = thf.isel(time=slice(1,nt-1))
Q_net_surf = Q_net_surf.isel(time=slice(1,nt-1))
sst = sst.isel(time=slice(1,nt-1))

nt = sst.shape[0]

# Make sure sst tendency times match up with other fields
tendsst.time.values = thf.time.values
tendps.time.values = thf.time.values

Qr = Cbar*tendsst - (-thf + Q_net_surf) 

#Qr = Cbar*tendsst -(-thf + Q_net_surf) 
Qr = Qr.transpose('time','lat','lon')

Qr_mean = Qr.mean(dim='time')


Q_s = -thf + Q_net_surf 

T_var = sst.var(dim='time')

#Q_s = -thf + Q_net_surf 

nt = sst.shape[0]
#timeslice = slice(0,nt)
timeslice = slice(int(Tn),nt-int(Tn))

Q_s = Q_s.isel(time=timeslice)
Qr = Qr.isel(time=timeslice)
tendsst = tendsst.isel(time=timeslice)
sst = sst.isel(time=timeslice)

Tn = 6.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
enso = st2.spatial_ave_xr(sst.sel(lon=slice(190,240)), lats=lats.sel(lat=slice(-5,5)))

enso = st.butter_lowpass_filter_xr(enso,cutoff,fs,order) 

if rENSO:
   sst = st2.regressout_x(enso, sst)

#Gulf Stream 37.5–45N, 72–42W
wlon=288
elon=318
nlat = 45
slat = 37.5

#Kuroshio 36–42N, 140–171E
# wlon=142
# elon=180
# nlat = 45
# slat = 36

#Eastern North Pacific
# wlon=190
# elon=225
# nlat = 52
# slat = 40

sst_index = st2.spatial_ave_xr(sst.sel(lon=slice(wlon,elon)), lats=lats.sel(lat=slice(slat,nlat)))
Q_s_index =  st2.spatial_ave_xr(Q_s.sel(lon=slice(wlon,elon)), lats=lats.sel(lat=slice(slat,nlat)))
Qr_index =  st2.spatial_ave_xr(Qr.sel(lon=slice(wlon,elon)), lats=lats.sel(lat=slice(slat,nlat)))
ps_index = st2.spatial_ave_xr(ps.sel(lon=slice(wlon,elon)), lats=lats.sel(lat=slice(slat,nlat)))
#sst_index = (sst_index - sst_index.mean(dim='time'))/sst_index.std(dim='time')

ps_index = (ps_index - ps_index.mean(dim='time'))/ps_index.std(dim='time')
#sst_index = (sst_index - sst_index.mean(dim='time'))/sst_index.std(dim='time')

lagx = 0

reg_sstsst, b = st2.reg(sst_index, sst, lagx=lagx)
reg_sstQs, b = st2.reg(Q_s_index, sst, lagx=lagx)
reg_sstQr, b = st2.reg(Qr_index, sst,lagx=lagx)
reg_psQs, b = st2.reg(Q_s_index, ps, lagx=lagx)
reg_psQr, b = st2.reg(Qr_index, ps, lagx=lagx)

reg_pssst, b = st2.reg(sst_index, ps, lagx=lagx)

reg_psps, b = st2.reg(ps_index, ps, lagx=lagx)

#reg_Qssst, b = st2.reg(sst_index, Q_s, lagx=lagx)
#reg_Qrsst, b = st2.reg(sst_index, Qr, lagx=lagx)



# reg_psQs_tend = (st2.reg(Q_s_index, ps, lagx=-1)[0] - st2.reg(Q_s_index, ps, lagx=1)[0])/2.
# reg_psQr_tend = (st2.reg(Qr_index, ps, lagx=-1)[0] - st2.reg(Qr_index, ps, lagx=1)[0])/2.
# reg_sstQs_tend = (st2.reg(Q_s_index, sst, lagx=-1)[0] - st2.reg(Q_s_index, sst, lagx=1)[0])/2.
# reg_sstQr_tend = (st2.reg(Qr_index, sst, lagx=-1)[0] - st2.reg(Qr_index, sst, lagx=1)[0])/2.

reg_psQs_tend, b = st2.reg(Q_s_index, tendps)
reg_psQr_tend, b =  st2.reg(Qr_index, tendps)
reg_sstQs_tend, b =  st2.reg(Q_s_index, tendsst)
reg_sstQr_tend, b = st2.reg(Qr_index, tendsst)
reg_pssst_tend, b = st2.reg(sst_index, tendps)
reg_sstsst_tend, b = st2.reg(sst_index, tendsst)

reg_psQs_tend = dt*reg_psQs_tend
reg_psQr_tend = dt*reg_psQr_tend
reg_sstQs_tend = dt*reg_sstQs_tend
reg_sstQr_tend = dt*reg_sstQr_tend
reg_pssst_tend = dt*reg_pssst_tend
reg_sstsst_tend = dt*reg_sstsst_tend

reg_pssst_tend_ave = st2.spatial_ave_xr(reg_pssst_tend.sel(lon=slice(wlon,elon)), lats=lats.sel(lat=slice(slat,nlat)))
reg_sstsst_tend_ave = st2.spatial_ave_xr(reg_sstsst_tend.sel(lon=slice(wlon,elon)), lats=lats.sel(lat=slice(slat,nlat)))

print('spatial average reg_pssst_tend', reg_pssst_tend_ave)
print('spatial average reg_sstsst_tend', reg_sstsst_tend_ave)

# #enso_reg, b = st.reg(enso,sst_index)  

# #sst_enso = enso_reg*enso

# #sst_index = sst_index - sst_enso
  
# enso_reg_field, b = st.reg(enso,sst) 

# sst_enso_field = enso_reg_field*enso

# sst = sst - sst_enso_field



# plt.plot(enso.time, enso)
# plt.plot(enso.time, enso_reg*enso)
# plt.plot(enso.time, sst_index - enso_reg*enso)
# #plt.plot(sst_index.time, sst_index)
# plt.axhline(0, bfracolor='k')

#plt.plot(sst_index.time, sst_index)
#plt.axhline(0, color='k')



# Compute covariance betwen SST tendency and fields
# cov_Qr = st2.cov(tendsst, Qr)
# cov_Qs = st2.cov(tendsst, Q_s)
#cov_Qek = st2.cov(tendsst, Q_ek)
#cov_Qek_f = st2.cov(tendsst,Q_ek_f)

# cov_Rnet = st2.cov(tendsst,Q_net_surf)
# cov_thf = st2.cov(tendsst,-thf)
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
    
    
sstvmax = 1.0
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



# latbounds = [30,50]
# lonbounds = [310,345]

# latbounds = [48,60]
# lonbounds = [310,330]

# lonbounds_box = [305,335]
# latbounds_box = [38,50]

lonbounds_box = [wlon,elon]
latbounds_box = [slat,nlat]



x1 = lonbounds_box[0]
x2 = lonbounds_box[1]
y1 = latbounds_box[0]
y2 = latbounds_box[1]


#if Qekplot:
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)

#fieldcmap = plot.Colormap('ColdHot')
cbfrac=0.05
delps = 20
SLPlevels = np.arange(-240,240+delps,delps)


mapper = Mapper()
m,ax=mapper(T_var, bnds=bnds, log=True, title=r'$\sigma_{{T}_{s}}^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=5.0)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, zorder=6, transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tsvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


m,ax=mapper(reg_sstsst, logscale=False, bnds=bnds, title=r'Lag {:1.0f}'.format(-lagx), units=r'K/K', cbfrac=cbfrac, cmap=fieldcmap, vmin=-2.0, vmax=2.0)
ct = ax.contour(x, y, reg_pssst, levels=SLPlevels, colors='k', zorder=7, linewidths=1, transform=cart.crs.PlateCarree())
        #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
for line in ct.collections:
    if line.get_linestyle() != [(None, None)]:
        line.set_linestyle([(0, (8.0, 8.0))])
if np.any(np.round(ct.levels, 5) == 0):
    ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, zorder=6, transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_psregsst_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, -lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
plt.close()

plt.figure(figsize=(12,10))
m,ax=mapper(reg_sstsst_tend, logscale=False, bnds=bnds, title=r'', units=r'K month${^-1}$/K', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.2, vmax=0.2)
ct = ax.contour(x, y, reg_pssst_tend, levels=SLPlevels, colors='k', zorder=7, linewidths=1, transform=cart.crs.PlateCarree())
for line in ct.collections:
    if line.get_linestyle() != [(None, None)]:
        line.set_linestyle([(0, (8.0, 8.0))])
if np.any(np.round(ct.levels, 5) == 0):
    ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, zorder=6, transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_psregsst_tend_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
plt.close()

plt.figure(figsize=(12,10))
m,ax=mapper(reg_psps/100., logscale=False, bnds=bnds, title=r'', units=r'hPa', cbfrac=cbfrac, cmap=fieldcmap, vmin=-4, vmax=4)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, zorder=6, transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_psregps_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
plt.close()

# m,ax=mapper(reg_sstQs, logscale=False, bnds=bnds, title=r'SLP and SST regressed onto $Q_s$ (Lag {:1.0f})'.format(lagx), units=r'K/(W m$^2$)', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.01, vmax=0.01)
# ct = ax.contour(x, y, reg_psQs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
#         #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
# for line in ct.collections:
#     if line.get_linestyle() != [(None, None)]:
#         line.set_linestyle([(0, (8.0, 8.0))])
# if np.any(np.round(ct.levels, 5) == 0):
#     ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstregQs_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(reg_sstQr, logscale=False, bnds=bnds, title=r'SLP and SST regressed onto $Q_o$ (Lag {:1.0f})'.format(lagx), units=r'K/(W m$^2$)', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.01, vmax=0.01)
# ct = ax.contour(x, y, reg_psQr, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
# for line in ct.collections:
#     if line.get_linestyle() != [(None, None)]:
#         line.set_linestyle([(0, (8.0, 8.0))])
# if np.any(np.round(ct.levels, 5) == 0):
#     ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstregQr_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
# plt.close()


# plt.figure(figsize=(12,10))
# m,ax=mapper(reg_sstQs_tend, logscale=False, bnds=bnds, title=r'SLP and SST tendency regressed onto $Q_s$', units=r'(K month$^{-1}$) /(W m$^2$)', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.01, vmax=0.01)
# ct = ax.contour(x, y, reg_psQs_tend, levels=SLPlevels/2., colors='k', linewidths=1, transform=cart.crs.PlateCarree())
#         #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
# for line in ct.collections:
#     if line.get_linestyle() != [(None, None)]:
#         line.set_linestyle([(0, (8.0, 8.0))])
# if np.any(np.round(ct.levels, 5) == 0):
#     ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstregQs_tend_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
# plt.close()

# plt.figure(figsize=(12,10))
# m,ax=mapper(reg_sstQr_tend, logscale=False, bnds=bnds, title=r'SLP and SST tendency regressed onto $Q_o$'.format(lagx), units=r'(K month$^{-1}$) /(W m$^2$)', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.01, vmax=0.01)
# ct = ax.contour(x, y, reg_psQr_tend, levels=SLPlevels/2., colors='k', linewidths=1, transform=cart.crs.PlateCarree())
# for line in ct.collections:
#     if line.get_linestyle() != [(None, None)]:
#         line.set_linestyle([(0, (8.0, 8.0))])
# if np.any(np.round(ct.levels, 5) == 0):
#     ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstregQr_tend_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
# plt.close()


# plt.figure(figsize=(12,10))
# m,ax=mapper(reg_sstQs_tend*np.std(Q_s_index,axis=0), logscale=False, bnds=bnds, title=r'SLP and SST tendency regressed onto $Q_s$', units=r'K/month', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.3, vmax=0.3)
# ct = ax.contour(x, y, reg_psQs_tend*np.std(Q_s_index,axis=0), levels=SLPlevels*20., colors='k', linewidths=1, transform=cart.crs.PlateCarree())
#         #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
# for line in ct.collections:
#     if line.get_linestyle() != [(None, None)]:
#         line.set_linestyle([(0, (8.0, 8.0))])
# if np.any(np.round(ct.levels, 5) == 0):
#     ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstregQsSTD_tend_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
# plt.close()

# plt.figure(figsize=(12,10))
# m,ax=mapper(reg_sstQr_tend*np.std(Qr_index, axis=0), logscale=False, bnds=bnds, title=r'SLP and SST tendency regressed onto $Q_o$'.format(lagx), units=r'K/month', cbfrac=cbfrac, cmap=fieldcmap, vmin=-0.3, vmax=0.3)
# ct = ax.contour(x, y, reg_psQr_tend*np.std(Qr_index,axis=0), levels=SLPlevels*20., colors='k', linewidths=1, transform=cart.crs.PlateCarree())
# for line in ct.collections:
#     if line.get_linestyle() != [(None, None)]:
#         line.set_linestyle([(0, (8.0, 8.0))])
# if np.any(np.round(ct.levels, 5) == 0):
#     ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=3, transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstregQrSTD_tend_lag{:1.0f}_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, lagx, Tn/12., latbounds_box[0], latbounds_box[1], str(detr)[0]))
# plt.close()













