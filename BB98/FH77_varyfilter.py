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
import numpy.fft as fft
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, ScalarFormatter,
                               AutoMinorLocator, LogLocator)
import proplot as plot
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
fta = xr.open_dataset(fin + 'MERRA2_t10m_monthly1980to2017.nc')
fthf = xr.open_dataset(fin + 'MERRA2_thf_monthly1980to2017.nc')
#fSLP = xr.open_dataset(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = xr.open_dataset(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = xr.open_dataset(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
#fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
#fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')
ftau = xr.open_dataset(fin + 'MERRA2_tau_monthly1980to2019.nc')
fssh = xr.open_dataset(fin + 'ncep.ssh.198001-201912.nc')
#fseaice = xr.open_dataset(fin + 'MERRA2_seaice_monthly1980to2019.nc')


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
#fta = xr.open_dataset(fin + 'oaflux_ta_1980to2017.nc')
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

plot.rc.update({'mathtext.fontset': 'cm'})
plot.rc.update({'mathtext.default': 'it'})
#plot.rc.update({'font.size': 18})
#plot.rc.update({'axes.titlesize': 22})
plot.rc.update({'small':12})
plot.rc.update({'large':14})


#EDIT THIS FOR BOUNDS
lonbounds = [0.5,359.5]
latbounds = [-89.5,89.5]

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
# lonbounds = [120.5,290.5]
# latbounds = [-10,60]

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

#Midlatitude Pacific
#latbounds = [30,45]
#lonbounds = [160,220]

minlon=lonbounds[0]
maxlon=lonbounds[1]
minlat = latbounds[0]
maxlat= latbounds[1]

#ps = fSLP.SLP
# ps = ps/1e2
# ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
# nt_ps = ps.shape[0]
# ps = ps[tskip:,:]

# #ps = fsst('msl')
# ps = ps/1e2
# ps = ps.assign_coords(lon=(ps.lon % 360)).roll(lon=((ps.shape[2] // 2)-1))
# ps = ps.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))


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
#sst = fsstM2.TSKINWTR
#ta = fta.T10M
#ta.load()
# lhf = fthf.EFLUX
# shf = fthf.HFLUX
# LW_net_surf = radfile.LWGNT
# SW_net_surf = radfile.SWGNT


#OAFlux
sst = fsstoa.tmpsf
ta = fta.T10M
ta.load()
#ta = fta.tmp2m 
#ta = ta.where(~(ta==327.66))
lhf = fthf.lhtfl
shf = fthf.shtfl
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT

sst = sst+273.15
#ta = ta+273.15

#ta.load()

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
    
if ta.lon.max() <= 180:
    ta = ta.assign_coords(lon=(ta.lon % 360)).roll(lon=((ta.shape[2] // 2)-1))
    
sstbar = sst.mean(dim='time')
tabar = ta.mean(dim='time')

ta= ta.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
ta = ta.transpose('time', 'lat', 'lon')
    
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
    ta = ta[:Q_net_surf.shape[0],:,:]
    #sst_interp = sst_interp[:Q_net_surf.shape[0],:,:]
thf = thf[:Q_net_surf.shape[0],:,:]
ssh = ssh[:Q_net_surf.shape[0],:,:]
# taux = taux[:Q_net_surf.shape[0],:,:]
# tauy = tauy[:Q_net_surf.shape[0],:,:]
#ps = ps[:Q_net_surf.shape[0],:,:]
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
ta.time.values = dates_sst
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

regridder = xe.Regridder(ta, ds_out, 'bilinear', reuse_weights=True)
ta = regridder(ta)  # print basic regridder information.

regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=True)
h = regridder(h)

#regridder = xe.Regridder(Tmxlfrac, ds_out, 'bilinear', reuse_weights=True)
#Tmxlfrac = regridder(Tmxlfrac)

regridder = xe.Regridder(thf, ds_out, 'bilinear', reuse_weights=True)
thf = regridder(thf)
#
regridder = xe.Regridder(hbar, ds_out, 'bilinear', reuse_weights=True)
hbar = regridder(hbar)

regridder = xe.Regridder(sstbar, ds_out, 'bilinear', reuse_weights=True)
sstbar = regridder(sstbar)

regridder = xe.Regridder(tabar, ds_out, 'bilinear', reuse_weights=True)
tabar = regridder(tabar)

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
    ta,ta_clim = st.anom(ta)
    #Q_ek,Q_ek_clim= st.anom(Q_ek)
    sst_ECCO,sst_ECCO_clim= st.anom(sst_ECCO)
    Tmxl_ECCO,Tmxl_ECCO_clim= st.anom(Tmxl_ECCO)



# Remove linear trend
if detr: 
 sst = sst.fillna(0.)    
 sst = xr.DataArray(signal.detrend(sst, axis=0), dims=sst.dims, coords=sst.coords)   
 
 ta = ta.fillna(0.)    
 ta = xr.DataArray(signal.detrend(ta, axis=0), dims=ta.dims, coords=ta.coords)   

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
ta = ta.where(ocean_points)
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
ta = ta.isel(time=slice(1,nt-1))

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

#lambdaQ_tot = (lambdaQ_totlag1 + lambdaQ_totlag2 + lambdaQ_totlag3)/3.
lambdaQ_tot = lambdaQ_totlag1

lambdaQ_slag0 = st.cov(sste, Q_se, lagx=0)/st.cov(sste,sste, lagx=0)
lambdaQ_slag1 = st.cov(sste, Q_se, lagx=-1)/st.cov(sste,sste, lagx=-1)
lambdaQ_slag2 = st.cov(sste, Q_se, lagx=-2)/st.cov(sste,sste, lagx=-2)
lambdaQ_slag3 = st.cov(sste, Q_se, lagx=-3)/st.cov(sste,sste, lagx=-3)

lambdaQ_rlag0 = st.cov(sste, Q_re, lagx=0)/st.cov(sste,sste, lagx=0)
lambdaQ_rlag1 = st.cov(sste, Q_re, lagx=-1)/st.cov(sste,sste, lagx=-1)
lambdaQ_rlag2 = st.cov(sste, Q_re, lagx=-2)/st.cov(sste,sste, lagx=-2)
lambdaQ_rlag3 = st.cov(sste, Q_re, lagx=-3)/st.cov(sste,sste, lagx=-3)

#lambdaQ_s = (lambdaQ_slag1 + lambdaQ_slag2 + lambdaQ_slag3)/3.
#lambdaQ_r = (lambdaQ_rlag1 + lambdaQ_rlag2 + lambdaQ_rlag3)/3.

lambdaQ_s = lambdaQ_slag1
lambdaQ_r = lambdaQ_rlag1

lambdaQ_totT = lambdaQ_tot*sst
lambdaQ_sT = lambdaQ_s*sst
lambdaQ_rT = lambdaQ_r*sst

#Q_totstar = Q_tot - lambdaQ_totT
Q_sstar = Q_s - lambdaQ_sT
Q_rstar = Q_r - lambdaQ_rT

Q_totstar = Q_tot - lambdaQ_totT

freqs =  fft.fftfreq(nt)
freqs = freqs[:int(nt/2)]

T_A = fft.fft(sst,axis=0, norm='ortho')
Qtot_A = fft.fft(Q_tot,axis=0, norm='ortho')
Qtotstar_A = fft.fft(Q_totstar,axis=0, norm='ortho')
Qsstar_A = fft.fft(Q_sstar,axis=0,norm='ortho')
Qrstar_A = fft.fft(Q_rstar,axis=0,norm='ortho')
Qs_A = fft.fft(Q_s,axis=0,norm='ortho')
Qr_A = fft.fft(Q_r,axis=0,norm='ortho')

Tpower_temp = np.abs(T_A)**2
Qtotpower_temp = np.abs(Qtot_A)**2
Qtotstarpower_temp = np.abs(Qtotstar_A)**2
Qsstarpower_temp = np.abs(Qsstar_A)**2
Qrstarpower_temp = np.abs(Qrstar_A)**2
Qspower_temp = np.abs(Qsstar_A)**2
Qrpower_temp = np.abs(Qrstar_A)**2

Tpower_temp = 2*Tpower_temp[:int(nt/2)]
Qtotpower_temp = 2*Qtotpower_temp[:int(nt/2)]
Qtotstarpower_temp = 2*Qtotstarpower_temp[:int(nt/2)]
Qsstarpower_temp = 2*Qsstarpower_temp[:int(nt/2)]
Qrstarpower_temp = 2*Qrstarpower_temp[:int(nt/2)]
Qspower_temp = 2*Qsstarpower_temp[:int(nt/2)]
Qrpower_temp = 2*Qrstarpower_temp[:int(nt/2)]

Tpower_temp = Tpower_temp/nt
Qtotpower_temp = Qtotpower_temp/nt
Qtotstarpower_temp = Qtotstarpower_temp/nt
Qsstarpower_temp = Qsstarpower_temp/nt
Qrstarpower_temp = Qrstarpower_temp/nt
Qspower_temp = Qsstarpower_temp/nt
Qrpower_temp = Qrstarpower_temp/nt

nfreq = len(freqs)
lats = sst.lat
lons = sst.lon
nlat = len(lats)
nlon = len(lons)

f = xr.DataArray(np.nan*np.zeros([nfreq]), coords={'freq': freqs},dims=['freq'])
f.values = freqs

Tpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qtotpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qtotstarpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])


Qsstarpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qrstarpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qspower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Qrpower = xr.DataArray(np.nan*np.zeros([nfreq,nlat,nlon]),
                      coords={'freq': f, 'lat': lats, 'lon':lons},dims=['freq', 'lat','lon'])

Tpower.values = Tpower_temp
Qtotpower.values = Qtotpower_temp
Qtotstarpower.values = Qtotstarpower_temp
Qsstarpower.values = Qsstarpower_temp
Qrstarpower.values = Qrstarpower_temp
Qspower.values = Qspower_temp
Qrpower.values = Qrpower_temp

lowmaxi=3
f = f[lowmaxi:]
Tpower = Tpower[lowmaxi:]
Qtotpower = Qtotpower[lowmaxi:]
Qtotstarpower = Qtotstarpower[lowmaxi:]
Qsstarpower = Qsstarpower[lowmaxi:]
Qrstarpower = Qrstarpower[lowmaxi:]
Qspower = Qspower[lowmaxi:]
Qrpower = Qrpower[lowmaxi:]

F_o = Q_rstar
F_a = Q_sstar

T_var = sst.var(dim='time')

F_o_var = F_o.var(dim='time')
F_a_var = F_a.var(dim='time')

Q_s_var = Q_s.var(dim='time')
Q_o_var = Q_r.var(dim='time')

Q_totstar_var = (Q_totstar).var(dim='time')
covQsQrstar = 0.5*(Q_totstar_var - F_a_var - F_o_var)

Q_tot_var = (Q_totstar).var(dim='time')
covQsQr = 0.5*(Q_tot_var - Q_s_var - Q_o_var)

QsQrpower = 0.5*(Qtotpower - Qspower - Qrpower)
QsQrstarpower = 0.5*(Qtotstarpower - Qsstarpower - Qrstarpower)


# Spatial averaging
# latbounds_ave = [30,45]
# lonbounds_ave = [160,220]

#North pacific
# lonbounds_ave = [120,270]
# latbounds_ave = [20,60]

#North Atlantic
# latbounds_ave=[20,60]
# lonbounds_ave=[270,360]

#Global
# lonbounds_ave = [0.5,359.5]
# latbounds_ave = [-64.5,64.5]

#Extratropical NH
lonbounds_ave = [0.5,359.5]
latbounds_ave = [30,60]

Tpower_ave = st2.spatial_ave_xr(Tpower.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
F_o_var_ave =  st2.spatial_ave_xr(F_o_var.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
F_a_var_ave =  st2.spatial_ave_xr(F_a_var.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
F_o_power_ave = st2.spatial_ave_xr(Qrstarpower.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
F_a_power_ave = st2.spatial_ave_xr(Qsstarpower.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
Cbar_ave = st2.spatial_ave_xr(Cbar.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))

FaFo_power_ave = st2.spatial_ave_xr(QsQrstarpower.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))


# Compute power spectra for T_o and T_a based on solutions of the BB98 model
monthtosec=3600*24*30
freq_sec = f/monthtosec

lambda_tot = lambdaQ_tot

#lambda_tot = lambdaQ_s

gamma_o = Cbar_ave

coef = lambda_tot**2 + (2*np.pi*gamma_o*freq_sec)**2
# White-noise assumption
F_a_pow_an = F_a_var_ave/len(f)
F_o_pow_an = F_o_var_ave/len(f)

#F_a_pow_an = F_a_power_ave
#F_o_pow_an = F_o_power_ave
#F_aF_o_pow_an = FaFo_power_ave

T_o_Fa = (F_a_pow_an)/coef
T_o_Fo = (F_o_pow_an)/coef

#T_o_Fo[:] = 0

#T_o_Fo = 0
#T_o_FaFo = (F_aF_o_pow_an)/coef

#T_o_Fa = T_o_Fa + T_o_FaFo
#T_o_Fo = T_o_Fo + T_o_FaFo

T_o_power_an = T_o_Fa + T_o_Fo 

T_o_power_an_ave = st2.spatial_ave_xr(T_o_power_an.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
T_o_Fa_ave = st2.spatial_ave_xr(T_o_Fa.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))
T_o_Fo_ave = st2.spatial_ave_xr(T_o_Fo.sel(lon=slice(lonbounds_ave[0],lonbounds_ave[1])), lats=lats.sel(lat=slice(latbounds_ave[0],latbounds_ave[1])))

T_o_var_sum = T_o_power_an.sum('freq')
T_o_var_Fa = T_o_Fa.sum('freq')
T_o_var_Fo = T_o_Fo.sum('freq')

# Plotting
# latbounds_plot = [15,60]
# lonbounds_plot = [120,260]

lonbounds_plot = [0,360]
latbounds_plot = [-65.,65.]

bnds = [np.round(lonbounds_plot[0]-359), np.round(lonbounds_plot[1]-361), latbounds_plot[0], latbounds_plot[1]]

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

x1 = lonbounds_ave[0]
x2 = lonbounds_ave[1]
y1 = latbounds_ave[0]
y2 = latbounds_ave[1]

varmin=10**2
varmax=10**4
lambdamax=60


cbfrac=0.10

#period in months
#ftop = 1/f
#period in years
#ftop = ftop/12.

# fmin = (1/25)/12.
# fmax = 0.5

# locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)

# plt.figure(figsize=(10,12))
# plt.plot(f, T_o_power_an_ave, color='k', label='solution')
# plt.plot(f, Tpower_ave, color='grey', label='observed')
# plt.legend(loc='best')
# ax=plt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(fmin,fmax)
# ax.set_ylim(10**(-5),10**(0))
# ax.set_xlabel('Frequency (month$^{-1}$)')
# ax.set_ylabel('Variance (K$^{2}$')
# ax2 = ax.twiny()
# ax2.set_xlim((1/fmin)/12, (1/fmax)/12)
# ax2.set_xscale('log')
# ax2.xaxis.set_major_formatter(ScalarFormatter())
# ax2.set_xticks([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xticklabels([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xlabel('Period (years)')
# ax.tick_params(which='major', length=10)
# ax.tick_params(which='minor', length=4)
# ax2.tick_params(which='major', length=10)
# ax2.tick_params(which='minor', length=0)
# ax.xaxis.set_minor_locator(locmin)
# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.savefig(fout + '{:s}_To_powerspectra_analytical_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds_ave[0], latbounds_ave[1], str(detr)[0]))

T_o_sum = T_o_power_an_ave.cumsum('freq')
T_o_Fo_var = T_o_Fo_ave.cumsum('freq')
T_o_Fa_var = T_o_Fa_ave.cumsum('freq')

T_o_sum = T_o_sum[::-1]
T_o_Fo_var = T_o_Fo_var[::-1]
T_o_Fa_var = T_o_Fa_var[::-1]

#T_o_sum = T_o_Fo_var

filter_lengths = (1/f[::-1])/12.

filter_lengths = filter_lengths - filter_lengths[0]

#T_o_save = T_o_sum

#T_o_save2 = T_o_save

plt.figure(1)
fig, axs = plot.subplots(ncols=1, nrows=2, aspect=1.2, tight=True, share=False, hratios=(3,2))
#plot.subplots(ncols=2, nrows=3)
h1=axs[0].plot(filter_lengths, T_o_Fa_var, color='C2', label=r'$F_a$', linewidth=2, zorder=5)
#plt.plot(Tns/12., ave_T_var_thf, color='C3', label=r'$THF$')
#plt.plot(Tns/12., ave_T_var_Rnet, color='C4', label=r'$R_{net}$')
h2=axs[0].plot(filter_lengths, T_o_Fo_var, color='C0', label=r'$F_o$', linewidth=2, zorder=5)
#plt.plot(Tns/12., ave_T_var_Qek, color='C1', label=r'$Q_{ek}$')
h3=axs[0].plot(filter_lengths, T_o_sum, color='k', label=r'$\sigma_T^2$', linewidth=2, zorder=5)
#h4=axs[0].plot(filter_lengths, T_o_save2, color='grey', label=r'$\sigma_{T^{2}_{nodyn}}$', linewidth=2, zorder=5)
#h4=axs[0].plot(Tns/12., ave_T_var, color='C5', label=r'$\sigma_T^2$')
axs[0].axhline(0, color='k', linewidth=1)

#hs=[h1,h2,h3,h4]
#Global/tropics
axs[0].set_ylim(-0.1,0.38)
#axs[0].set_ylim(-0.1,0.6)
#WBC
#plt.ylim(-0.22,0.7)
#NH
# yticklbls = np.round(np.arange(-0.3,0.6,0.1),1)
# axs[0].set_yticks(yticklbls)
# axs[0].set_ylim(-0.35,0.55)
#SH
#ax1.set_ylim(-0.04,0.3)
#NA
#axs[0].set_ylim(-0.1,0.8)
#plt.ylim(-0.02,0.6)
axs[0].set_xlim(0,6)
axs[0].set_ylabel('Contribution to $\sigma_T^2$ (K$^{2}$)')

#leg=fig.legend(hs, loc='r')
frac_T_var_Qs = T_o_Fa_var/T_o_sum
#frac_T_var_thf = ave_T_var_thf/ave_T_var_sum
#frac_T_var_Rnet = ave_T_var_Rnet/ave_T_var_sum
frac_T_var_Qr = T_o_Fo_var/T_o_sum
#frac_T_var_Qnet = ave_T_var_Rnet/ave_T_var_sum
#frac_T_var_Qek = ave_T_var_Qek/ave_T_var_sum

y0=frac_T_var_Qr
y1=y0+frac_T_var_Qs
#y2=y1+frac_T_var_Qek

yticklbls = np.array([0,0.2,0.4,0.6,0.8,1.0])

#leg_texts = [leg[1].get_texts()[0], leg[2].get_texts()[0], leg[3].get_texts()[0], leg[4].get_texts()[0]]

#maxs = ax1.panel('b', space=0.5, share=False)
axs[1].fill_between(filter_lengths, y0, color='C0', label=r'$Q_o$', alpha=0.8, linewidth=0)
axs[1].fill_between(filter_lengths, y1, y0, color='C2', label=r'$Q_s$', alpha=0.8, linewidth=0)
axs[1].set_xlabel('Filter Length (years)')
axs[1].set_ylim(0,1.0)
axs[1].set_xlim(0,6)
axs[1].set_yticks(yticklbls)
axs[1].set_ylabel('Fractional Contribution')
#plt.legend(loc='best')
# shift = max([t.get_window_extent().width for t in leg_texts])
# for t in leg_texts:
#     t.set_ha('right') # ha is alias for horizontalalignment
#     t.set_position((shift,0))
fig.savefig(fout + '{:s}_MODELsstvarbudget_varytimefilter_{:2.0f}Nto{:2.0f}N.png'.format(dataname, latbounds[0], latbounds[1]))
plt.close(fig)

# #fig,ax=plt.subplots(nrows=5,ncols=1,sharex=True)
# plt.figure(figsize=(10,12))
# plt.plot(f,Tpower_ave,label='$T_o$',color='k')
# plt.plot(f,F_a_power_ave,label='$F_a$',color='g')
# plt.plot(f,F_o_power_ave,label='$F_o$',color='b')
# plt.legend(loc='best')
# ax=plt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(fmin,fmax)
# ax.set_ylim(10**(-5),10**(2))
# ax.set_xlabel('Frequency (month$^{-1}$)')
# ax.set_ylabel('Variance')
# ax2 = ax.twiny()
# ax2.set_xlim((1/fmin)/12, (1/fmax)/12)
# ax2.set_xscale('log')
# ax2.xaxis.set_major_formatter(ScalarFormatter())
# ax2.set_xticks([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xticklabels([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xlabel('Period (years)')
# ax.tick_params(which='major', length=10)
# ax.tick_params(which='minor', length=4)
# ax2.tick_params(which='major', length=10)
# ax2.tick_params(which='minor', length=0)
# ax.xaxis.set_minor_locator(locmin)
# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.savefig(fout + '{:s}_forcing_powerspectra_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds_ave[0], latbounds_ave[1], str(detr)[0]))

# #fig,ax=plt.subplots(nrows=5,ncols=1,sharex=True)
# plt.figure(figsize=(10,12))
# plt.plot(f,Tpower_ave,label='$T_o$',color='k')
# plt.plot(f,Qspower_ave,label='$Q_s$',color='g')
# plt.plot(f,Qopower_ave,label='$Q_o$',color='b')
# plt.legend(loc='lower left')
# ax=plt.gca()
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlim(fmin,fmax)
# ax.set_ylim(10**(-5),10**(2))
# ax.set_xlabel('Frequency (month$^{-1}$)')
# ax.set_ylabel('Variance')
# ax2 = ax.twiny()
# ax2.set_xlim((1/fmin)/12, (1/fmax)/12)
# ax2.set_xscale('log')
# ax2.xaxis.set_major_formatter(ScalarFormatter())
# ax2.set_xticks([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xticklabels([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xlabel('Period (years)')
# ax.tick_params(which='major', length=10)
# ax.tick_params(which='minor', length=4)
# ax2.tick_params(which='major', length=10)
# ax2.tick_params(which='minor', length=0)
# ax.xaxis.set_minor_locator(locmin)
# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.savefig(fout + '{:s}_QsQo_powerspectra_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds_ave[0], latbounds_ave[1], str(detr)[0]))

# #fig,ax=plt.subplots(nrows=5,ncols=1,sharex=True)
# plt.figure(figsize=(8,10))
# #plt.plot(f,QsQopower_ave,label='$|Q_sQ_o|$',color='k')
# plt.plot(f,FaFopower_ave,label='$|F_aF_o|$',color='k')
# plt.legend(loc='best')
# ax=plt.gca()
# ax.set_xscale('log')
# ax.set_ylim(-10**(2),10**(2))
# ax.set_yscale('symlog')
# ax.set_xlim(fmin,fmax)
# ax.set_xlabel('Frequency (month$^{-1}$)')
# ax.set_ylabel('Variance')
# ax2 = ax.twiny()
# ax.axhline(0, color='k')
# ax2.set_xlim((1/fmin)/12, (1/fmax)/12)
# ax2.set_xscale('log')
# ax2.xaxis.set_major_formatter(ScalarFormatter())
# ax2.set_xticks([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xticklabels([0.1,1.0,2.5,5.0,10,25])
# ax2.set_xlabel('Period (years)')
# ax.tick_params(which='major', length=10)
# ax.tick_params(which='minor', length=4)
# ax2.tick_params(which='major', length=10)
# ax2.tick_params(which='minor', length=0)
# ax.xaxis.set_minor_locator(locmin)
# ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# ax.yaxis.set_minor_locator(locmin)
# ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# plt.savefig(fout + '{:s}_FaFo_cospectrum_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds_ave[0], latbounds_ave[1], str(detr)[0]))

# mapper = Mapper()
# m,ax=mapper(T_var, log=True, bnds=bnds, title='$\sigma^2_T$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tvar_obs_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)

# mapper = Mapper()
# m,ax=mapper(T_o_var_sum, log=True, bnds=bnds, title='sum', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tvar_sum_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(T_o_var_Fo, norm=lognorm, bnds=bnds, title='$\widetilde{F}_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tvar_Fo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(T_o_var_Fa, norm=lognorm, bnds=bnds, title='$\widetilde{F}_a$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Tvar_Fa_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()


# m,ax=mapper(sstbar, bnds=bnds, logscale=False, log=False,  title=r'$\overline{T}_o$', units=r'K', cbfrac=cbfrac, cmap=sstcmap, vmin=280, vmax=300)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_sstbar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(tabar, bnds=bnds, logscale=False, log=False,  title=r'$\overline{T}_a$', units=r'K', cbfrac=cbfrac, cmap=sstcmap, vmin=280, vmax=300)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_tabar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(F_o_var, bnds=bnds, logscale=False, log=True,  title=r'$\overline{F^{\,\prime 2}_o}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Fovar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(F_a_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{F^{\,\prime 2}_a}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_Favar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(lambda_o, logscale=False, bnds=bnds, title=r'$\lambda_{o}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambda_o_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(lambda_odyn, logscale=False, bnds=bnds, title=r'$\lambda_{o,dyn}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambda_odyn_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(lambda_a, logscale=False, bnds=bnds, title=r'$\lambda_{a}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambda_a_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(lambda_sa, logscale=False, bnds=bnds, title=r'$\lambda_{s,a}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambda_sa_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper = Mapper()
# m,ax=mapper(lambda_so, logscale=False, bnds=bnds, title=r'$\lambda_{s,o}$', units=r'W m$^{-2}$ K$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-lambdamax, vmax=lambdamax)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_lambda_so_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()




