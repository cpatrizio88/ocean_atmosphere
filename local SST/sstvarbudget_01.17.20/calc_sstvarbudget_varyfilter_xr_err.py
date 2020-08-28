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
import proplot as plot
#import seaborn as sns
#import proplot as plot

# class MidpointNormalize(colors.Normalize):
#     def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#         self.midpoint = midpoint
#         colors.Normalize.__init__(self, vmin, vmax, clip)

#     def __call__(self, value, clip=None):
#         # I'm ignoring masked values and all kinds of edge cases to make a
#         # simple example...
#         x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#         return np.ma.masked_array(np.interp(value, x, y))




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
#fh = xr.open_dataset(fin + 'CESM2_hbar36years.nc')

flhf =   xr.open_dataset(fin + 'OAFlux_lhf_monthly1980to2017.nc')
fshf =   xr.open_dataset(fin + 'OAFlux_shf_monthly1980to2017.nc')

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

#lats = h.lat
#lons = h.lon

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
#hbar = h
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
lonbounds = [0.5,359.5]
latbounds = [-64.5,64.5]

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

#Extratropical NH
lonbounds = [0,360]
latbounds = [30.,60.]

#Extratropical SH
# lonbounds = [0,360]
# latbounds = [-58,-30]

#Tropics
# lonbounds = [0,360]
# latbounds = [-30,30]



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


err_lhf = flhf.err
err_shf = fshf.err
err_sst = fsstoa.err

err_lhf = err_lhf.where(err_lhf < 3200)
err_shf = err_shf.where(err_shf < 3200)

err_thf = err_lhf + err_shf





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

err_thf = err_thf.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
err_sst = err_sst.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))

sst = sst.transpose('time', 'lat', 'lon')
err_thf = err_thf.transpose('time', 'lat', 'lon')
err_sst = err_sst.transpose('time', 'lat', 'lon')

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

err_sst=err_sst[:Q_net_surf.shape[0],:,:]
err_thf=err_thf[:Q_net_surf.shape[0],:,:]
#sst_err=sst_err[:Q_net_surf.shape[0],:,:]
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
err_sst.time.values = dates
err_thf.time.values = dates
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

if dataname == 'OAFlux':
    sst = sst.sel(time=slice('1980-12-01','2015-01-01'))
    thf = thf.sel(time=slice('1980-12-01','2015-01-01'))
    Q_net_surf = Q_net_surf.sel(time=slice('1980-12-01','2015-01-01'))
    err_sst = err_sst.sel(time=slice('1980-12-01','2015-01-01'))
    err_thf = err_thf.sel(time=slice('1980-12-01','2015-01-01'))
# sst = sst.sel(time=slice('1992-02-01','2015-11-01'))
# thf = thf.sel(time=slice('1992-02-01','2015-11-01'))
# Q_net_surf = Q_net_surf.sel(time=slice('1992-12-01','2015-11-01'))
# h = h.sel(time=slice('1992-02-01','2015-11-01'))
# taux = taux.sel(time=slice('1992-02-01','2015-11-01'))
# tauy = tauy.sel(time=slice('1992-02-01','2015-11-01'))

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
fsave = False
anom_flag = True
timetend=False
detr=True
rENSO=False
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

regridder = xe.Regridder(sst, ds_out, 'bilinear', reuse_weights=False)
sst = regridder(sst)  # print basic regridder information.
regridder.clean_weight_file()

regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=False)
h = regridder(h)
regridder.clean_weight_file()

#regridder = xe.Regridder(Tmxlfrac, ds_out, 'bilinear', reuse_weights=True)
#Tmxlfrac = regridder(Tmxlfrac)

regridder = xe.Regridder(thf, ds_out, 'bilinear', reuse_weights=False)
thf = regridder(thf)
regridder.clean_weight_file()
#
regridder = xe.Regridder(hbar, ds_out, 'bilinear', reuse_weights=False)
hbar = regridder(hbar)
regridder.clean_weight_file()

# regridder = xe.Regridder(hmean, ds_out, 'bilinear', reuse_weights=False)
# hmean= regridder(hmean)
# regridder.clean_weight_file()

regridder = xe.Regridder(Q_net_surf, ds_out, 'bilinear', reuse_weights=False)
Q_net_surf  = regridder(Q_net_surf)
regridder.clean_weight_file()

regridder = xe.Regridder(err_sst, ds_out, 'bilinear', reuse_weights=False)
err_sst  = regridder(err_sst)
regridder.clean_weight_file()

regridder = xe.Regridder(err_thf, ds_out, 'bilinear', reuse_weights=False)
err_thf  = regridder(err_thf)
regridder.clean_weight_file()


# regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
# taux  = regridder(taux)
# #
# regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
# tauy  = regridder(tauy)

regridder = xe.Regridder(sst_ECCO, ds_out, 'bilinear', reuse_weights=False)
sst_ECCO  = regridder(sst_ECCO)
regridder.clean_weight_file()
#
regridder = xe.Regridder(Tmxl_ECCO, ds_out, 'bilinear', reuse_weights=False)
Tmxl_ECCO  = regridder(Tmxl_ECCO)
regridder.clean_weight_file()

#hbar = hbar.interpolate_na(dim='lon').interpolate_na(dim='lat')



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

# Q_ek = Q_ek.transpose('time', 'lat', 'lon')


# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    err_sst,err_sst_clim = st.anom(err_sst)
    err_thf,err_thf_clim = st.anom(err_thf)
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
lats = sst.lat
lons = sst.lon
 
Tn = 6.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
enso = st2.spatial_ave_xr(sst.sel(lon=slice(190,240)), lats=lats.sel(lat=slice(-5,5))) 
 
  
if rENSO:
    sst = st2.regressout_x(enso, sst)
    thf = st2.regressout_x(enso, thf)
    Q_net_surf = st2.regressout_x(enso, Q_net_surf)

 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)

delTn = 4
Tnmax = 6*12
Tns = np.arange(0,Tnmax+delTn,delTn)

Tns = Tns*1.


ave_T_var_Qek = np.zeros((len(Tns)))    
ave_T_var_Qs = np.zeros((len(Tns)))
ave_T_var_thf = np.zeros((len(Tns)))
ave_T_var_Rnet = np.zeros((len(Tns)))
ave_T_var_Qr = np.zeros((len(Tns)))
ave_T_var_sum = np.zeros((len(Tns)))
ave_T_var = np.zeros((len(Tns)))

ave_errTvar_Qs = np.zeros((len(Tns)))
ave_errTvar_Qr = np.zeros((len(Tns)))
ave_errTvar_sum = np.zeros((len(Tns)))



# Tmxl_var_ECCO = Tmxl_ECCO.var(dim='time')
# sst_var_ECCO = sst_ECCO.var(dim='time')


# Scale the SST tendency by the ratio of MXL temp tendency / SST tendency from ECCO (as an esimate for a more physical result)


sst_raw = sst
thf_raw = thf
Q_net_surf_raw = Q_net_surf
sst_ECCO_raw = sst_ECCO
Tmxl_ECCO_raw = Tmxl_ECCO
#Q_ek_raw = Q_ek


for jj, Tn in enumerate(Tns):
    
    cutoff = 1/Tn
    
    print('Tn',Tn)
    
    if Tn > 0:
        sst = st.butter_lowpass_filter_xr(sst_raw, cutoff, fs, order)
        thf = st.butter_lowpass_filter_xr(thf_raw, cutoff, fs, order)
        Q_net_surf = st.butter_lowpass_filter_xr(Q_net_surf_raw, cutoff, fs, order)
        err_sst = st.butter_lowpass_filter_xr(err_sst, cutoff, fs, order)
        err_thf = st.butter_lowpass_filter_xr(err_thf, cutoff, fs, order)
        sst_ECCO = st.butter_lowpass_filter_xr(sst_ECCO_raw, cutoff, fs, order)
        Tmxl_ECCO = st.butter_lowpass_filter_xr(Tmxl_ECCO_raw, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)
        
    Tmxl_var_ECCO = Tmxl_ECCO.var(dim='time')
    sst_var_ECCO = sst_ECCO.var(dim='time')
   
    # Scale the SST tendency by the ratio of MXL temp tendency / SST tendency from ECCO (as an esimate for a more physical result)
    Tmxlfrac = Tmxl_var_ECCO/sst_var_ECCO
    sst = sst*np.sqrt(Tmxlfrac)
    
    # Mask zero values (continents)    
    #sst = sst.where(~(sst==0))
    #thf = thf.where(~(thf==0))
    #Q_net_surf = Q_net_surf.where(~(Q_net_surf==0))
    
    # Mask continents
    ocean_points = ~(sst==0)
    # Mask equator when computing Ekman contribution
    #sst = sst.where(lats > 0)
    #ocean_points2 = ~(xr.ufuncs.isnan(sst))
    #ocean_points = xr.ufuncs.logical_or(ocean_points1, ocean_points2)
    sst = sst.where(ocean_points)
    sst = sst.where(np.abs(sst) < 10e5)
#    thf = thf.where(~(thf==0))
#    Q_net_surf = Q_net_surf.where(~(Q_net_surf==0))
    #Q_ek = Q_ek.where(~(Q_ek==0))
    #Q_ek = Q_ek.where(np.abs(lats) > 0)
    # Compute SST tendency
    #ocean_points1 = ~(sst==0)
    #ocean_points2 = ~(xr.ufuncs.isnan(sst))
    #ocean_points = xr.ufuncs.logical_or(ocean_points1, ocean_points2)
    #sst = sst.where(ocean_points)
    #thf = thf.where(ocean_points)
    #Q_net_surf = Q_net_surf.where(ocean_points)
    #Q_ek = Q_ek.where(~(Q_ek==0))
    #Q_ek = Q_ek.where(np.abs(lats) > 0)




    
    # Compute SST tendency
    tendsst = (sst.shift(time=-2)-sst)[:-2]
    tendsst = tendsst/(2*dt)
    
    nt = sst.shape[0]
    
    thf = thf.isel(time=slice(1,nt-1))
    Q_net_surf = Q_net_surf.isel(time=slice(1,nt-1))
    sst = sst.isel(time=slice(1,nt-1))
    
    err_sst = err_sst.isel(time=slice(1,nt-1))
    err_thf = err_thf.isel(time=slice(1,nt-1))
    
    nt = sst.shape[0]
    
    # Make sure sst tendency times match up with other fields
    tendsst.time.values = thf.time.values
    
    #Qr = Cbar*tendsst -(-thf + Q_net_surf) - Q_ek
    Qr = Cbar*tendsst -(-thf + Q_net_surf) 
    Qr = Qr.transpose('time','lat','lon')
    
    Q_s = -thf + Q_net_surf
    
    #Q_s = -thf + Q_net_surf + Q_ek
    
    nlat = len(lats)
    nlon = len(lons)
    
    #timeslice = slice(0,nt)
    
    
    err_Qs = err_thf
    
    err_tendsst = 2*err_sst/(2*dt)
    
    err_Qr = Cbar*err_tendsst + err_Qs
    
    timeslice = slice(int(Tn),nt-int(Tn))

    Q_s = Q_s.isel(time=timeslice)
    Qr = Qr.isel(time=timeslice)
    tendsst = tendsst.isel(time=timeslice)
    sst = sst.isel(time=timeslice)
    err_Qs = err_Qs.isel(time=timeslice)
    err_tendsst = err_tendsst.isel(time=timeslice)
    
    # var_Qo = st2.cov(Qr,Qr)
    # var_Qs = st2.cov(Q_s,Q_s)
    # cov_QsQo = st2.cov(Qr,Q_s)
    
    Q_tot = Q_s + Qr
    
    # Compute covariance betwen SST tendency and fields
    cov_Qr = st2.cov(tendsst,Qr)
    cov_Qs = st2.cov(tendsst,Q_s)
    #cov_Qek = st2.cov(tendsst,Q_ek)
    cov_Rnet = st2.cov(tendsst, Q_net_surf)
    cov_thf = st2.cov(tendsst, -thf)
    #cov_ssttend = st2.cov(tendsst,tendsst)
    
    cov_Qr_err = st2.cov(tendsst,err_Qr) + st2.cov(err_tendsst,Qr)
    cov_Qs_err = st2.cov(tendsst,err_Qs) + st2.cov(err_tendsst,Q_s)
    
    # var_Qr = Qr.var(dim='time')
    # var_Qs = Q_s.var(dim='time')
    # var_Qtot = Q_tot.var(dim='time')
    
    # covQsQr = 0.5*(var_Qtot - var_Qr - var_Qs)
    
    # Compute lagged sst autocorrelations
    r1corrs = st2.cor(sst,sst,lagx=1)
    r2corrs = st2.cor(sst,sst,lagx=2)
    
    
    
    # Scaling factor (to convert from units of W*K/(s*m^2) to K^2)
    fac = (2*dt**2/(Cbar*(1-r2corrs)))
    
    G = fac/Cbar
    
    #MERRA-2 sometimes blows up fac...
    fac = fac.where(~(xr.ufuncs.isinf(fac)))
    
    # var_Qo_T = (fac/Cbar)*var_Qo
    # var_Qs_T = (fac/Cbar)*var_Qs
    # cov_QsQo_T = (fac/Cbar)*cov_QsQo
    
    # Compute observed SST variance
    T_var = np.var(sst,axis=0)
    
    # Compute contributions to SST variance
    T_var_Qr = fac*cov_Qr
    T_var_Qs = fac*cov_Qs
    
    # T_var_Qr = G*(var_Qr + covQsQr)
    # T_var_Qs = G*(var_Qs + covQsQr)
    #T_var_Qek = fac*cov_Qek
    #T_var_Rnet = fac*cov_Rnet
    #T_var_thf = fac*cov_thf
    #T_var_sum = T_var_Qr + T_var_Qs + T_var_Qek
    T_var_sum = T_var_Qr + T_var_Qs 
    
    #err_Tvar = err_sst.var(dim='time')
    err_Tvar_Qr = np.abs(fac*cov_Qr_err)
    err_Tvar_Qs = np.abs(fac*cov_Qs_err)
    err_Tvar_sum = err_Tvar_Qr + err_Tvar_Qs

    
    ave_T_var_Qr[jj] = st2.spatial_ave_xr(T_var_Qr, lats)
    ave_T_var_Qs[jj] = st2.spatial_ave_xr(T_var_Qs, lats)
    ave_errTvar_Qr[jj] = st2.spatial_ave_xr(err_Tvar_Qr, lats)
    ave_errTvar_Qs[jj] = st2.spatial_ave_xr(err_Tvar_Qs, lats)
    ave_errTvar_sum[jj] = st2.spatial_ave_xr(err_Tvar_sum, lats)
    #ave_T_var_thf[jj] = st2.spatial_ave_xr(T_var_thf, lats)
    #ave_T_var_Qek[jj] = st2.spatial_ave_xr(T_var_Qek, lats)
    #ave_T_var_Rnet[jj] = st2.spatial_ave_xr(T_var_Rnet, lats)
    ave_T_var_sum[jj] = st2.spatial_ave_xr(T_var_sum, lats)
    ave_T_var[jj] = st2.spatial_ave_xr(T_var,lats)
    
    print('T_var_sum', ave_T_var_sum[jj])
    print('T_var', ave_T_var[jj])
    
    
print('ave_T_var_Qr[0]', ave_T_var_Qr[0])
print('ave_T_var_Qs[0]', ave_T_var_Qs[0])

lpi = int((12*5)/4)

print('LP ave_T_var_Qr', ave_T_var_Qr[lpi])
print('LP ave_T_var_Qs', ave_T_var_Qs[lpi])


fig, axs = plot.subplots(ncols=1, nrows=2, aspect=1.2, tight=True, share=False, hratios=(3,2))
#plot.subplots(ncols=2, nrows=3)
h1=axs[0].plot(Tns/12., ave_T_var_Qs, color='C2', label=r'$Q_s$', linewidth=2, zorder=5)
#plt.plot(Tns/12., ave_T_var_thf, color='C3', label=r'$THF$')
#plt.plot(Tns/12., ave_T_var_Rnet, color='C4', label=r'$R_{net}$')
h2=axs[0].plot(Tns/12., ave_T_var_Qr, color='C0', label=r'$Q_o$', linewidth=2, zorder=5)
#plt.plot(Tns/12., ave_T_var_Qek, color='C1', label=r'$Q_{ek}$')
h3=axs[0].plot(Tns/12., ave_T_var_sum, color='k', label=r'$\sigma^2_T$', linewidth=2, zorder=5)
errup = ave_T_var_sum+ave_errTvar_sum
errlow = ave_T_var_sum-ave_errTvar_sum
errup_Qs = ave_T_var_Qs-ave_errTvar_Qs
errlow_Qs = ave_T_var_Qs+ave_errTvar_Qs
errup_Qr = ave_T_var_Qr-ave_errTvar_Qr
errlow_Qr = ave_T_var_Qr+ave_errTvar_Qr
#errlow[errlow < 0] = 0
h4=axs[0].fill_between(Tns/12., errlow, errup, color='k', label=r'$\sigma^2_T$', alpha=0.3, linewidth=0, zorder=5)
h4=axs[0].fill_between(Tns/12., errlow_Qs, errup_Qs, color='C2', label=r'$\sigma^2_T$', alpha=0.3, linewidth=0, zorder=6)
h4=axs[0].fill_between(Tns/12., errlow_Qr, errup_Qr, color='C0', label=r'$\sigma^2_T$', alpha=0.3, linewidth=0, zorder=7)

#h4=axs[0].plot(Tns/12., ave_T_var, color='C5', label=r'$\sigma_T^2$')
axs[0].axhline(0, color='k', linewidth=1)

hs=[h1,h2,h3]
#Global/tropics
#axs[0].set_ylim(-0.1,0.38)
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
axs[0].set_ylim(-0.1,0.38)
#plt.ylim(-0.02,0.6)
axs[0].set_xlim(0,6)
axs[0].set_ylabel('Contribution to $\sigma^2_T$ (K$^{2}$)')
#plt.legend(loc='best')
#fig.legend(hs, loc='b', col=1)
frac_T_var_Qs = ave_T_var_Qs/ave_T_var_sum
#frac_T_var_thf = ave_T_var_thf/ave_T_var_sum
#frac_T_var_Rnet = ave_T_var_Rnet/ave_T_var_sum
frac_T_var_Qr = ave_T_var_Qr/ave_T_var_sum
#frac_T_var_Qnet = ave_T_var_Rnet/ave_T_var_sum
frac_T_var_Qek = ave_T_var_Qek/ave_T_var_sum

y0=frac_T_var_Qr
y1=y0+frac_T_var_Qs
#y2=y1+frac_T_var_Qek

yticklbls = np.array([0,0.2,0.4,0.6,0.8,1.0])

#maxs = ax1.panel('b', space=0.5, share=False)
axs[1].fill_between(Tns/12., y0, color='C0', label=r'$Q_o$', alpha=0.8, linewidth=0)
axs[1].fill_between(Tns/12., y1, y0, color='C2', label=r'$Q_s$', alpha=0.8, linewidth=0)
axs[1].set_xlabel('Filter Length (years)')
axs[1].set_ylim(0,1.0)
axs[1].set_xlim(0,6)
axs[1].set_yticks(yticklbls)
axs[1].set_ylabel('Fractional Contribution')
#plt.legend()
fig.savefig(fout + '{:s}_sstvarbudget_err_varytimefilter_{:2.0f}Nto{:2.0f}N.png'.format(dataname, latbounds[0], latbounds[1]))
plt.close(fig)




# ax2 = ax1.twinx()

# #plt.figure(figsize=(12,10))
# ax2.fill_between(Tns/12., y0, color='C0', label=r'$Q_s$', alpha=0.5, linewidth=0)
# ax2.fill_between(Tns/12., y1, y0, color='C2', label=r'$Q_o$', alpha=0.5, linewidth=0)
# #plt.fill_between(Tns/12., y2, y1, color='C1', label=r'$Q_{ek}$')
# #plt.xlabel('Filter Length (years)')
# ax2.set_ylim(0,1.0)
# #plt.xlim(0,Tns[-1]/12.)
# ax2.tick_params(axis='y', labelcolor='dimgrey')
# ax2.set_ylabel('$Q_s$ Fractional Contribution', color='dimgrey')
# ax1.patch.set_alpha(0)
# ax1.set_zorder(ax2.get_zorder()+5)
# fig.savefig(fout + '{:s}_sstvarbudget_varytimefilter_{:2.0f}Nto{:2.0f}N.png'.format(dataname, latbounds[0], latbounds[1]))
# plt.close(fig)

# ave_T_var_Qs = xr.DataArray(ave_T_var_Qs, dims={'Tn'}, coords={'Tn':Tns})
# #ave_T_var_Qs = ave_T_var_Qs.rename('ave_T_var_Qs')
# ave_T_var_Qo = xr.DataArray(ave_T_var_Qr, dims={'Tn'}, coords={'Tn':Tns})


#ave_T_var_Qo = ave_T_var_Qo.rename('ave_T_var_Qs')

#conts = xr.Dataset(ave_T_var_Qs)
#conts['ave_T_var_Qo'] = ave_T_var_Qo

#conts = xr.Dataset({'ave_T_var_Qs': (['Tn'], ave_T_var_Qs), 'ave_T_var_Qo': (['Tn'], ave_T_var_Qo)},coords={'filter_length': (['Tn'], Tns/12.)})

#conts.to_netcdf('/Users/cpatrizio/data/{:}_global_conts.nc'.format(dataname))
#plt.legend(loc='best')
#plt.savefig(fout + '{:s}_sstvarbudgetFRACfill_varytimefilter.png'.format(dataname))
#plt.close()

#SAVE_frac_T_var_Qs = ave_T_var_Qs/ave_T_var_sum
#SAVE_frac_T_var_Qr = ave_T_var_Qr/ave_T_var_sum










