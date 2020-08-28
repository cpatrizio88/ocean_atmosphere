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
import cmocean
#import seaborn as sns
#import proplot as plot


fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
#fin = '/Users/cpatrizio/data/OAFlux/'
#fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/ocean-atmosphere/localSST_global/'

fout = '/Users/cpatrizio/figures/ocean-atmosphere/localSST_global/'


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

fh = xr.open_dataset(fin + 'ncep.mixedlayerdepth.198001-201712.nc')
#fh = cdms2.open(fin + 'ECCO_mxldepth_monthly1992to2015.nc')

ssh = fssh.sshg

#dataname = 'ERAi'
#dataname = 'MERRA2'
dataname = 'OAFlux'
#dataname = 'ERA5'

#OAFlux 
fin = '/Users/cpatrizio/data/OAFlux/'
fsst =  xr.open_dataset(fin + 'oaflux_ts_1980to2017.nc')
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
sst = fsst.tmpsf
lhf = fthf.lhtfl
shf = fthf.shtfl
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT


thf = lhf + shf
Q_net_surf = LW_net_surf + SW_net_surf

if dataname == 'ERA5':
    sst=sst.rename({'latitude':'lat', 'longitude':'lon'})
    sst = sst.sortby('lat',ascending=True)
lats = sst.lat
lons = sst.lon
if lons.max() <= 180:
    sst = sst.assign_coords(lon=(sst.lon % 360)).roll(lon=((sst.shape[2] // 2)-1))
 
sst = sst.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))


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


thf = thf[:Q_net_surf.shape[0],:,:]
sst = sst[:Q_net_surf.shape[0],:,:]
#h = h[:Q_net_surf.shape[0],:,:]
ssh = ssh[:Q_net_surf.shape[0],:,:]
taux = taux[:Q_net_surf.shape[0],:,:]
tauy = tauy[:Q_net_surf.shape[0],:,:]
ps = ps[:Q_net_surf.shape[0],:,:]
#SW_net_surf = SW_net_surf[:Q_net_surf.shape[0],:,:]


months = np.arange(sst.shape[0])
tyears = 1980 + months/12.
if dataname == 'ERA5':
    dates = pd.date_range('1979-01-01', periods=len(months), freq='MS')
else:
    dates = pd.date_range('1980-01-01', periods=len(months), freq='MS')

sst.time.values = dates
thf.time.values = dates
#h.time.values = dates
Q_net_surf.time.values = dates
#SW_net_surf.time.values = dates
#ssh.time.values = dates
taux.time.values = dates
tauy.time.values = dates
#ps.time.values = dates

ii=-1

#if dataname == 'MERRA2':

sst = sst[tskip:ii,:,:]
#h = h[tskip:ii,:,:]
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
corr=False
lterm=True
drawmaps=True
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

#regridder = xe.Regridder(ssh, ds_out, 'bilinear', reuse_weights=True)
#ssh = regridder(ssh)

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

#regridder = xe.Regridder(ps, ds_out, 'bilinear', reuse_weights=True)
#ps  = regridder(ps)

taux_clim = taux.mean(dim='time')
tauy_clim = tauy.mean(dim='time')


rho = 1000
c_p = 3850
dt = 30*3600*24
C = rho*c_p*h
Cbar = C.mean(dim='time')

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

u_ek = (1/(rho*f3D*hbar3D))*tauy
v_ek = -(1/(rho*f3D*hbar3D))*taux

u_ek =  xr.DataArray(u_ek, dims=sst.dims, coords=sst.coords) 
v_ek =  xr.DataArray(v_ek, dims=sst.dims, coords=sst.coords) 

u_ek_clim = u_ek.mean(dim='time')
v_ek_clim = v_ek.mean(dim='time')

#u_ekT = u_ek*sst
#v_ekT = v_ek*sst

#dekTdx_temp = u_ekT.diff(dim='lon',n=1)/dx3D 
#dekTdy_temp = v_ekT.diff(dim='lat',n=1)/dy

#dekdx_temp = u_ek.diff(dim='lon', n=1)/dx3D
#dekdy_temp = v_ek.diff(dim='lat', n=1)/dy


#dekdx = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])
#dekdy = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

#dekTdx = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])
#dekTdy = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

#dekTdx.values[:,:,1:] = dekTdx_temp
#dekTdy.values[:,1:,:] = dekTdy_temp

#dekdx.values[:,:,1:] = dekdx_temp
#dekdy.values[:,1:,:] = dekdy_temp

#div_ek = dekdx + dekdy

#w_ek_temp = hbar3D*div_ek

#div_ekT = dekTdx + dekTdy

adv_ek = u_ek*dSSTdx + v_ek*dSSTdy

Q_ek = -Cbar*adv_ek

#Q_ek_f = -Cbar*(div_ekT)

#Q_ek_v = -Cbar*sst*div_ek

#w_ek = xr.DataArray(np.nan*np.zeros([nt,nlat,nlon]),
#                     coords={'time': times, 'lat': lats, 'lon':lons},dims=['time', 'lat','lon'])

#w_ek.values = w_ek_temp
#
eqi = np.where(lats==0)
Q_ek = Q_ek.transpose('time','lat','lon')
Q_ek.values[:,eqi,:] = 0

#w_ek_mean = w_ek.mean(dim='time')
#
Q_ek_mean = Q_ek.mean(dim='time')

#Q_ek_v_mean = Q_ek_v.mean(dim='time')

#Q_ek_f_mean = Q_ek_f.mean(dim='time')
#
#Q_ek = Q_ek.where(np.abs(lats) > 0)

# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    Q_ek,Q_ek_clim= st.anom(Q_ek)
    #Q_ek_f,Q_ek_f_clim = st.anom(Q_ek_f)

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
#Q_ek = Q_ek.where(~(Q_ek==0))
#Q_ek = Q_ek.where(np.abs(lats) > 0)
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

#Qr = Cbar*tendsst - (-thf + Q_net_surf) - Q_ek

Qr = Cbar*tendsst -(-thf + Q_net_surf)
Qr = Qr.transpose('time','lat','lon')

Qr_mean = Qr.mean(dim='time')

#Q_s = -thf + Q_net_surf + Q_ek

Q_s = -thf + Q_net_surf 

T_Qs = (Q_s/Cbar).cumsum(dim='time')*dt
T_Qo = (Qr/Cbar).cumsum(dim='time')*dt

T_sum = T_Qs + T_Qo

var_T_Qs = T_Qs.var(dim='time')
var_T_Qo = T_Qo.var(dim='time')

cov_T_Qs_T_Qo = st2.cov(T_Qs,T_Qo)





nlat = len(lats)
nlon = len(lons)

reflon=320
reflat=50

plt.figure(figsize=(16,10))
plt.plot(T_Qs.time, T_Qs.sel(lon=reflon,lat=reflat),label=r'$T_{Q_s}$',color='C0')
plt.plot(T_Qo.time, T_Qo.sel(lon=reflon,lat=reflat),label=r'$T_{Q_o}$',color='C2')
plt.plot(T_sum.time, T_sum.sel(lon=reflon,lat=reflat),label='sum',color='k')
plt.plot(sst.time, sst.sel(lon=reflon,lat=reflat), label='T', color='C5')
plt.legend(loc='upper right')
plt.xlabel('Time (year)')
plt.ylabel('Temperature (K)')
plt.savefig(fout + '{:s}_sst_timeseries_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
#plt.show()
#plt.plot()

plt.figure(figsize=(16,10))
plt.plot(Q_s.time, Q_s.sel(lon=reflon,lat=reflat),label=r'${Q_s}$',color='C0')
plt.plot(Qr.time, Qr.sel(lon=reflon,lat=reflat),label=r'${Q_o}$',color='C2')
plt.plot(tendsst.time, (Cbar*tendsst).sel(lon=reflon,lat=reflat),label=r'$C_o\frac{\partial T}{\partial t}$',color='k')
#plt.plot(sst.time, sst.sel(lon=reflon,lat=reflat), label='T', color='C5')
plt.legend(loc='upper right')
plt.xlabel('Time (year)')
plt.ylabel('Heat Flux (W/m$^{2}$)')
plt.savefig(fout + '{:s}_heatbudget_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

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
mer = np.arange(-180.,180.+merdiff,merdiff)
x, y = np.meshgrid(lons, lats)


orient = 'horizontal'
if np.abs(latbounds[1] - latbounds[0]) > np.abs(lonbounds[1] - lonbounds[0]):
    orient = 'vertical'
    
    
sstvmax = 1.6
vmin=-1.0
vmax=1.0
if lowpass:
    sstvmax = 0.5
    vmin=-0.5
    vmax=0.5
#    

    
#sstcmap = cmocean.cm.plasma
#sstcmap = plt.cm.cubehelix_r
#fieldcmap = cmocean.cm.balance
fieldcmap = plt.cm.RdBu_r

sstcmap = cc.cm.CET_L17
#fieldcmap = cc.cm.CET_D1A
#fieldcmap = cc.cm.coolwarm


#if Qekplot:
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)
#    T_var_Qek=T_var_Qek.where(np.abs(lats)>0)

var_T_sum = var_T_Qs + var_T_Qo + 2*cov_T_Qs_T_Qo

mapper = Mapper()
mapper(var_T_Qs, bnds=bnds, title=r'$var(T_{Q_s})$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-40, vmax=40)
plt.savefig(fout + '{:s}_var_T_Qs_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
mapper(var_T_Qo, bnds=bnds,  title=r'$var(T_{Q_o})$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-40, vmax=40)
plt.savefig(fout + '{:s}_var_T_Qo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
mapper(cov_T_Qs_T_Qo, bnds=bnds,  title=r'$cov(T_{Q_s},T_{Q_o})$', units=r'K$^{2}$', cmap=fieldcmap, vmin=-40, vmax=40)
plt.savefig(fout + '{:s}_cov_T_Qs_T_Qo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


var_T = sst.var(dim='time')

mapper = Mapper()
mapper(var_T, bnds=bnds, title=r'$\sigma^2_T$', units=r'K$^{2}$', cmap=sstcmap, vmin=0, vmax=sstvmax)
plt.savefig(fout + '{:s}_var_T_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(var_T_Qs+cov_T_Qs_T_Qo, bnds=bnds, title=r'$Q_s$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
plt.savefig(fout + '{:s}_var_T_Qs+cov_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
mapper(var_T_Qo+cov_T_Qs_T_Qo, bnds=bnds, title=r'$Q_o$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
plt.savefig(fout + '{:s}_var_T_Qo+cov_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(var_T_sum, bnds=bnds, title=r'SUM', units=r'K$^{2}$', cmap=sstcmap, vmin=0, vmax=sstvmax)
plt.savefig(fout + '{:s}_var_T_sum_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

var_T = sst.var(dim='time')

mapper = Mapper()
mapper(var_T, bnds=bnds, title=r'$\sigma^2_T$', units=r'K$^{2}$', cmap=sstcmap, vmin=0, vmax=sstvmax)
plt.savefig(fout + '{:s}_var_T_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
















