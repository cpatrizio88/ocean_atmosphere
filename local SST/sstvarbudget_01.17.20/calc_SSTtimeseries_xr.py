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
dataname = 'MERRA2'
#dataname = 'OAFlux'
#dataname = 'ERA5'

#OAFlux 
#fin = '/Users/cpatrizio/data/OAFlux/'
#fin_rad = '/Users/cpatrizio/data/ISCCP/'
#fsst =  xr.open_dataset(fin + 'oaflux_ts_1980to2017.nc')
#fthf =   xr.open_dataset(fin + 'oaflux_thf_1980to2017.nc')
#lwfile = xr.open_dataset(fin_rad + 'ISCCP_lw_1983to2009.nc')
#swfile = xr.open_dataset(fin_rad + 'ISCCP_sw_1983to2009.nc')

#ERA5
#fin = '/Users/cpatrizio/data/ERA5/'
#
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
sst = fsst.TSKINWTR
lhf = fthf.EFLUX
shf = fthf.HFLUX
#lhf = fsst.EFLUXWTR
#shf = fsst.HFLUXWTR

thf = lhf + shf

#OAFlux
#sst = fsst.tmpsf
#lhf = fthf.lhtfl
#shf = fthf.shtfl
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT
Q_net_surf = LW_net_surf + SW_net_surf


if dataname == 'ERA5':
    sst=sst.rename({'latitude':'lat', 'longitude':'lon'})
    sst = sst.sortby('lat',ascending=True)
lats = sst.lat
lons = sst.lon
if lons.max() <= 180:
    sst = sst.assign_coords(lon=(sst.lon % 360)).roll(lon=((sst.shape[2] // 2)-1))
 
sst = sst.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
#sst = sst[tskip:nt_ps,:]


thf = lhf + shf

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
#taux.time.values = dates
#tauy.time.values = dates
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
    cstep_lat=1.0
    cstep_lon=1.0
if dataname == 'MERRA2':
    cstep_lat = 1.0
    cstep_lon= 1.0
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

regridder = xe.Regridder(thf, ds_out, 'bilinear', reuse_weights=True)
thf = regridder(thf)

regridder = xe.Regridder(Q_net_surf, ds_out, 'bilinear', reuse_weights=True)
Q_net_surf  = regridder(Q_net_surf)

regridder = xe.Regridder(hbar, ds_out, 'bilinear', reuse_weights=True)
hbar = regridder(hbar)

regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
taux  = regridder(taux)

regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
tauy  = regridder(tauy)

#regridder = xe.Regridder(ssh, ds_out, 'bilinear', reuse_weights=True)
#ssh = regridder(ssh)

#regridder = xe.Regridder(ps, ds_out, 'bilinear', reuse_weights=True)
#ps  = regridder(ps)
rho = 1000
c_p = 3850
dt = 30*3600*24
C = rho*c_p*h
Cbar = C.mean(dim='time')

##Ekman transport is c/f*(-tauy*dSST/dx + taux*dSST/dy)
##c is specific heat capacity of seawater == 3850 J/(kg C)
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

Q_ek = -Cbar*(u_ek*dSSTdx + v_ek*dSSTdy)

#eqi = np.where(lats==0)
#Q_ek = Q_ek.transpose('time','lat','lon')
#Q_ek.values[:,eqi,:] = 0

Q_ek_mean = Q_ek.mean(dim='time')

#Q_ek[]

#Q_ek = Q_ek.where(np.abs(lats) > 0)

# Compute monthly anomaly
if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    #Q_ek,Q_ek_clim= st.anom(Q_ek)

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

 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)

delTn = 4
Tnmax = 6*12
Tns = np.arange(0,Tnmax+delTn,delTn)

Tns = Tns*1.


ave_T_var_Qek = np.zeros((len(Tns)))    
ave_T_var_Qs = np.zeros((len(Tns)))
ave_T_var_Qr = np.zeros((len(Tns)))
ave_T_var_sum = np.zeros((len(Tns)))
ave_T_var = np.zeros((len(Tns)))

sst_raw = sst
thf_raw = thf
Q_net_surf_raw = Q_net_surf


for jj, Tn in enumerate(Tns):
    
    cutoff = 1/Tn
    
    print('Tn',Tn)
    
    if Tn > 0:
        sst = st.butter_lowpass_filter_xr(sst_raw, cutoff, fs, order)
        thf = st.butter_lowpass_filter_xr(thf_raw, cutoff, fs, order)
        Q_net_surf = st.butter_lowpass_filter_xr(Q_net_surf_raw, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)
    
    
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
#    thf = thf.where(~(thf==0))
#    Q_net_surf = Q_net_surf.where(~(Q_net_surf==0))
    #Q_ek = Q_ek.where(~(Q_ek==0))
    #Q_ek = Q_ek.where(np.abs(lats) > 0)
    # Compute SST tendency


    
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
    
    Qr = Cbar*tendsst -(-thf + Q_net_surf) - Q_ek
    #Qr = Cbar*tendsst -(-thf + Q_net_surf) 
    Qr = Qr.transpose('time','lat','lon')
    
    Q_s = -thf + Q_net_surf
    
    #Q_s = -thf + Q_net_surf + Q_ek
    
    nlat = len(lats)
    nlon = len(lons)
    
    # Compute covariance betwen SST tendency and fields
    cov_Qr = st2.cov(tendsst,Qr)
    cov_Qs = st2.cov(tendsst,Q_s)
    cov_Qek = st2.cov(tendsst,Q_ek)
    #cov_ssttend = st2.cov(tendsst,tendsst)
    
    # Compute lagged sst autocorrelations
    r1corrs = st2.cor(sst,sst,lagx=1)
    r2corrs = st2.cor(sst,sst,lagx=2)
    
    # Scaling factor (to convert from units of W*K/(s*m^2) to K^2)
    fac = (2*dt**2/(Cbar*(1-r2corrs)))
    
    #MERRA-2 sometimes blows up fac...
    fac = fac.where(~(xr.ufuncs.isinf(fac)))
    
    # Compute observed SST variance
    T_var = np.var(sst,axis=0)
    
    # Compute contributions to SST variance
    T_var_Qr = fac*cov_Qr
    T_var_Qs = fac*cov_Qs
    T_var_Qek = fac*cov_Qek
    T_var_sum = T_var_Qr + T_var_Qs + T_var_Qek
    #T_var_sum = T_var_Qr + T_var_Qs 
    
    ave_T_var_Qr[jj] = st2.spatial_ave_xr(T_var_Qr, lats)
    ave_T_var_Qs[jj] = st2.spatial_ave_xr(T_var_Qs, lats)
    ave_T_var_Qek[jj] = st2.spatial_ave_xr(T_var_Qek, lats)
    ave_T_var_sum[jj] = st2.spatial_ave_xr(T_var_sum, lats)
    ave_T_var[jj] = st2.spatial_ave_xr(T_var,lats)
    
    print('T_var_sum', ave_T_var_sum[jj])
    print('T_var', ave_T_var[jj])

plt.figure(figsize=(12,10))
plt.plot(Tns/12., ave_T_var_Qs, color='C0', label=r'$Q_s$')
plt.plot(Tns/12., ave_T_var_Qr, color='C2', label=r'$Q_o$')
plt.plot(Tns/12., ave_T_var_Qek, color='C1', label=r'$Q_{ek}$')
plt.plot(Tns/12., ave_T_var_sum, color='k', label=r'sum')
plt.plot(Tns/12., ave_T_var, color='C5', label=r'$\sigma_T^2$')
plt.axhline(0, color='k', linewidth=1)
plt.ylim(-0.02,0.38)
plt.xlabel('Filter Length (years)')
plt.ylabel('SST Variance (K$^{2}$)')
plt.legend(loc='best')
plt.savefig(fout + '{:s}_sstvarbudget_varytimefilter_{:2.0f}Nto{:2.0f}N.png'.format(dataname, latbounds[0], latbounds[1]))
plt.close()

frac_T_var_Qs = ave_T_var_Qs/ave_T_var_sum
frac_T_var_Qr = ave_T_var_Qr/ave_T_var_sum
frac_T_var_Qek = ave_T_var_Qek/ave_T_var_sum

#y0=frac_T_var_Qs
#y1=y0+frac_T_var_Qr
#y2=y1+frac_T_var_Qek

plt.figure(figsize=(12,10))
plt.plot(Tns/12., frac_T_var_Qs, color='C0', label=r'$Q_s$')
plt.plot(Tns/12., frac_T_var_Qr, color='C2', label=r'$Q_o$')
plt.plot(Tns/12., frac_T_var_Qek, color='C1', label=r'$Q_{ek}$')
plt.xlabel('Filter Length (years)')
plt.ylim(0,1.0)
plt.ylabel('Fractional Contribution To SST Variance')
plt.legend(loc='best')
plt.savefig(fout + '{:s}_sstvarbudgetFRAC_varytimefilter_{:2.0f}Nto{:2.0f}N.png'.format(dataname, latbounds[0], latbounds[1]))
plt.close()

#plt.figure(figsize=(12,10))
#plt.fill_between(Tns/12., y0, color='C0', label=r'$Q_s$')
#plt.fill_between(Tns/12., y1, y0, color='C2', label=r'$Q_r$')
#plt.fill_between(Tns/12., y2, y1, color='C1', label=r'$Q_{ek}$')
#plt.xlabel('Filter Length (years)')
#plt.ylim(0,1.0)
#plt.xlim(0,Tns[-1]/12.)
#plt.ylabel('Fractional Contribution To SST Variance')
#plt.legend(loc='best')
#plt.savefig(fout + '{:s}_sstvarbudgetFRACfill_varytimefilter.png'.format(dataname))
#plt.close()












