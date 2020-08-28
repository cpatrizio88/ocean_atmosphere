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


#dataname = 'ERAi'
#dataname = 'MERRA2'
dataname = 'OAFlux'

#OAFlux 
fin = '/Volumes/GoogleDrive/My Drive/data_drive/OAFlux/'
fsstoa =  xr.open_dataset(fin + 'oaflux_ts_1980to2017.nc')
fthf =   xr.open_dataset(fin + 'oaflux_thf_1980to2017.nc')

#ECCO
fin = '/Volumes/GoogleDrive/My Drive/data_drive/ECCO/'
#ft= xr.open_dataset(fin + 'ECCO_theta_monthly1992to2015.nc')
fh = xr.open_dataset(fin + 'ECCO_mxldepth_monthly1992to2015.nc')
#fTmxlfrac = xr.open_dataset(fin + 'ECCO_Tmxlfrac.nc')

fsst_ECCO = xr.open_dataset(fin + 'ecco_SST.nc')
fTmxl_ECCO = xr.open_dataset(fin + 'ecco_T_mxl.nc')

#fsst = fsst.rename({'__xarray_dataarray_variable__':'Ts'})
#fTmxl = fTmxl.rename({'__xarray_dataarray_variable__':'Tmxl'})

sst_ECCO = fsst_ECCO.Ts
Tmxl_ECCO = fTmxl_ECCO.Tmxl




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


#h = h.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
hbar = h.mean(dim='time')
#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201712.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')


matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'axes.titlesize': 30})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 3})
matplotlib.rcParams.update({'legend.fontsize': 32})
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
#lonbounds = [315,335]
#latbounds = [42,52]

latbounds = [38,50]
lonbounds = [305,335]


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

#OAFlux
sst = fsstoa.tmpsf
lhf = fthf.lhtfl
shf = fthf.shtfl
LW_net_surf = radfile.LWGNT
SW_net_surf = radfile.SWGNT


#sst = fsst('sst')
#sst = fsst.tmpsf
#sst = fsst('TSKINWTR')
lats = sst.lat
sst = sst.sel(lat=slice(minlat,maxlat),lon=slice(minlon,maxlon))
#sst = sst[tskip:nt_ps,:]

#cE = fcE('CDH')
#cD = fcD('CN')

#lhf = fsst('EFLUXWTR')
#shf = fsst('HFLUXWTR')
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

#lhf = fthf.lhtfl
#shf = fthf.shtfl
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
#ssh = ssh[:Q_net_surf.shape[0],:,:]
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
    taux = taux.sel(time=slice('1992-02-01','2015-12-01'))
    tauy = tauy.sel(time=slice('1992-02-01','2015-12-01'))
    #h = h.sel(time=slice('1992-02-01','2015-12-01'))


ii=-1

#if dataname == 'MERRA2':

sst = sst[tskip:ii,:,:]
h = h[tskip:ii,:,:]
#ssh = ssh[tskip:ii,:,:]
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

# regridder = xe.Regridder(ssh, ds_out, 'bilinear', reuse_weights=True)
# ssh = regridder(ssh)

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


regridder = xe.Regridder(ps, ds_out, 'bilinear', reuse_weights=True)

regridder = xe.Regridder(sst_ECCO, ds_out, 'bilinear', reuse_weights=True)
sst_ECCO  = regridder(sst_ECCO)
#
regridder = xe.Regridder(Tmxl_ECCO, ds_out, 'bilinear', reuse_weights=True)
Tmxl_ECCO  = regridder(Tmxl_ECCO)

#ps  = regridder(ps)

taux_clim = taux.mean(dim='time')
tauy_clim = tauy.mean(dim='time')


# Scale the SST anomalies by the ratio of the mixed layer temperature variance to the SST in ECCO 
# This is to account for lower variability when averaged over the entire mixed layer
#sst = sst*Tmxlfrac


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
    sst_ECCO,sst_ECCO_clim= st.anom(sst_ECCO)
    Tmxl_ECCO,Tmxl_ECCO_clim= st.anom(Tmxl_ECCO)
    #Q_ek_f,Q_ek_f_clim = st.anom(Q_ek_f)

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
 
  
 Q_ek = Q_ek.fillna(0.)    
 Q_ek = xr.DataArray(signal.detrend(Q_ek, axis=0), dims=Q_ek.dims, coords=Q_ek.coords) 

 
 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)

delTn = 4
Tnmax = 6*12
Tns = np.arange(0,Tnmax+delTn,delTn)

Tns = Tns*1.

##
lagmax = 12
lagstep = 2
lags = np.arange(-lagmax,lagmax+2*lagstep, lagstep)
#lags=-lags


sst_raw = sst
thf_raw = thf
Q_net_surf_raw = Q_net_surf
Q_ek_raw = Q_ek
Tmxl_ECCO_raw = Tmxl_ECCO
sst_ECCO_raw = sst_ECCO

avelagcorrs = np.zeros((len(Tns), len(lags)))


for jj, Tn in enumerate(Tns):
    
    cutoff = 1./Tn
    
    print('Tn', Tn)
    
    if Tn > 0:
        sst = st.butter_lowpass_filter_xr(sst_raw, cutoff, fs, order)
        thf = st.butter_lowpass_filter_xr(thf_raw, cutoff, fs, order)
        Q_net_surf = st.butter_lowpass_filter_xr(Q_net_surf_raw, cutoff, fs, order)
        Q_ek = st.butter_lowpass_filter_xr(Q_ek_raw, cutoff, fs, order)
        sst_ECCO = st.butter_lowpass_filter_xr(sst_ECCO_raw, cutoff, fs, order)
        Tmxl_ECCO = st.butter_lowpass_filter_xr(Tmxl_ECCO_raw, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)
        
    Tmxl_var_ECCO = Tmxl_ECCO.var(dim='time')
    sst_var_ECCO = sst_ECCO.var(dim='time')
    
    #sst_bak = sst

    # Scale the SST tendency by the ratio of MXL temp tendency / SST tendency from ECCO (as an esimate for a more physical result)
    Tmxlfrac = Tmxl_var_ECCO/sst_var_ECCO
    sst = sst*np.sqrt(Tmxlfrac)
    
    #sst_bak = sst


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
    
    #Qr = Cbar*tendsst -(-thf + Q_net_surf) - Q_ek
    Qr = Cbar*tendsst -(-thf + Q_net_surf)
    Qr = Qr.transpose('time','lat','lon')
    
    #Q_s = -thf + Q_net_surf + Q_ek
    Q_s = -thf + Q_net_surf 
    
    nlat = len(lats)
    nlon = len(lons)
    
    lagcorrs = np.zeros((len(lags), nlat, nlon))
    
    # Edit this to change field
    field = Q_s
    fieldname = 'Qs'
    fieldtitle = r'$Q_s$'
    color='C0'
    if corr:
        units = ''
    else:
        units = r'W/m$^2$'
    
    for ll, lag in enumerate(lags):
        
        #print('lag',lag)
        
        print 
        
        if corr:
           lagcorrs[ll,:] = st2.cor(sst,field,lagx=-lag)
        else:
           # Standardize SST
           sst = (sst - sst.mean(dim='time'))/sst.std(dim='time')
           lagcorrs[ll,:], intercept = st2.reg(sst,field,lagx=-lag)
        
    avelagcorrs[jj,:] = st2.spatial_ave(np.ma.array(lagcorrs, mask = np.isnan(lagcorrs)), lats)
        

# Plotting
    
vmin=-0.6
vmax=0.6
cbstep=0.2
if not(corr):
   vmin=-15.0
   vmax=15.0
   cbstep=5.0
   
lagoffset = np.diff(lags)[0]/2.

nlags = len(lags)

#if fsave == 'THF':
#    oceandyni = np.where(np.ma.mean(avelagcorrs_new[:,:nlags/2-1],axis=1) < 0)
#elif fsave == 'negTHF':
#    oceandyni = np.where(np.ma.mean(avelagcorrs_new[:,:nlags/2-1],axis=1) > 0)
#
#if oceandyni[0].size:
#    threshi = np.min(oceandyni)
#    thresh = Tns_new[threshi]
#else:
#    thresh = Tns_new[-1]

#lags = lags[:-1]

lagss, Tnss = np.meshgrid(lags-lagoffset, Tns)
        
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

ticks = np.round(np.arange(vmin,vmax+cbstep,cbstep),2)
ticklbls = np.round(np.arange(vmin,vmax+cbstep,cbstep),2)
ticklbls[ticklbls == -0.00] = 0.00
              
fig=plt.figure(figsize=(12,6))
ax = plt.gca()
h=plt.pcolor(lagss, Tnss, avelagcorrs, cmap=fieldcmap, vmin=vmin, vmax=vmax)
cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(units))
cb.set_ticks(ticks)
ax.set_xticks(lags[::2])
ax.set_xticklabels(lags[::2])
ax.set_xlabel('Lag (months)')
ax.set_ylabel('Filter Length (months)')
plt.title('Correlation between $Q_s$ and SST')
if corr:
    plt.savefig(fout + '{:s}_sst{:s}corr_varyfilter_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, latbounds[0], latbounds[1], str(detr)[0]))
else:
    plt.savefig(fout + '{:s}_sst{:s}regr_varyfilter_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

fig=plt.figure(figsize=(8,6))
ax = plt.gca()
plt.plot(lagss[0,:],avelagcorrs[0,:], color='k')
plt.axhline(0,color='k', linewidth=1)
plt.axvline(0,color='k', linewidth=1)
#plt.ylabel('Correlation')
plt.ylim(-0.55,0.55)
ax.set_xticks(lags[::2])
ax.set_xticklabels(lags[::2])
ax.set_xlabel('Lag (months)')
ax.set_xlim(-12,12)
#ax.set_ylabel('{:s} Correlation'.format(fieldtitle))
if corr:
    plt.savefig(fout + '{:s}_sst{:s}corr_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, latbounds[0], latbounds[1], str(detr)[0]))
else:
    plt.savefig(fout + '{:s}_sst{:s}regr__{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

lpi = int(60/delTn)

fig=plt.figure(figsize=(8,6))
ax = plt.gca()
plt.plot(lagss[0,:],avelagcorrs[lpi,:], color='k')
#plt.ylabel('Correlation')
#plt.plot(lagss[0,:],avelagcorrs[lpi,:], label=r'$Q_{o,adv}$', color='C1')
#plt.plot(lagss[0,:],avelagcorrs[lpi,:], label=r'$Q_{o.dif}$', color='C3')
#plt.plot(lagss[0,:],avelagcorrs[lpi,:], label='$Q_o$', color='C0')
#plt.legend()
plt.axhline(0,color='k', linewidth=1)
plt.axvline(0,color='k', linewidth=1)
#plt.legend()
plt.ylim(-0.55,0.55)
ax.set_xticks(lags[::2])
ax.set_xticklabels(lags[::2])
ax.set_xlabel('Lag (months)')
ax.set_xlim(-12,12)
#ax.set_ylabel('{:s} Correlation'.format(fieldtitle))
if corr:
    plt.savefig(fout + '{:s}_sst{:s}corr_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, Tnss[lpi,0]/12., latbounds[0], latbounds[1], str(detr)[0]))
else:
    plt.savefig(fout + '{:s}_sst{:s}regr_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, fieldname, Tnss[lpi,0]/12., latbounds[0], latbounds[1], str(detr)[0]))
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













