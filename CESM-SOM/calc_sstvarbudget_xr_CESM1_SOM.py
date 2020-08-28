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

#fin = '/Volumes/GoogleDrive/My Drive/data_drive/MERRA2/'

fin = '/Users/cpatrizio/data/CESM-SOM/'

fin_full = '/Users/cpatrizio/data/CESM2/'

fout = '/Users/cpatrizio/figures_arc/'


#MERRA-2
#fsstM2 =  xr.open_dataset(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
#fthf = xr.open_dataset(fin + 'MERRA2_thf_monthly1980to2017.nc')
#fSLP = xr.open_dataset(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = xr.open_dataset(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = xr.open_dataset(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
#fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
#fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')
#ftau = xr.open_dataset(fin + 'MERRA2_tau_monthly1980to2019.nc')
#fssh = xr.open_dataset(fin + 'ncep.ssh.198001-201912.nc')
#fseaice = xr.open_dataset(fin + 'MERRA2_seaice_monthly1980to2019.nc')

#fh = xr.open_dataset(fin + 'ncep.mixedlayerdepth.198001-201712.nc')

#dataname = 'ERAi'
#dataname = 'MERRA2'
#dataname = 'OAFlux'
#dataname = 'ERA5'
#dataname = 'ECCO'
dataname = 'CESM-SOM'

#ECCO
# fin_ECCO = '/Volumes/GoogleDrive/My Drive/data_drive/ECCO/'
# #ft= xr.open_dataset(fin + 'ECCO_theta_monthly1992to2015.nc')
# fh_ECCO = xr.open_dataset(fin_ECCO + 'ECCO_mxldepth_interp_1992to2015.nc')
# #fTmxlfrac = xr.open_dataset(fin + 'ECCO_Tmxlfrac.nc')

# h = fh_ECCO.MXLDEPTH
# #theta = ft.THETA

# time = h.tim
# lats = h.lat[:,0]
# lons = h.lon[0,:]
# #z = theta.dep
# #z = z.rename({'i2':'k'})

# h.i1.values = h.tim.values[:]
# h.i2.values = h.lat.values[:,0]
# h.i3.values = h.lon.values[0,:]

# h = h.drop('lat')
# h = h.drop('lon')
# h = h.drop('tim')

# h = h.rename({'i1':'time','i2': 'lat', 'i3':'lon'})

#fsst = xr.open_dataset(fin + 'ecco_SST.nc')
#fTmxl = xr.open_dataset(fin + 'ecco_T_mxl.nc')

#fsst = fsst.rename({'__xarray_dataarray_variable__':'Ts'})
#fTmxl = fTmxl.rename({'__xarray_dataarray_variable__':'Tmxl'})

#sst_ECCO = fsst.Ts
#Tmxl_ECCO = fTmxl.Tmxl

#Tmxlfrac = Tmxl_ECCO/sst_ECCO

#Tmxlfrac = fTmxlfrac.Tmxlfrac

#ssh = fssh.sshg


#OAFlux 
#fin = '/Volumes/GoogleDrive/My Drive/data_drive/OAFlux/'
#fsstoa =  xr.open_dataset(fin + 'oaflux_ts_1980to2017.nc')
#fthf =   xr.open_dataset(fin + 'oaflux_thf_1980to2017.nc')

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

#CESM2
fsst = xr.open_dataset(fin + 'e.e11.E1850C5CN.f09_g16.001.cam.h0.TS.090001-100112.nc')
fh = xr.open_dataset(fin +  'pop_frc.b.e11.B1850LENS.f09_g16.pi_control.002.20190923.nc')
frad = xr.open_dataset(fin + 'e.e11.E1850C5CN.f09_g16.001.cam.h0.SRFRAD.090001-100112.nc')
fshf = xr.open_dataset(fin + 'e.e11.E1850C5CN.f09_g16.001.cam.h0.SHFLX.090001-100112.nc')
flhf = xr.open_dataset(fin + 'e.e11.E1850C5CN.f09_g16.001.cam.h0.LHFLX.090001-100112.nc')

h = fh.hblt

#h = h/100.
theta = fsst.TS
shf = fshf.SHFLX
lhf = flhf.LHFLX
Qs = shf + lhf
#Qs = fQs.SHF
#tendT = ftendT.TEND_TEMP
#theta = ft.THETA

mask = (fh.mask == 0)



#theta = theta.where(~mask)
#Qs = Qs.where(~mask)

time = theta.time
#lats = theta.ULAT[:,0]
#lons = theta.ULONG[0,:]

#lats = lats.rename({'nlat': 'lat'})
#lons = lons.rename({'nlon': 'lon'})

lats_n = fh.yc
lons_n = fh.xc

h = h.assign_coords(lon=lons_n,lat=lats_n)


#regrid onto the native CESM grid in order to use mask
ds_out = xr.Dataset({'lon':lons_n, 'lat':lats_n})

regridder = xe.Regridder(theta, ds_out, 'bilinear', reuse_weights=True)
theta = regridder(theta)
regridder.clean_weight_file()

regridder = xe.Regridder(Qs, ds_out, 'bilinear', reuse_weights=True)
Qs= regridder(Qs)
regridder.clean_weight_file()

theta = theta.where(~mask)
Qs = Qs.where(~mask)


#Q_o_temp = Q_o[0:12,:]

#To test that model is reading the forcing correctly, add a large constant Q_o anomaly in the midlatitude Pacific for months 12-24
#Q_o.loc[dict(lon=lons_Qo[(lons_Qo > 140) & (lons_Qo < 220)], lat=lats_Qo[(lats_Qo > 20) & (lats_Qo < 50)])] = 400

#Q_o[0:12,:] = Q_o_temp

lats = theta.lat
lons = theta.lon



#lats = lats.sortby(lats)

tstart = -12*37

theta = theta[tstart:-1,:]
Qs = Qs[tstart:-1,:]
#h = h[tstart:-1,:]
#tendT = tendT[tstart:-1,:]

#mxldepth_clim = h.mean(dim='time')

sst = theta

#hmean = h.mean(dim='time')
#hbar=hmean

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
latbounds = [-89.5,89.5]

lonbounds_plot = [0.5, 359.5]
latbounds_plot = [-61.5,61.5]


#lonbounds = [0.5,359.5]
#latbounds = [-65.5,65.5]


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


months_sst = np.arange(sst.shape[0])
#months_sst_interp = np.arange(sst_interp.shape[0])
months = np.arange(Qs.shape[0])
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
Qs.time.values = dates
#tendT_mxl.time.values = dates_sst

ii=-1


#True for low-pass filtering 
lowpass = False
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

Qs= Qs.transpose('time', 'nj','ni')
sst = sst.transpose('time', 'nj', 'ni')
#tendT_mxl = tendT_mxl.transpose('time', 'nlat', 'nlon')
#barbar = hbar.transpose('time', 'nlat', 'nlon')
#delz_sum = delz_sum.transpose('time', 'nlat', 'nlon')

#hbar = hbar.assign_coords(lon=lons_n,lat=lats_n)


ds_out = xe.util.grid_global(1, 1)

regridder = xe.Regridder(sst, ds_out, 'bilinear', reuse_weights=False)
sst = regridder(sst)
regridder.clean_weight_file()  # print basic regridder information.


#regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=True)
#h = regridder(h)

#regridder = xe.Regridder(Tmxlfrac, ds_out, 'bilinear', reuse_weights=True)
#Tmxlfrac = regridder(Tmxlfrac)

# regridder = xe.Regridder(thf, ds_out, 'bilinear', reuse_weights=True)
# thf = regridder(thf)
#
regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=False)
h = regridder(h)
regridder.clean_weight_file()

#regridder = xe.Regridder(hmean, ds_out, 'bilinear', reuse_weights=True)
#hmean= regridder(hmean)

regridder = xe.Regridder(Qs, ds_out, 'bilinear', reuse_weights=False)
Qs  = regridder(Qs)
regridder.clean_weight_file()

#regridder = xe.Regridder(tendT_mxl, ds_out, 'bilinear', reuse_weights=True)
#tendT_mxl  = regridder(tendT_mxl)

#regridder = xe.Regridder(delz_sum, ds_out, 'bilinear', reuse_weights=True)
#delz_sum  = regridder(delz_sum)

# regridder = xe.Regridder(taux, ds_out, 'bilinear', reuse_weights=True)
# taux  = regridder(taux)
# #
# regridder = xe.Regridder(tauy, ds_out, 'bilinear', reuse_weights=True)
# tauy  = regridder(tauy)

# regridder = xe.Regridder(sst_ECCO, ds_out, 'bilinear', reuse_weights=True)
# sst_ECCO  = regridder(sst_ECCO)
#
# regridder = xe.Regridder(Tmxl_ECCO, ds_out, 'bilinear', reuse_weights=True)
# Tmxl_ECCO  = regridder(Tmxl_ECCO)


lons = sst.lon[0,:]
lats = sst.lat[:,0]

sst = sst.drop({'lon','lat'})
#sst['lon'] = lons
#sst['lat'] = lats
sst = sst.assign_coords(y=lats,x=lons)
sst = sst.rename({'y':'lat','x':'lon'})
               

Qs = Qs.drop({'lon','lat'})
Qs = Qs.assign_coords(x=lons,y=lats)
Qs = Qs.rename({'y':'lat','x':'lon'})

h = h.drop({'lon','lat'})
h = h.assign_coords(x=lons,y=lats)
h = h.rename({'y':'lat','x':'lon'})


sst = sst.assign_coords(lon=(sst.lon % 360)).roll(lon=((sst.shape[2] // 2)))
#tendT_mxl = tendT_mxl.assign_coords(lon=(tendT_mxl.lon % 360)).roll(lon=((tendT_mxl.shape[2] // 2)))
Qs = Qs.assign_coords(lon=(Qs.lon % 360)).roll(lon=((Qs.shape[2] // 2)))
#delz_sum = delz_sum.assign_coords(lon=(delz_sum.lon % 360)).roll(lon=((delz_sum.shape[1] // 2)))
h = h.assign_coords(lon=(h.lon % 360)).roll(lon=((h.shape[2] // 2)))

#interpolate missing values at lon=320.5
#delz_sum[:,320] = (delz_sum[:,319] + delz_sum[:,321])/2.
h[:,:,320] = (h[:,:,318] + h[:,:,321])/2.
h[:,:,319] = (h[:,:,318] + h[:,:,320])/2.

sst[:,:,320] = (sst[:,:,318] + sst[:,:,321])/2.
sst[:,:,319] = (sst[:,:,318] + sst[:,:,320])/2.

Qs[:,:,320] = (Qs[:,:,318] + Qs[:,:,321])/2.
Qs[:,:,319] = (Qs[:,:,318] + Qs[:,:,320])/2.
#tendT_mxl[:,:,320] = (tendT_mxl[:,:,319] + tendT_mxl[:,:,321])/2.



hbar = h.mean(dim='time')

rho = 1025
c_p = 3850
dt = 30*3600*24
#C = rho*c_p*h

#C_anom, C_clim = st.anom(C)

#Cbar = C_clim.mean(dim='month')

Cbar = rho*c_p*hbar



# h_anom, h_clim = st.anom(h)

# h_clim_std = h_clim.std(dim='month')
# h_bar = h_clim.mean(dim='month')


# Compute monthly anomaly
if anom_flag:
    Qs,Qs_clim = st.anom(Qs)
    #thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)
    #tendT_mxl, tendT_mxl_clim = st.anom(tendT_mxl)
    #Q_ek,Q_ek_clim= st.anom(Q_ek)
    # sst_ECCO,sst_ECCO_clim= st.anom(sst_ECCO)
    # Tmxl_ECCO,Tmxl_ECCO_clim= st.anom(Tmxl_ECCO)



# Remove linear trend
if detr: 
 sst = sst.fillna(0.)    
 sst = xr.DataArray(signal.detrend(sst, axis=0), dims=sst.dims, coords=sst.coords)   

# h = h.fillna(0.)    
# h = xr.DataArray(signal.detrend(h, axis=0), dims=h.dims, coords=h.coords)   
 
 Qs = Qs.fillna(0.)    
 Qs = xr.DataArray(signal.detrend(Qs, axis=0), dims=Qs.dims, coords=Qs.coords) 
 
 #tendT_mxl = tendT_mxl.fillna(0.)    
 #tendT_mxl = xr.DataArray(signal.detrend(tendT_mxl, axis=0), dims=tendT_mxl.dims, coords=tendT_mxl.coords) 
  
 # Q_net_surf = Q_net_surf.fillna(0.)    
 # Q_net_surf = xr.DataArray(signal.detrend(Q_net_surf, axis=0), dims=Q_net_surf.dims, coords=Q_net_surf.coords) 
 
# SW_net_surf = SW_net_surf.fillna(0.)    
# SW_net_surf = xr.DataArray(signal.detrend(SW_net_surf, axis=0), dims=SW_net_surf.dims, coords=SW_net_surf.coords) 
 
 # Q_ek = Q_ek.fillna(0.)    
 # Q_ek = xr.DataArray(signal.detrend(Q_ek, axis=0), dims=Q_ek.dims, coords=Q_ek.coords) 
# 
#  
# Q_ek_f = Q_ek_f.fillna(0.)    
# Q_ek_f = xr.DataArray(signal.detrend(Q_ek_f, axis=0), dims=Q_ek_f.dims, coords=Q_ek_f.coords) 
 
#Tn = 6.
#cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
#enso = st2.spatial_ave_xr(sst.sel(lon=slice(190,240)), lats=lats.sel(lat=slice(-5,5))) 
 
#if rENSO:
#    sst = st2.regressout_x(enso, sst)
    #thf = st2.regressout_x(enso, thf)
#    Qs = st2.regressout_x(enso, Qs)
#    tendT_mxl = st2.regressout_x(enso, tendT_mxl)


#eqi = np.where(lats[1:-1]==0)
#Q_ek[:,eqi,:] = 0
#Q_ek_test = Q_ek
 

#adv_hs = -(adv_hDivH['X'] + adv_hDivH['Y'])/vol
 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn =4.*12
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

if not(lowpass):
    Tn = 0.
    
# Butterworth low-pass filter
if lowpass:
    #sst = xr.DataArray(st.butter_lowpass_filter(sst, cutoff, fs, order),dims=sst.dims,coords=sst.coords)
    #thf = xr.DataArray(st.butter_lowpass_filter(thf, cutoff, fs, order),dims=thf.dims,coords=thf.coords)
    #Q_net_surf = xr.DataArray(st.butter_lowpass_filter(Q_net_surf, cutoff, fs, order),dims=Q_net_surf.dims,coords=Q_net_surf.coords)
    sst = st.butter_lowpass_filter_xr(sst, cutoff, fs, order)
    #thf = st.butter_lowpass_filter_xr(thf, cutoff, fs, order)
    Qs = st.butter_lowpass_filter_xr(Qs, cutoff, fs, order)
    #tendT_mxl = st.butter_lowpass_filter_xr(tendT_mxl, cutoff, fs, order)
    #Q_ek = st.butter_lowpass_filter_xr(Q_ek, cutoff, fs, order)
    # sst_ECCO = st.butter_lowpass_filter_xr(sst_ECCO, cutoff, fs, order)
    # Tmxl_ECCO = st.butter_lowpass_filter_xr(Tmxl_ECCO, cutoff, fs, order)
    #Q_ek_f = st.butter_lowpass_filter_xr(Q_ek_f, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)

ocean_points = ~(sst==0)

sst = sst.where(ocean_points)
sst = sst.where(np.abs(sst) < 10e5)

Qs = Qs.where(ocean_points)
Qs = Qs.where(np.abs(Qs) < 10e5)

nt = sst.shape[0]

tendT_mxl = Qs/Cbar


Qr = Cbar*tendT_mxl - Qs

#Qr = Cbar*tendsst - Qs
Qr = Qr.transpose('time','lat','lon')

Qr_mean = Qr.mean(dim='time')
Qs_mean = Qs.mean(dim='time')

#Q_s = -thf + Q_net_surf + Q_ek

#Q_s = -thf + Q_net_surf 
Q_s = Qs

nt = sst.shape[0]
#timeslice = slice(0,nt)
timeslice = slice(int(Tn),nt-int(Tn))

Q_s = Q_s.isel(time=timeslice)
Qr = Qr.isel(time=timeslice)
#tendsst = tendsst.isel(time=timeslice)
sst = sst.isel(time=timeslice)
tendT_mxl = tendT_mxl.isel(time=timeslice)


tendT_mxl_squarebar = (tendT_mxl**2).mean(dim='time')
sst_var = sst.var(dim='time')

k = sst_var/tendT_mxl_squarebar
alpha = k/Cbar

Qr_var = Qr.var(dim='time')
Q_s_var = Q_s.var(dim='time')

covQsQo = st2.cov(Qr,Q_s)

nlat = len(lats)
nlon = len(lons)

var_Qo = st2.cov(Qr,Qr)
var_Qs = st2.cov(Q_s,Q_s)
#cov_QsQo = st2.cov(Qr,Q_s)

covQsQo_test = 0.5*((Qr + Q_s).var(dim='time') - var_Qo - var_Qs)




# Compute covariance betwen SST tendency and fields
cov_Qr = st2.cov(tendT_mxl, Qr)
cov_Qs = st2.cov(tendT_mxl, Q_s)
#cov_Qek = st2.cov(tendsst, Q_ek)
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

var_Qo_T = (G)*var_Qo
var_Qs_T = (G)*var_Qs
cov_QsQo_T = (G)*covQsQo

# Compute observed SST variance
T_var = sst.var(dim='time')

H_var = (Cbar*tendT_mxl).var(dim='time')

# Compute contributions to SST variance
T_var_Qr = alpha*cov_Qr
T_var_Qs = alpha*cov_Qs
#T_var_thf = fac*cov_thf
#T_var_Qek = fac*cov_Qek
#T_var_Qek_f = fac*cov_Qek_f
#T_var_Rnet = fac*cov_Rnet
#T_var_sum = T_var_Qr + T_var_Qs + T_var_Qek

T_var_sum = T_var_Qr + T_var_Qs

#T_var_RnetRnet = G*cov_RnetRnet
#T_var_thfRnet = G*cov_thfRnet
#T_var_QrRnet = G*cov_QrRnet
#
#T_var_thfthf = G*cov_thfthf
#T_var_Qrthf = G*cov_Qrthf
#T_var_QrQr = G*cov_QrQr


T_var_ave = st2.spatial_ave_xr(T_var,lats)
print('T_var_ave', T_var_ave)

# Approximation of SST variance by the covariance of the SST tendency
#T_var_test = (fac*Cbar)*cov_ssttend

# Plotting
bnds = [lonbounds_plot[0], lonbounds_plot[1], latbounds_plot[0], latbounds_plot[1]]

cent = (bnds[0]+bnds[1])/2.
prj = cart.crs.PlateCarree(central_longitude=cent)

# bnds[0] = bnds[0] + 1
# bnds[2] = bnds[2] + 2

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

# latbounds = [48,60]
# lonbounds = [310,330]


# latbounds = [30,50]
# lonbounds = [310,345]

# latbounds = [48,60]
# lonbounds = [310,330]

# lonbounds_box = [305,335]
# latbounds_box = [38,50]

lonbounds_box = [310,330]
latbounds_box = [48,60]



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


mapper = Mapper()
mapper(tendT_mxl_squarebar, logscale=False, log=True, bnds=bnds, title=r'$\overline{(\frac{\partial T^\prime}{\partial t})^2}$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=1e-15, vmax=1e-13)
plt.savefig(fout + '{:s}_tendTmxlvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper = Mapper()
mapper(T_var, bnds=bnds, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0, vmax=sstvmax)
plt.savefig(fout + '{:s}_Tvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(H_var, bnds=bnds, logscale=False, log=True, title=r'Mixed Layer Heat Storage Variance', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Hvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Q_s_var, bnds=bnds, logscale=False, log=True,  title=r'$\overline{Q^{\prime 2}_s}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qsvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(Qr_var, bnds=bnds, logscale=False, log=True, title=r'$\overline{Q^{\prime 2}_o}$',  units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=sstcmap, vmin=varmin, vmax=varmax)
plt.savefig(fout + '{:s}_Qrvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

lvmin=-varmax
lvmax=-varmin
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=lvmin, vmax=lvmax)

mapper = Mapper()
mapper(covQsQo, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$\overline{Q^\prime_sQ^\prime_o}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17_r, vmin=lvmin, vmax=lvmax)
plt.savefig(fout + '{:s}_covQsQo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(covQsQo_test, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$\overline{Q^\prime_sQ^\prime_o}$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17_r, vmin=lvmin, vmax=lvmax)
plt.savefig(fout + '{:s}_covQsQoTEST_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#mapper = Mapper()
#mapper(Q_s_var+covQsQo, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$Q_s$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17, vmin=10e1, vmax=10e3)
#plt.savefig(fout + '{:s}_Qsconttest_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#
#mapper = Mapper()
#mapper(Qr_var+covQsQo, logscale=False, bnds=bnds, log=False, norm=lognorm, title=r'$Q_o$', units=r'W$^2$ m$^{-4}$', cbfrac=cbfrac, cmap=cc.cm.CET_L17, vmin=10e1, vmax=10e3)
#plt.savefig(fout + '{:s}_Qoconttest_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

sstlognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0, vmax=sstvmax)
lognorm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=vmin, vmax=vmax)

mapper = Mapper()
m,ax=mapper(T_var, bnds=bnds, log=True, title=r'$\sigma_T^2$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1, linestyle='dashed', transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_Tvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(T_var_sum,  log=True, bnds=bnds, title='SUM', units=r'K$^{2}$', cbfrac=cbfrac, cmap=sstcmap, vmin=0.01, vmax=sstvmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvarSUM__{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(T_var_Qs, norm=lognorm, bnds=bnds, title='$Q_s$', units=r'K$^{2}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvarQs_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(T_var_Qr, norm=lognorm, bnds=bnds, title='$Q_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_TvarQo_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

# m,ax=mapper(var_Qo_T, norm=lognorm, bnds=bnds, title='', units=r'K$^{2}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-4.0, vmax=4.0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_varQoT_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(var_Qs_T, norm=lognorm, bnds=bnds, title='', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=-4.0, vmax=4.0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_varQsT_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# m,ax=mapper(cov_QsQo_T, norm=lognorm, bnds=bnds, title='', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=-4.0, vmax=4.0)
# if drawbox:
#     ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
# plt.savefig(fout + '{:s}_covQsQoT_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

m,ax=mapper(cov_Qs, log=False, bnds=bnds, title='', units=r'K W m$^{-2}$ s$^{-1}$', cbfrac=cbfrac, cmap=fieldcmap, vmin=-1e-5, vmax=1e-5)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_covQs_ {:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

m,ax=mapper(cov_Qr, log=False,bnds=bnds, title='', units=r'K W m$^{-2}$ s$^{-1}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=-1e-5, vmax=1e-5)
if drawbox:
    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
plt.savefig(fout + '{:s}_cov_Qr_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


#T_var_Qr_SAVE = T_var_Qr

#T_var_Qr_error = T_var_Qr - T_var_Qr_SAVE 
#
#m,ax=mapper(T_var_Qr_error, norm=lognorm, bnds=bnds, title='$Q_o$', units=r'K$^{2}$',  cbfrac=cbfrac, cmap=fieldcmap, vmin=vmin, vmax=vmax)
#if drawbox:
#    ax.add_patch(mpatches.Rectangle(xy=[x1,y1], width=(x2-x1), height=(y2-y1), facecolor='none', edgecolor='black',linewidth=1,linestyle='dashed',transform=cart.crs.PlateCarree()))
#plt.savefig(fout + '{:s}_TvarQo_error_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

# mapper(T_var_Rnet, norm=lognorm, bnds=bnds, title='$R_{net}$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
# plt.savefig(fout + '{:s}_TvarRnet_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper(T_var_thf, norm=lognorm, bnds=bnds, title='$THF$', units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
# plt.savefig(fout + '{:s}_TvarTHF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper(T_var_Qek, norm=lognorm, bnds=bnds, title='$Q_{ek}$', cbfrac=cbfrac, units=r'K$^{2}$', cmap=fieldcmap, vmin=vmin, vmax=vmax)
# plt.savefig(fout + '{:s}_TvarQek_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

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
# mapper(Q_ek_mean, logscale=False, bnds=bnds, title='$Q_{ek}$', units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-200, vmax=200)
# plt.savefig(fout + '{:s}_meanQek_h_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

##
# mapper(sst_mean, logscale=False, bnds=bnds, title='$Q_{ek}$', units=r'W/m$^{2}$', cmap=sstcmap, vmin=-10, vmax=30)
# plt.savefig(fout + '{:s}_sst_mean_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

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

logfracabs = np.log(np.abs(T_var_Qs)/np.abs(T_var_Qr)) 
    
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

mapper(T_var_Qs/T_var_sum - T_var_Qr/T_var_sum, logscale=False, bnds=bnds, title='$Q_o$', cbfrac=cbfrac, units=r'', cmap=plt.cm.Spectral_r, vmin=fracvmin, vmax=fracvmax)
plt.savefig(fout + '{:s}_TvarQoFRACDIFF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(logfrac, logscale=False, bnds=bnds, title='$log(\sigma^2_{T,Q_s} / \sigma^2_{T,Q_o})$', cbfrac=cbfrac, units=r'', cmap=logcmap, vmin=logfracmin, vmax=logfracmax)
plt.savefig(fout + '{:s}_TvarQsQoLOGFRAC_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(logfracabs, logscale=False, bnds=bnds, title='$log(|\sigma^2_{T,Q_s}| / |\sigma^2_{T,Q_o}|)$', cbfrac=cbfrac, units=r'', cmap=logcmap, vmin=logfracmin, vmax=logfracmax)
plt.savefig(fout + '{:s}_TvarQsQoLOGFRACABS_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


mapper(T_var_Qs/T_var_Qr, logscale=False, bnds=bnds, title='$\sigma^2_{T,Q_s} / \sigma^2_{T,Q_o}$', cbfrac=cbfrac, units=r'', cmap=logcmap, vmin=-1.0, vmax=3.0)
plt.savefig(fout + '{:s}_TvarQsQoRATIO_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#mapper(-(Tmxlfrac-1), logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} / \sigma^2_{T,mxl}$ - 1', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
#plt.savefig(fout + '{:s}_TmxlTsratio_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

mapper(-(T_var/sst_var-1), logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} / \sigma^2_{T,mxl}$ - 1', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
plt.savefig(fout + '{:s}_TmxlTsratio_TEST_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(sst_var - T_var, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} - \sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-1.0, vmax=1.0)
plt.savefig(fout + '{:s}_TmxlTsDIFF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(Qr_var, bnds=bnds, logscale=False, log=True, title='$Q_o$ Variance', cbfrac=cbfrac, units=r'W$^{2}$ m$^{-4}$', cmap=sstcmap, vmin=10**2, vmax=10**4)
plt.savefig(fout + '{:s}_Qovar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

# mapper(sst_var_ECCO, bnds=bnds, logscale=False, log=True, title='$\sigma^2_T$', cbfrac=cbfrac, units=r'W$^{2}$ m$^{-4}$', cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# plt.savefig(fout + '{:s}_sstECCOvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()
# #
# #mapper(Tmxl_var_ECCO, bnds=bnds, logscale=False, log=True, title='$\sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'W$^{2}$ m$^{-4}$', cmap=sstcmap, vmin=0.01, vmax=sstvmax)
# #plt.savefig(fout + '{:s}_TmxlECCOvar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# #plt.close()
# #
# mapper(sst_var_ECCO - Tmxl_var_ECCO, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} - \sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-5.0, vmax=5.0)
# plt.savefig(fout + '{:s}_TmxlTsDIFFECCO_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper(T_var - Tmxl_var_ECCO, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} - \sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-5.0, vmax=5.0)
# plt.savefig(fout + '{:s}_OAFluxECCOTmxlvarDIFF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper(T_var - sst_var_ECCO, logscale=True, norm=lognorm, bnds=bnds, title='$\sigma^2_{T} - \sigma^2_{T,mxl}$', cbfrac=cbfrac, units=r'', cmap=fieldcmap, vmin=-5.0, vmax=5.0)
# plt.savefig(fout + '{:s}_OAFluxECCOsstvarDIFF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()
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

# mapper(hbar, bnds=bnds, logscale=False, log=False, title='Max. Monthly Climatological Mixed Layer Depth', cbfrac=cbfrac, units=r'm', cmap=cmocean.cm.deep, vmin=0, vmax=300)
# plt.savefig(fout + '{:s}_hmax_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()


mapper(hbar, bnds=bnds, logscale=False, log=False, title='$\overline{h}$', cbfrac=cbfrac, units=r'm', cmap=logcmap, vmin=0, vmax=300)
plt.savefig(fout + '{:s}_hbar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

if not(anom_flag):
    mapper(Qs_mean, bnds=bnds, logscale=False, log=False, title='$\overline{Q}_s$', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-200, vmax=200)
    plt.savefig(fout + '{:s}_Qsbar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    mapper(Qr_mean, bnds=bnds, logscale=False, log=False, title='$\overline{Q}_o$', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-200, vmax=200)
    plt.savefig(fout + '{:s}_Qrbar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()

# mapper(delz_sum, bnds=bnds, logscale=False, log=False, title='Sum $\Delta z$', cbfrac=cbfrac, units=r'm', cmap=logcmap, vmin=0, vmax=300)
# plt.savefig(fout + '{:s}_delzsumtest_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

# mapper(h_clim_std, bnds=bnds, logscale=False, log=False, title='std(monthly mean of h)', cbfrac=cbfrac, units=r'm', cmap=logcmap, vmin=0, vmax=300)
# plt.savefig(fout + '{:s}_hstd_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
# plt.close()

#fac = (2*dt**2/(Cbar*(1-r2corrs)))
#c

#mapper(fac, logscale=False, log=True, bnds=bnds, title=r'$\alpha$', cbfrac=cbfrac, units=r'K s W$^{-1}$', cmap=logcmap, vmin=1e4, vmax=1e6)
#plt.savefig(fout + '{:s}_alpha_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

mapper(alpha, logscale=False, log=True, bnds=bnds, title=r'$\alpha$', cbfrac=cbfrac, units=r'K s W$^{-1}$', cmap=logcmap, vmin=1e4, vmax=1e6)
plt.savefig(fout + '{:s}_alpha_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(G, logscale=False, log=True, bnds=bnds, title='$G$', cbfrac=cbfrac, units=r'K s W$^{-1}$', cmap=logcmap, vmin=1e-5, vmax=1e-2)
plt.savefig(fout + '{:s}_G_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper(r2corrs, logscale=False, log=False, bnds=bnds, title='$r_2$', cbfrac=cbfrac, units=r'', cmap=logcmap, vmin=0, vmax=1.0)
plt.savefig(fout + '{:s}_r2_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()















