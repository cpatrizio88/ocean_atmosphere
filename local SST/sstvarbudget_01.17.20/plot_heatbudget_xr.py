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

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'axes.titlesize': 22})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 18})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
matplotlib.rcParams.update({'axes.labelsize': 16})
matplotlib.rcParams.update({'ytick.labelsize': 16})
matplotlib.rcParams.update({'xtick.labelsize': 16})

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
lowpass = True
anom_flag = True
timetend=False
detr=True
rENSO=False
corr=False
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
c = 3850
#omega = 7.2921e-5
rho = 1000
#f = 2*omega*np.sin(np.deg2rad(lats))
#r = 6.371e6
#g=9.81

#dphi = np.diff(lats)[0]*(2*np.pi/360.)
#dpsi = np.diff(lons)[0]*(2*np.pi/360.)

#dx = r*np.cos(np.deg2rad(lats))*dpsi
#dy = r*dphi

#dx2D = np.zeros((nt, nlat))
#dx2D[:,:] = dx
#dx3D = np.repeat(dx2D[:,:,np.newaxis],nlon-2,axis=2)

#f2D = np.zeros((nt, nlat))
#f2D[:,:] = f
#f3D = np.repeat(f2D[:,:,np.newaxis],nlon-2,axis=2)
#f3D = f3D[:,1:-1,:]

#dSSTdx = (sst[:,:,2:] - sst[:,:,:-2])/(2*dx3D)

#dSSTdy = (sst[:,2:,:] - sst[:,:-2,:])/(2*dy)

#This is the pressure due to sea-surface height perturbations
#dpdx = rho*g*(ssh[:,:,2:] - ssh[:,:,:-2])/(2*dx3D)
#dpdy = rho*g*(ssh[:,2:,:]  - ssh[:,:-2,:])/(2*dy)



#dSSTdx = dSSTdx[:,1:-1,:]
#dSSTdy = dSSTdy[:,:,1:-1]
#dpdx = dpdx[:,1:-1,:]
#dpdy = dpdy[:,:,1:-1]
#taux = taux[:,1:-1,1:-1]
#tauy = tauy[:,1:-1,1:-1]

#f3D = np.ma.array(f3D, mask=f3D == 0)



#hbar3D = np.ma.zeros((nt, nlat-2, nlon-2))

#hbar3D[:,:,:] = hbar[1:-1,1:-1]



#u_ek = (1/(rho*f3D*hbar3D))*tauy

#v_ek = -(1/(rho*f3D*hbar3D))*taux


#Q_ek = -(c*rho*hbar3D)*(u_ek*dSSTdx + v_ek*dSSTdy)

#eqi = np.where(lats[1:-1]==0)
#Q_ek[:,eqi,:] = 0

#Q_ek_test = Q_ek


#Geostrophic currents are u = -1/(rho*f)*dp/dy, v = 1/(rho*f)*dp/dx
#Advection by geostrophic current is then u*dT/dx + v*dT/dy

#the pressure is actually calculated HYRDOSTATICALLY in the upper ocean mixed layer (so density of water*dh/d)

#u_g = -1/(rho*f3D)*dpdy
#v_g = 1/(rho*f3D)*dpdx

#Q_g = -(c*rho*hbar3D)*(u_g*dSSTdx + v_g*dSSTdy)

#timeaxis = sst.getAxis(0)
#Q_g.setAxis(0, timeaxis)

#Q_g = np.ma.array(Q_g, mask=Q_g>1e3)

#Q_g_test = Q_g

#CHANGE C_P to 3850
c_p = 3850
dt = 30*3600*24
C = rho*c_p*h
#Tbar = MV.average(sst,axis=0)
Cbar = C.mean(dim='time')

#ssttend = (sst[2:]-sst[:-2])/(2*dt)

#Qo = Cbar*ssttend -(-thf[1:-1] + Q_net_surf[1:-1])


#Qr = Qo[:,1:-1,1:-1] - Q_ek[1:-1,:]

#timeaxis = sst[1:-1,:].getAxis(0)
#Qr.setAxis(0, timeaxis)



#lats = sst.getLatitude()[:]
#lons = sst.getLongitude()[:]


#Qr = Qo[:,1:-1,1:-1] - (Q_ek[1:-1,:] + Q_g[1:-1,:])

#Qr_test = Qr

#Tbar = Tbar[1:-1]
#Cbar = Cbar[1:-1]

#cdutil.setTimeBoundsMonthly(sst)
#cdutil.setTimeBoundsMonthly(thf)
#cdutil.setTimeBoundsMonthly(Q_net_surf)
#cdutil.setTimeBoundsMonthly(h)
#cdutil.setTimeBoundsMonthly(Q_ek)
#cdutil.setTimeBoundsMonthly(Q_g)
#cdutil.setTimeBoundsMonthly(Qr)


if anom_flag:
    Q_net_surf,Q_net_surf_clim = st.anom(Q_net_surf)
    thf,thf_clim = st.anom(thf)
    sst,sst_clim = st.anom(sst)

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
 
 #Q_ek = detrend(Q_ek)
 #Q_g = detrend(Q_g)
 #Qr = detrend(Qr)
 #Qo = detrend(Qo)
 
 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12.*10
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

if not(lowpass):
    Tn = 0.
    

if lowpass:
    #sst = xr.DataArray(st.butter_lowpass_filter(sst, cutoff, fs, order),dims=sst.dims,coords=sst.coords)
    #thf = xr.DataArray(st.butter_lowpass_filter(thf, cutoff, fs, order),dims=thf.dims,coords=thf.coords)
    #Q_net_surf = xr.DataArray(st.butter_lowpass_filter(Q_net_surf, cutoff, fs, order),dims=Q_net_surf.dims,coords=Q_net_surf.coords)
    sst = st.butter_lowpass_filter_xr(sst, cutoff, fs, order)
    thf = st.butter_lowpass_filter_xr(thf, cutoff, fs, order)
    Q_net_surf = st.butter_lowpass_filter_xr(Q_net_surf, cutoff, fs, order)
    #Q_ek = butter_lowpass_filter(Q_ek, cutoff, fs, order)
    #Q_g = butter_lowpass_filter(Q_g, cutoff, fs, order)



#if dataname == 'OAFlux':
#    sst = MV.array(sst, mask=np.abs(sst) > 1e2)
#    thf = MV.array(thf, mask=np.abs(thf) > 1e3)
#    Q_net_surf = MV.array(Q_net_surf, mask=np.abs(Q_net_surf) > 1e3)
    
    
sst = sst.where(~(sst==0))
thf = thf.where(~(thf==0))
Q_net_surf = Q_net_surf.where(~(thf==0))

tendsst = (sst.shift(time=-2)-sst)[:-2]
tendsst = tendsst/(2*dt)

nt = sst.shape[0]

thf = thf.isel(time=slice(1,nt-1))
Q_net_surf = Q_net_surf.isel(time=slice(1,nt-1))
sst = sst.isel(time=slice(1,nt-1))

# make sure sst tendency times match up with other field times
tendsst.time.values = thf.time.values
 
#tendsst  = (sst[2:]-sst[:-2])/(2.*dt)
#tendsst  = (sst[1:]-sst[:-1])/(dt)

#thf = (thf[1:] + thf[:-1])/2.
#Q_net_surf = (Q_net_surf[1:] + Q_net_surf[:-1])/2.
#sst = (sst[1:] + sst[:-1])/2.



Qr = Cbar*tendsst -(-thf + Q_net_surf)
Qr = Qr.transpose('time','lat','lon')
#Qr = Cbar*tendsst -(-thf + Q_net_surf)

#thf = thf[1:-1]
#Q_net_surf = Q_net_surf[1:-1]
#sst = sst[1:-1]
#Q_ek = Q_ek[1:-1]
#Q_g = Q_g[1:-1]
#Qo.setAxis(0,timeaxis)

Q_s = -thf + Q_net_surf


#Q_s= -thf[1:-1,1:-1,1:-1] + Q_net_surf[1:-1,1:-1,1:-1] + Q_ek
#Q_s= -thf[1:-1,1:-1,1:-1] + Q_net_surf[1:-1,1:-1,1:-1]

#Qr = Qo[:,1:-1,1:-1] - Q_ek
#Qr = Qo[:,1:-1,1:-1]

#Qr = Cbar[1:-1,1:-1]*ssttend[:,1:-1,1:-1] - Q_s
#timeaxis = sst.getAxis(0)

#tendsst.setAxis(0, timeaxis)
#Qo.setAxis(0,timeaxis)
#Q_ek.setAxis(0, timeaxis)
#Q_g.setAxis(0, timeaxis)
#Qr.setAxis(0, timeaxis)
#Q_s.setAxis(0, timeaxis)
#htend.setAxis(0, timeaxis


#tendH = Cbar*tendsst

#
#nt = sst.shape[0]
#timeslice = slice(int(Tn),nt-int(Tn))
#
#sst = sst.isel(time=timeslice)
#tendsst = tendsst.isel(time=timeslice)
#Q_s = Q_s.isel(time=timeslice)
#Qr = Qr.isel(time=timeslice)



# Concatenate ends of time series after smoothing (due to error there)
#t1=int(Tn)
#t2=nt-int(Tn)
#if not(lowpass):
#   t1=0
#   t2=nt

#tendsst = tendsst[t1:t2,:]
#Q_s = Q_s[t1:t2,:]
#Qr = Qr[t1:t2,:]
#Q_net_surf = Q_net_surf[t1:t2,:]
#thf = thf[t1:t2,:]
#sst = sst[t1:t2,:]
#times = timeaxis[t1:t2]
   

tendH = Cbar*tendsst

totalH = Q_s + Qr

error = tendH - totalH

error_sum = error.sum(dim='time')

#Plot maps of SST and THF patterns associated with AMO
#CHANGE THIS FOR MAP PROJECTION
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

times = sst.time
#tyears = 1980 + times/12.
           
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
plot = ax.pcolormesh(x, y, error_sum, cmap=plt.cm.RdBu_r, vmin=-0.1, vmax=0.1, transform=cart.crs.PlateCarree())
cb = plt.colorbar(plot, orientation = orient, label=r'W/m$^{2}$')
plt.title(r'heat budget error')
plt.savefig(fout + '{:s}_heat_budget_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

j=35
i=35 


plt.figure(figsize=(16,12))
plt.subplot(2, 1, 1)
plt.plot(times, tendH.isel(lat=i,lon=j), lw=4, color='K', marker='.',label='total tendency')
plt.plot(times, Q_s.isel(lat=i,lon=j), lw=2, color='C0', marker='.',label='surface forcing')
plt.plot(times, Qr.isel(lat=i,lon=j), lw=2, color='C1', marker='.',label='residual')
plt.axhline(0,color='k',lw=1)
plt.setp(plt.gca(), 'xticklabels',[])
plt.legend(loc='best',frameon=False,fontsize=14)

plt.subplot(2, 1, 2)
plt.plot(times, totalH.isel(lat=i,lon=j), lw=4, color='red', marker='.',label='RHS')
plt.plot(times, tendH.isel(lat=i,lon=j), lw=2, color='blue', marker='.',label='LHS')
plt.plot(times, error.isel(lat=i,lon=j), lw=2, color='k', marker='.',label='RHS - LHS')
plt.legend(loc='best',frameon=False,fontsize=14)






