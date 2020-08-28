#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:37:06 2020

@author: cpatrizio
"""

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
import cftime
#import proplot as plot
#import seaborn as sns
#import proplot as plot


#fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
#fin = '/Users/cpatrizio/data/OAFlux/'
#fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/ocean-atmosphere/localSST_global/'

#fin = '/Volumes/GoogleDrive/My Drive/data_drive/MERRA2/'

fin = '/Users/cpatrizio/data/CESM-SOM/'

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
dataname = 'CESM2-SOM'

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

#CESM2 s
#ffrc = xr.open_dataset(fin + 'pop_frc.1x1d.090130.nc')

ffrc = xr.open_dataset(fin + 'pop_frc.b.e11.B1850C5CN.f09_g16.005.150217.nc')
ffrc_new = xr.open_dataset(fin + 'forcing_default.nc')
#fsst = xr.open_dataset(fin + 'som.test2.sst.0001-01.nc')
#fflx = xr.open_dataset(fin + 'som.test2.FLXS.0001-01.nc')
#fflx = xr.open_dataset(fin + 'som.test2.FLXS.0001-01.nc')
#fQo = xr.open_dataset(fin + 'OAFlux_Qoanom_monthly1981to2017.nc')
#ffrc = xr.open_dataset(fin + 'pop_frc.b.e20.B1850.f09_g17.pi_control.all.297_bugfix.20181031.nc')

#ffrc = xr.open_dataset(fin + 'pop_frc.b.e11.B1850C5CN.f09_g16.005.fieldvmax217.nc')

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
latbounds_plot = [-65,65]


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

 # Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn =4.*12
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

if not(lowpass):
    Tn = 0.
    
#ffrc = ffrc.rename({'lat': 'nlat', 'lon':'nlon'})
ffrc = ffrc.rename({'ni': 'nlat', 'nj':'nlon'})
ffrc = ffrc.rename({'xc':'lon','yc':'lat'})

ffrc_new = ffrc_new.rename({'ni': 'nlat', 'nj':'nlon'})
ffrc_new = ffrc_new.rename({'xc':'lon','yc':'lat'})

#fsst = fsst.rename({'ni': 'nlat', 'nj':'nlon'})
#fsst = fsst.rename({'TLON':'lon','TLAT':'lat'})

#fflx = fflx.rename({'ni': 'nlat', 'nj':'nlon'})
#fflx = fflx.rename({'TLON':'lon','TLAT':'lat'})


lats = ffrc.lat
lons = ffrc.lon

qflux = ffrc.qdp
qflux = qflux.assign_coords(lat=lats,lon=lons)

h = ffrc.hblt
h = h.assign_coords(lat=lats,lon=lons)

h_new = ffrc_new.hblt
h_new = h_new.assign_coords(lat=lats,lon=lons)
#sst = fsst.sst
#sst = sst.assign_coords(lat=lats,lon=lons)

#Q_o = fQo.Qo_anom

#lhf = fflx.LHFLX
#shf = fflx.SHFLX
#swnet = fflx.FSNS
#lwnet = fflx.FLNS

#R_net_surf = lwup + lwdn + swup - swdn
#Rnet_surf = swnet + lwnet

#Q_s = -(lhf + shf) + R_net_surf

#Q_s = lhf


#qflux_bar = qflux.mean(dim='time')

# Test interpolation of the given T-grid (for later use when adding Q_o variability to q-flux)

#ds_out = xe.util.grid_global(1.0, 1.0)

#ds_out = xr.Dataset({'lon':lons, 'lat':lats})

#regridder = xe.Regridder(Q_o, ds_out, 'bilinear', reuse_weights=False)
#Q_o_regr = regridder(Q_o)
#regridder.clean_weight_file()

#Q_o_slice = Q_o_regr.sel(time=slice('1981-01-01','1982-01-01'))
#Q_o_slice2 = Q_o_regr.sel(time=slice('1982-01-01','1983-01-01'))

times = qflux.time.values

#times2_days = cftime.date2num(times, 'days since 0001-01-15')+365

#times2 = cftime.num2date(times2_days, 'days since 0001-01-15')

#Q_o_slice.time.values = times
#Q_o_slice2.time.values = times2

#Modified q-flux (added year-1 variability)
#qflux_new_gn = qflux - Q_o_slice

#qflux_new_gn2 = qflux - Q_o_slice2


# must drop lat lon coords to mirror the format of ffrc.qdp exactly
#qflux_new_save = qflux_new_gn.drop('lat')
#qflux_new_save = qflux_new_save.drop('lon')

# overwrite ffrc.qdp and save as a new .nc file


ds_out = xe.util.grid_global(1.0, 1.0)

regridder = xe.Regridder(qflux, ds_out, 'bilinear', reuse_weights=False)
qflux = regridder(qflux) 
regridder.clean_weight_file()

#regridder = xe.Regridder(qflux_new_gn, ds_out, 'bilinear', reuse_weights=False)
#qflux_new = regridder(qflux_new_gn) 
#regridder.clean_weight_file()

#regridder = xe.Regridder(sst, ds_out, 'bilinear', reuse_weights=False)
#sst = regridder(sst) 
#regridder.clean_weight_file()

regridder = xe.Regridder(h, ds_out, 'bilinear', reuse_weights=False)
h = regridder(h) 
regridder.clean_weight_file()

regridder = xe.Regridder(h_new, ds_out, 'bilinear', reuse_weights=False)
h_new = regridder(h_new) 
regridder.clean_weight_file()

#regridder = xe.Regridder(Q_s, ds_out, 'bilinear', reuse_weights=False)
#Q_s = regridder(Q_s) 
#regridder.clean_weight_file()

lons = qflux.lon[0,:]
lats = qflux.lat[:,0]

qflux = qflux.drop({'lon','lat'})
qflux = qflux.assign_coords(y=lats,x=lons)
qflux = qflux.rename({'y':'lat','x':'lon'})

#qflux_new = qflux_new.drop({'lon','lat'})
#qflux_new = qflux_new.assign_coords(y=lats,x=lons)
#qflux_new = qflux_new.rename({'y':'lat','x':'lon'})

h = h.drop({'lon','lat'})
h = h.assign_coords(y=lats,x=lons)
h = h.rename({'y':'lat','x':'lon'})

h_new = h_new.drop({'lon','lat'})
h_new = h_new.assign_coords(y=lats,x=lons)
h_new = h_new.rename({'y':'lat','x':'lon'})

#sst = sst.drop({'lon','lat'})
#sst = sst.assign_coords(y=lats,x=lons)
#sst = sst.rename({'y':'lat','x':'lon'})

qflux = qflux.assign_coords(lon=(qflux.lon % 360)).roll(lon=((qflux.shape[2] // 2)))
#qflux_new = qflux_new.assign_coords(lon=(qflux_new.lon % 360)).roll(lon=((qflux_new.shape[2] // 2)))
h = h.assign_coords(lon=(h.lon % 360)).roll(lon=((h.shape[2] // 2)))
h_new = h_new.assign_coords(lon=(h_new.lon % 360)).roll(lon=((h_new.shape[2] // 2)))
#sst = sst.assign_coords(lon=(sst.lon % 360)).roll(lon=((sst.shape[2] // 2)))
#Q_s = Q_s.assign_coords(lon=(sst.lon % 360)).roll(lon=((sst.shape[2] // 2)))

qflux_bar = qflux.mean(dim='time')

h[:,:,320] = (h[:,:,318] + h[:,:,321])/2.
h[:,:,319] = (h[:,:,318] + h[:,:,320])/2.

h_new[:,:,320] = (h_new[:,:,318] + h_new[:,:,321])/2.
h_new[:,:,319] = (h_new[:,:,318] + h_new[:,:,320])/2.

qflux[:,:,320] = (qflux[:,:,318] + qflux[:,:,321])/2.
qflux[:,:,319] = (qflux[:,:,318] + qflux[:,:,320])/2.

#qflux_new[:,:,320] = (qflux_new[:,:,318] + qflux_new[:,:,321])/2.
#qflux_new[:,:,319] = (qflux_new[:,:,318] + qflux_new[:,:,320])/2.

#sst[:,:,320] = (sst[:,:,318] + sst[:,:,321])/2.
#sst[:,:,319] = (sst[:,:,318] + sst[:,:,320])/2.

#Q_s[:,:,320] = (Q_s[:,:,318] + Q_s[:,:,321])/2.
#Q_s[:,:,319] = (Q_s[:,:,318] + Q_s[:,:,320])/2.

qflux_bar = qflux.mean(dim='time')
h_bar = h.mean(dim='time')

hbar_new = h.mean(dim='time')

lats = qflux.lat
lons = qflux.lon

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
    
    
#sstcmap = cmocean.cm.plasma
#sstcmap = plt.cm.cubehelix_r
#fieldcmap = cmocean.cm.balance
fieldcmap = plt.cm.RdBu_r

sstcmap = cc.cm.CET_L17
#fieldcmap = cc.cm.CET_D1A
#fieldcmap = cc.cm.coolwarm

qflux_JJA = qflux[5:8,:].mean(dim='time')
qflux_DJF = (qflux[0:2].mean(dim='time') + qflux[-1])/2.

#qflux_new_bar = qflux_new.mean(dim='time')

#qflux_new_JJA = qflux_new[5:8,:].mean(dim='time')
#qflux_new_DJF = (qflux_new[0:2].mean(dim='time') + qflux_new[-1])/2.

# h_JJA = h[5:8,:].mean(dim='time')
# h_DJF = (h[0:2].mean(dim='time') + h[-1])/2.

fieldvmin=-200
fieldvmax=200

mapper = Mapper()
cbfrac=0.11
#mapper(sst[0,:], bnds=bnds, logscale=False, log=False, title='SST', cbfrac=cbfrac, units=r'deg C', cmap=sstcmap, vmin=0, vmax=30)
#plt.savefig(fout + '{:s}_sst_test_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()


mapper = Mapper()
mapper(h_bar, bnds=bnds, logscale=False, log=False, title='mixed-layer depth (mean)', cbfrac=cbfrac, units=r'm', cmap=plt.cm.Spectral_r, vmin=0, vmax=300)
plt.savefig(fout + '{:s}_hbar_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(hbar_new, bnds=bnds, logscale=False, log=False, title='mixed-layer depth (mean)', cbfrac=cbfrac, units=r'm', cmap=plt.cm.Spectral_r, vmin=0, vmax=300)
plt.savefig(fout + '{:s}_hbar_new_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(h_bar/hbar_new, bnds=bnds, logscale=False, log=False, title='mixed-layer depth (mean)', cbfrac=cbfrac, units=r'm', cmap=plt.cm.Spectral_r, vmin=0, vmax=1)
plt.savefig(fout + '{:s}_hbar_ratio_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(qflux_bar, bnds=bnds, logscale=False, log=False, title='q-flux (mean)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
plt.savefig(fout + '{:s}_qflux_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#mapper = Mapper()
#mapper(Q_s[0,:], bnds=bnds, logscale=False, log=False, title='$Q_s$', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
#plt.savefig(fout + '{:s}_Qs_test_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

mapper = Mapper()
mapper(qflux_JJA, bnds=bnds, logscale=False, log=False, title='q-flux (JJA)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
plt.savefig(fout + '{:s}_qflux_JJA_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

mapper = Mapper()
mapper(-qflux_DJF, bnds=bnds, logscale=False, log=False, title='q-flux (DJF)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
plt.savefig(fout + '{:s}_qflux_DJF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#mapper = Mapper()
#mapper(-qflux_new_bar, bnds=bnds, logscale=False, log=False, title='q-flux year 1 (mean)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
#plt.savefig(fout + '{:s}_qflux_new_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper = Mapper()
#mapper(-qflux_new_JJA, bnds=bnds, logscale=False, log=False, title='q-flux year 1 (JJA)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
#plt.savefig(fout + '{:s}_qflux_new_JJA_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper = Mapper()
#mapper(-qflux_new_DJF, bnds=bnds, logscale=False, log=False, title='q-flux year 1 (DJF)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=fieldvmin, vmax=fieldvmax)
#plt.savefig(fout + '{:s}_qflux_new_DJF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper = Mapper()
#mapper(-qflux_new_bar+qflux_bar, bnds=bnds, logscale=False, log=False, title='$Q^{\prime}_o$ year 1 (mean)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-100, vmax=100)
#plt.savefig(fout + '{:s}_Qoanom__{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper = Mapper()
#mapper(-qflux_new_JJA+qflux_JJA, bnds=bnds, logscale=False, log=False, title='$Q^{\prime}_o$ year 1 (JJA)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-100, vmax=100)
#plt.savefig(fout + '{:s}_Qoanom_JJA_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#mapper = Mapper()
#mapper(-qflux_new_DJF+qflux_DJF, bnds=bnds, logscale=False, log=False, title='$Q^{\prime}_o$ year 1 (DJF)', cbfrac=cbfrac, units=r'W/m$^{2}$', cmap=fieldcmap, vmin=-100, vmax=100)
#plt.savefig(fout + '{:s}_Qoanom_DJF_{:1.0f}LP_{:2.0f}Nto{:2.0f}N_detr{:s}.png'.format(dataname, Tn/12., latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()