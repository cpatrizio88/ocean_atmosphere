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

ffrc = xr.open_dataset(fin + 'pop_frc.b.e11.B1850LENS.f09_g16.pi_control.002.20190923.nc')
#ffrc = xr.open_dataset(fin + 'forcing_default.nc')
#fsst = xr.open_dataset(fin + 'som.test2.sst.0001-01.nc')
#fflx = xr.open_dataset(fin + 'som.test2.FLXS.0001-01.nc')
#fflx = xr.open_dataset(fin + 'som.test2.FLXS.0001-01.nc')
fQo = xr.open_dataset(fin + 'OAFlux_Qoanom_CESM2hbar_monthly1981to2015.nc')
#fQo = xr.open_dataset(fin + 'CESM2_Qoanom_monthly36years.nc')
fhbar = xr.open_dataset(fin + 'CESM2_hbar36years.nc')
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
latbounds_plot = [-89.5,89.5]


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
    

lats = ffrc.yc
lons = ffrc.xc

qflux = ffrc.qdp
#qflux = qflux.assign_coords(lat=lats,lon=lons)

#h = ffrc.hblt
#h = h.assign_coords(lat=lats,lon=lons)

#sst = fsst.sst
#sst = sst.assign_coords(lat=lats,lon=lons)

Q_o = fQo.Qo_anom

hbar = fhbar.hblt

#zero out polar regions
#latmax = np.where(Q_o.lat > 30)[0][0]
#latmin = np.where(Q_o.lat > -30)[0][0]

#Q_o[:,0:latmin,:] = 0
#Q_o[:,latmax:-1,:] = 0

#zero out along prime meridian
#lonmer = np.where(Q_o.lon == 0.5)[0][0]
#lonmer2 = np.where(Q_o.lon == 359.5)[0][0]

#Q_o[:,:,lonmer] = 0
#Q_o[:,:,lonmer2] = 0

lats_Qo = Q_o.lat
lons_Qo = Q_o.lon

ds_out = xr.Dataset({'lon':lons, 'lat':lats})

#Q_o_temp = Q_o[0:12,:]

#To test that model is reading the forcing correctly, add a large constant Q_o anomaly in the midlatitude Pacific for months 12-24
#Q_o.loc[dict(lon=lons_Qo[(lons_Qo > 140) & (lons_Qo < 220)], lat=lats_Qo[(lats_Qo > 20) & (lats_Qo < 50)])] = 400

#Q_o[0:12,:] = Q_o_temp

regridder = xe.Regridder(Q_o, ds_out, 'bilinear', reuse_weights=False)
Q_o_regr = regridder(Q_o)
regridder.clean_weight_file()

regridder = xe.Regridder(hbar, ds_out, 'bilinear', reuse_weights=False)
hbar_regr = regridder(hbar)
regridder.clean_weight_file()

#modify this to match the number of years in the q-flux forcing file
nyears = 1

#get mask for  where Q_o anomaly field is missing data
missing_vals=np.isnan(Q_o_regr)
missing_vals_hbar = np.isnan(hbar_regr)

#Q_o_regr = Q_o_regr.where(hbar_regr != 0)

#Q_o_regr = Q_o_regr.interpolate_na(dim='nj').interpolate_na(dim='ni')

#get mask for default forcing file
mask = (ffrc.mask == 0)

#get mask for indices where Q_o is missing data, but unmasked in default forcing file
zero_fill = np.bitwise_and(missing_vals, ~mask)

#hbar_fill = np.bitwise_and(missing_vals_hbar, ~mask)


#any missing data in the Q_o anomlies or hbar that correspond to unmasked indices in the default file must be filled
#Filling missing points with 50 m works... but is not entirely physical

#hbar_regr = hbar_regr.where(~hbar_fill, 50)

#Interpolate missing values
hbar_regr = hbar_regr.where(hbar_regr != 0)
hbar_regr = hbar_regr.interpolate_na(dim='nj').interpolate_na(dim='ni')

#cover any points missed by the interpolation
missing_vals_hbar = np.isnan(hbar_regr)
hbar_regr = hbar_regr.where(~missing_vals_hbar, 50)

#Fill Q_o nomalies with zero for missing data
Q_o_regr = Q_o_regr.where(~zero_fill, 0)

# Use default mask to mask new hbar 
hbar_regr = hbar_regr.where(~mask)

Q_o_slice = Q_o_regr[0:12*nyears,:]

Q_o_slice = Q_o_slice.drop({'lon','lat'})

times = qflux.time.values

times_num = cftime.date2num(times, units='days since 0001-01-01 00:00:00', calendar='noleap')
times_new = cftime.num2date(times_num, units='days since 1981-01-01 00:00:00', calendar='noleap')

Q_o_slice.time.values = times_new

hblt_new = hbar_regr.expand_dims({'time':times_new})

#Q_o_slice = Q_o_slice

#the minus sign is because for some reason the slab ocean solves the following equation: dT/dt = Fnet - Qflux
#so positive Qflux means heat divergence
#qflux_new = qflux - Q_o_slice

qflux_new = qflux
qflux_new.time.values = times_new
qflux_new.load()

qflux_new = qflux_new.astype('float32')

#qflux_new_bar_spatialave = (qflux_new*ffrc.area).sum()

#qflux_new = qflux_new - qflux_new_bar_spatialave

ffrc_new = ffrc.copy()

ffrc_new.qdp.values = qflux_new.values

#ffrc_new.hblt.values = 2*ffrc.hblt.values
ffrc_new.hblt.values = hblt_new.values


#ffrc_new.to_netcdf(fin + 'forcing_test_OAFlux.nc', format='NETCDF3_CLASSIC')
#ffrc_new.to_netcdf(fin + 'forcing_test_CESM2.nc', format='NETCDF3_CLASSIC')
ffrc_new.to_netcdf(fin + 'forcing_test_hist.nc', format='NETCDF3_CLASSIC')


ffrc.close()
ffrc_new.close()





