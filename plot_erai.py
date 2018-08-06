#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:38:51 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats.stats import pearsonr, linregress
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from ocean_atmosphere.misc_fns import an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
#matplotlib.rcParams.update({'figure.autolayout': True})

#fin = '/Users/cpatrizio/data/MERRA2/'
fin = '/Users/cpatrizio/data/ECMWF/'
#fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

dataname = 'ERAi'
#dataname = 'MERRA2'

#MERRA-2
#fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
#fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')

#ERA-interim
#fsst = cdms2.open(fin + 'skt.197901-201612.nc')
fthf = cdms2.open(fin + 'thf.197901-201712_new.nc')
lhf = fthf('slhf')
lhf = lhf/(12*3600)
shf = fthf('sshf')
#sshf is accumulated 
shf = shf/(12*3600)

thf = shf+lhf

#ERAi has thf positive down
thf = -thf


maxlat = 70
minlat = -70

maxlon = 360
minlon = 0

thf = thf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))

lats = thf.getLatitude()[:]

plt.plot(spatial_ave(thf, lats))
plt.show()

ts = np.array([])
thfs = np.array([])

fnames = glob.glob(fin + 'era_*new.nc')

for fname in fnames:
    fthf = cdms2.open(fname)
    lhf = fthf('slhf')
    lhf = lhf/(12*3600)
    shf = fthf('sshf')
    #sshf is accumulated 
    shf = shf/(12*3600)
    thf = shf+lhf
    
    thf = thf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))

    lats = thf.getLatitude()[:]
    
    thf_globe_ave = spatial_ave(thf, lats)
    
    t = lhf.getTime().asRelativeTime("hours since 1900")
    t = np.array([x.value for x in t])
    t = 1900 + t/(24*365)
    
    ts = np.concatenate([ts, t])
    thfs = np.concatenate([thfs, thf_globe_ave])
    

thfs_an = an_ave(thfs)

tyears = np.arange(ts[0], ts[-1])

plt.plot(tyears, -thfs_an)
plt.show()
    

