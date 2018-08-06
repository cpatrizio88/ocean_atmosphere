#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import cdutil
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats.stats import pearsonr, linregress
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, regressout_x, cov2_coeff, butter_lowpass_filter
from matplotlib.patches import Polygon

fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/NA index/'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')


#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201612.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#dataname = 'ERAi'
dataname = 'MERRA2'

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
maxlat = 90
minlat = -90

maxlon = 360
minlon = 0

tskip = 12

ps = fSLP('SLP')
ps = ps/1e2
ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#nt_ps = ps.shape[0]
#ps = ps[tskip:,:]

#sst = fsst('skt')
sst = fsst('TSKINWTR')
#lats = sst.getLatitude()[:]
#sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#sst = sst[tskip:nt_ps,:]

#cE = fcE('CDH')
#cD = fcD('CN')

lhf = fsst('EFLUXWTR')
shf = fsst('HFLUXWTR')
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

#lhf = fthf('slhf')
#lhf = lhf/(12*3600)
#shf = fthf('sshf')
#sshf is accumulated 
#shf = shf/(12*3600)
thf = lhf + shf
#thf is positive down in ERAi, convert to positive up
#thf = thf

#cf = cffile('MDSCLDFRCLO')
#cf = cffile('MDSCLDFRCHI')
cf = cffile('MDSCLDFRCTTL')
#cf = cffile('ISCCPCLDFRC')


#ps = fSLP('SLP')
#ps = ps/1e2
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#ps = ps[tskip:,:]/1e2

#u = fuv('U10M')
#u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#u = u[tskip:nt_ps,:]
#
#v = fuv('V10M')
#
#v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#v = v[tskip:nt_ps,:]

#umag = np.sqrt(np.square(v) + np.square(u))

#qv10m = fRH('QV10M')
LW_net_surf = radfile['LWGNT']
LW_net_surf_cs = radfile('LWGNTCLR')
SW_net_surf = radfile['SWGNT']
SW_net_surf_cs = radfile('SWGNTCLR')
#LW_net_TOA = radfile['LWTUP']
#lwnettoa_cs = radfile('LWTUPCLR')
#swnettoa = radfile('SWTNT')
#swnettoa_cs = radfile('SWTNTCLR')

SW_down_surf = radfile('SWGDN')
SW_down_surf_cs = radfile('SWGDNCLR')

Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

Q_net_surf = LW_net_surf + SW_net_surf

CRE_surf = Q_net_surf - Q_net_surf_cs

SWCRE_surf = SW_net_surf - SW_net_surf_cs
LWCRE_surf = LW_net_surf - LW_net_surf_cs

alb = 1 - SW_down_surf/SW_net_surf

albcs = 1 - SW_down_surf_cs/SW_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH MM
#field = umag
#ftitle = r'$|\mathbf{u}_{10m}|$'
#fsave = 'umag'
#units = 'm/s'

#field = cD*1e3
#ftitle = r'$c_D$'
#fsave = 'cD'
#units = r'10$^{-3}$'

#field = cE*1e3
#ftitle = r'$c_{E,heat}$'
#fsave = 'cEheat'
#units = r'10$^{-3}$ kg m$^{-2}$ s$^{-1}$'

#field=qv10m
#ftitle=r'RH$_{10m}$'
#fsave = 'RH10m'
#units = '%'

#field = Q_net_surf
#ftitle = r'$Q_{net}$'
#fsave = 'Qnetsurf'
#units = r'W m$^{-2}$'

#field = Q_net_surf_cs
#ftitle = r'$Q_{net,clear}$'
#fsave = 'Qnetsurfcs'
#units = r'W m$^{-2}$'

#field = LW_net_surf
#ftitle = r'$LW_{net}$'
#fsave = 'LWnetsurf'
#units = r'W m$^{-2}$'
#
#field = SW_net_surf
#ftitle = r'$SW_{net}$'
#fsave = 'SWnetsurf'
#units = r'W m$^{-2}$'

#field = SW_down_surf
#ftitle = r'$SW_{down}$'
#fsave = 'SWdownsurf'
#units = r'W m$^{-2}$'

#field = SW_down_surf_cs
#ftitle = r'$SW_{down,clear}$'
#fsave = 'SWdownsurfcs'
#units = r'W m$^{-2}$'
#
field = albcs
ftitle = r'$\alpha_{clear}$'
fsave = 'albedocs'
units = r''


#field = SW_net_surf_cs
#ftitle = r'$SW_{net,clear}$'
#fsave = 'SWnetsurfcs'
#units = r'W m$^{-2}$'

#field = LW_net_surf_cs
#ftitle = r'$LW_{net,clear}$'
#fsave = 'LWnetsurfcs'
#units = r'W m$^{-2}$'

#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

#field = SWCRE_surf
#ftitle = r'SW CRE$_{surf}$'
#fsave = 'SWCREsurf'
#units = r'W m$^{-2}$'

field = LWCRE_surf
ftitle = r'LW CRE$_{surf}$'
fsave = 'LWCREsurf'
units = r'W m$^{-2}$'



#field = thf
#ftitle = r'THF'
#fsave = 'thf'
#units = r'W m$^{-2}$'

#field = lhf
#ftitle = r'LHF'
#fsave = 'lhf'
#units = r'W m$^{-2}$'

#field = shf
#ftitle = r'SHF'
#fsave = 'shf'
#units = r'W m$^{-2}$'


#cf = cffile['MDSCLDFRCTTL']
#cf = cf[tskip:,:]

#field = cf*100.
#ftitle = r'$f_{total}$'
#fsave = 'ftotal'
#units = '%'

#field = cf*100.
#ftitle = r'$f_{low}$'
#fsave = 'flow'
#units = '%'

#field = cf*100.
#ftitle = r'$f_{high}$'
#fsave = 'fhigh'
#units = '%'

#field = sst
#ftitle = r'SST'
#fsave = 'sst'
#units = 'K'

#field = sst
#ftitle = r'$\frac{\partial T_s}{\partial t}$'
#fsave = 'dsstdt'
#units = 'K/s'

#field = ps
#ftitle = 'SLP'
#fsave = 'SLP'
#units = 'hPa'



#NAmaxlati = np.where(lats > maxlat)[0][0]
#NAminlati = np.where(lats > minlat)[0][0]

#sst = sst.subRegion(longitude=(minlon, maxlon))
#sst = sst[tskip:,NAminlati:NAmaxlati,:]
#field = field[tskip:,NAminlati:NAmaxlati,:]

if field.shape[0] < ps.shape[0]:
    nt_ps = field.shape[0]
    ps = ps[tskip:nt_ps,:]
else:
    nt_ps = ps.shape[0]
    ps = ps[tskip:,:]

if not(fsave == 'sst'):
    field = field.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
    field = field[tskip:nt_ps,:]
    
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:nt_ps,:]

#field = field[:ps.shape[0],...]/qsat

sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
##field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)
#ps_mask = np.ma.getmaskarray(ps)
#ps_mask = np.ma.mask_or(sst_mask[:ps.shape[0],:,:], ps_mask)
#ps = np.ma.array(ps, mask=ps_mask)

#True for detrending data, False for raw data
detr=True
corr=False
lterm=True
rENSO=True
drawbox=True

#EDIT THIS FOR BOUNDS
lonbounds = [280.,360.]
latbounds = [0,70.]

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
# ps = detrend(ps)
# u = detrend(u)
# v = detrend(v)
 field = detrend(field)
 
 
lats = field.getLatitude()[:]
lons = field.getLongitude()[:]


#lats = sst.getLatitude()[:]
#lons = sst.getLongitude()[:]
#nt = sst.shape[0]
#lons[0] = 0
#nlat = len(lats)
#nlon = len(lons)


t = field.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
tyears = 1980 + t/12.




#tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(field)

field = cdutil.ANNUALCYCLE.departures(field)
sst = cdutil.ANNUALCYCLE.departures(sst)

CTIminlati = np.argmin(np.abs(lats - (-6)))
CTImaxlati = np.argmin(np.abs(lats - 6))
CTIminloni = np.argmin(np.abs(lons - 0))
CTImaxloni = np.argmin(np.abs(lons - 90))

# CTI Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 3.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
 
CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])

CTI = butter_lowpass_filter(CTI, cutoff, fs, order)

#TODO:  SUBTRACT COLD TONGUE INDEX FROM SST

#coarse grid lat/lon spacing
cstep=2
lats = np.arange(minlat,maxlat+cstep,cstep)
lons = np.arange(0,360+cstep,cstep)


cgrid = cdms2.createGenericGrid(lats,lons)
#regridfunc = Regridder(ingrid, cgrid)
sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

sst = sst.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
field = field.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]
nt = sst.shape[0]
#lons[0] = 0
nlat = len(lats)
nlon = len(lons)

latdiff = np.diff(lats)[0] 
londiff = np.diff(lons)[0]



#Regress out CTI
if rENSO:
    CTIlag = 2
    sst = regressout_x(sst[CTIlag:,...], CTI[:-CTIlag])
    #field = regressout_x(field[CTIlag:,...], CTI[:-CTIlag])
    field = field[CTIlag:,...]
    tyears = tyears[CTIlag:,...]
#field = regressout_x(field, CTI)
    
nt = sst.shape[0]

# CTI Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12*7.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

#subtract global annual mean to isolate local processes
#sstprime = sst.T - sst_globe
#sstprime = sstprime.T
sstprime = sst

#fieldprime = field.T - field_globe
#fieldprime = fieldprime.T
#fieldprime = field

#need to fill missing field data by interpolation... THF blows up otherwise 
#sst_df = pd.DataFrame(sst.reshape(nt, nlat*nlon))
field_df = pd.DataFrame(field.reshape(nt, nlat*nlon))
#sst_df = sst_df.interpolate()
field_df = field_df.interpolate()
#sstprime = sst_df.values.reshape(nt, nlat, nlon)
fieldprime = field_df.values.reshape(nt, nlat, nlon)

field_lt = butter_lowpass_filter(fieldprime, cutoff, fs, order)
sst_lt = butter_lowpass_filter(sstprime, cutoff, fs, order)

fieldprime = np.ma.masked_array(fieldprime, mask=~np.isfinite(fieldprime))
field_lt = np.ma.masked_array(field_lt, mask=~np.isfinite(field_lt))

#field_lt = running_mean(fieldprime, Tn)
#sst_lt = running_mean(sst, Tn)
#ci = (Tn-1)/2
#field_lt = np.ma.masked_array(field_lt, mask=np.abs(field_lt) > 1e4)
#sst_lt = np.ma.masked_array(sst_lt, mask=np.abs(sst_lt) > 1e4)

 
field_st =  fieldprime - field_lt
sst_st = sstprime - sst_lt


#EDIT FOR INDEX OF INTEREST

latw = 15
slats = np.array([5])   
    
for slati in slats:  
    
    nlati = slati+latw
    wloni = 305
    eloni = 335
    
    si = np.argmin(np.abs(lats - slati))
    ni = np.argmin(np.abs(lats - nlati))
    wi = np.argmin(np.abs(lons - wloni))
    ei = np.argmin(np.abs(lons - eloni))
    
    
    index = spatial_ave(sstprime[:,si:ni,wi:ei], lats[si:ni])
    

    
    #need to fill missing field data by interpolation... THF blows up otherwise 
    #sst_df = pd.DataFrame(sst.reshape(nt, nlat*nlon))
    #field_df = pd.DataFrame(field.reshape(nt, nlat*nlon))
    ##sst_df = sst_df.interpolate()
    #field_df = field_df.interpolate()
    ##sstprime = sst_df.values.reshape(nt, nlat, nlon)
    #fieldprime = field_df.values.reshape(nt, nlat, nlon)
    
    
    index_lt = butter_lowpass_filter(index, cutoff, fs, order)
    index_st = index - index_lt
    
    
    nt = field.shape[0]
    nt_lt = sst_lt.shape[0]
    
    
    scaler = StandardScaler()
    indexstd = scaler.fit_transform(index.reshape(-1,1))
    indexstd_lt = scaler.fit_transform(index_lt.reshape(-1,1))
    indexstd_st = scaler.fit_transform(index_st.reshape(-1,1))
    
    CTIstd = scaler.fit_transform(CTI.reshape(-1,1))
    
    indexstdrep = np.squeeze(np.repeat(indexstd[:,np.newaxis], nlon, axis=1))
    indexstd_ltrep = np.squeeze(np.repeat(indexstd_lt[:,np.newaxis], nlon, axis=1))
    indexstd_strep = np.squeeze(np.repeat(indexstd_st[:,np.newaxis], nlon, axis=1))
    
    
    sstcorrs = MV.zeros((nlat,nlon))
    sstcorrs_lt = MV.zeros((nlat, nlon))
    sstcorrs_st = MV.zeros((nlat, nlon))
    fieldcorrs = MV.zeros((nlat, nlon))
    fieldcorrs_lt = MV.zeros((nlat,nlon))
    fieldcorrs_st = MV.zeros((nlat,nlon))
    
    CTIcorrs = MV.zeros((nlat,nlon))
    
    sstprime_g = sstprime.reshape(nt, nlat*nlon)
    clf = linear_model.LinearRegression()
    clf.fit(indexstd.reshape(-1,1), sstprime_g)
    sstcorrs = clf.coef_.reshape(nlat, nlon) 
    
    if lterm:
        sst_lt_g = sst_lt.reshape(nt_lt, nlat*nlon)
        clf = linear_model.LinearRegression()
        clf.fit(indexstd.reshape(-1,1), sst_lt_g)
        sstcorrs_lt = clf.coef_.reshape(nlat, nlon)  
        
        sst_st_g = sst_st.reshape(nt_lt, nlat*nlon)
        clf = linear_model.LinearRegression()
        clf.fit(indexstd.reshape(-1,1), sst_st_g)
        sstcorrs_st = clf.coef_.reshape(nlat, nlon)   
    
    
    
    
    #compute correlation between long-term/short-term index and 2D field
    print r'calculating correlations between index and {:s}...'.format(ftitle)
    for i in range(nlat):   
            print 'latitude', lats[i]
         
         #for j in range(nlon):
             
            sstprime_g = sstprime[:,i,:]
            fieldprime_g = fieldprime[:,i,:]
      
            
            field_lt_g = field_lt[:,i,:]
            field_st_g = field_st[:,i,:]
            sst_lt_g = sst_lt[:,i,:]
            sst_st_g = sst_st[:,i,:]
            
    
            N = sstprime_g.shape[1]
        
#            clf = linear_model.LinearRegression()
#            clf.fit(indexstd.reshape(-1,1), sstprime_g)
#            sstcorrs[i,:] = np.squeeze(clf.coef_)
            
#            clf = linear_model.LinearRegression()
#            clf.fit(indexstd.reshape(-1,1), fieldprime_g)
#            fieldcorrs[i,:] = np.squeeze(clf.coef_)
            
            #sstcoefs = np.diag(cov2_coeff(indexstdrep.T, sstprime_g.T))
            fieldcoefs = np.diag(cov2_coeff(indexstdrep.T, fieldprime_g.T))
            
            #sstcorrs[i,:] = sstcoefs
            fieldcorrs[i,:] = fieldcoefs
            

            if lterm:

#                clf = linear_model.LinearRegression()
#                clf.fit(indexstd_lt.reshape(-1,1), sst_lt_g)
#                sstcorrs_lt[i,:] = np.squeeze(clf.coef_)
#             
#                clf = linear_model.LinearRegression()
#                clf.fit(indexstd_st.reshape(-1,1), sst_st_g)
#                sstcorrs_st[i,:] = np.squeeze(clf.coef_)
                
                #sstcoefs_lt = np.diag(cov2_coeff(indexstd_ltrep.T, sst_lt_g.T))
                #sstcoefs_st = np.diag(cov2_coeff(indexstd_strep.T, sst_st_g.T))
                
                fieldcoefs_lt = np.diag(cov2_coeff(indexstd_ltrep.T, field_lt_g.T))
                fieldcoefs_st = np.diag(cov2_coeff(indexstd_strep.T, field_st_g.T))
            
                #sstcorrs_lt[i,:] = sstcoefs_lt
                #sstcorrs_st[i,:] = sstcoefs_st
                
                fieldcorrs_lt[i,:] = fieldcoefs_lt
                fieldcorrs_st[i,:] = fieldcoefs_st
    
    
    #Plot maps of SST and THF patterns associated with index
    #CHANGE THIS FOR MAP PROJECTION
    cent=-(lonbounds[1]-lonbounds[0])/2.
    prj = cart.crs.PlateCarree(central_longitude=cent)
    #prj = cart.crs.Mollweide(central_longitude=cent)
    
    bnds = [np.round(lonbounds[0]-359), np.round(lonbounds[1]-361), latbounds[0], latbounds[1]]
    
    #latitude/longitude labels
    
    pardiff = 30.
    merdiff = 60.
    if lonbounds[1] - lonbounds[0] <= 180:
        pardiff = 15.
        merdiff = 15.
    par = np.arange(-90.,91.,pardiff)
    mer = np.arange(-180.,180.,merdiff)
    
    lstep = 0.01
    levels = np.arange(-1.0, 1.0+lstep, lstep)
    x, y = np.meshgrid(lons, lats)
    pstep = 0.2
    sststep = 0.02
    
    if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
        fieldmin=-4
        fieldmax=4
        fieldstep = 0.05
        cbstep = 1.0
    elif fsave == 'sst':
        fieldmin=-0.8
        fieldmax=0.8
        fieldstep = 0.02
        cbstep = 0.2
    elif fsave == 'umag':
        fieldmin=-0.5
        fieldmax=0.5
        fieldstep=0.01
        cbstep = 0.1
    elif fsave == 'RH10m':
        fieldmin = -2
        fieldmax = 2
        fieldstep = 0.01
        cbstep = 0.5
    elif fsave == 'qvdiff':
        fieldmin = -2
        fieldmax = 2
        fieldstep = 0.01
        cbstep = 0.5
    elif fsave == 'cE':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep = 0.1
    elif fsave == 'cD':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep = 0.1
#    elif fsave == 'CREsurf' or fsave == 'Qnetsurfcs':
#        fieldmin = -5
#        fieldmax = 5
#        fieldstep = 0.1
#        cbstep = 2.5
    else:
        fieldmin=-10
        fieldmax=10
        fieldstep =0.2
        cbstep=2.5
        
    #NAlats = lats[NAminlati:NAmaxlati]
    #NEED TO AVERAGE OVER NA LONGITUDES
    #NAsst = spatial_ave(sst, lats)
    #NAsst_lt = spatial_ave(sst_lt, lats)
    
    NAfield = spatial_ave(fieldprime[:,si:ni,wi:ei], lats[si:ni])
    NAfield_lt = spatial_ave(field_lt[:,si:ni,wi:ei], lats[si:ni])
    
    
    
    #PLOT index INDEX AND FIELD
    fig=plt.figure(figsize=(14,14))
    ax = fig.add_subplot(211)
    #plt.figure()
    #ax=plt.gcf().gca()
    ax.plot(tyears, index)
    ax.plot(tyears, index_lt, label='{:1.0f}-yr LP'.format((Tn/12.)))
    #plt.plot(tyears[ci:-ci], index_st, label='short-term residual')
    #ax.fill_between(tyears[ci:-ci], 0, index_smooth, where= index_smooth>0, color='red')
    #ax.fill_between(tyears[ci:-ci], 0, index_smooth, where= index_smooth<0, color='blue')
    ax.set_title(r'SST$_{{{0}^\circ-{1}^\circ N}}$'.format(np.round(np.abs(lats[si]),0), np.round(lats[ni],0)))
    ax.axhline(0, color='black')
    ax.set_ylim(-0.70,0.70)
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    ax.set_xlabel('time (years)')
    ax.legend(loc='upper right')
    ax = fig.add_subplot(212)
    #plt.figure()
    #ax=plt.gcf().gca()
    ax.plot(tyears, NAfield)
    ax.plot(tyears, NAfield_lt, label='{:1.0f}-yr LP'.format((Tn/12.)))
    ax.legend(loc='upper right')
    #plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
    #ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
    #ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
    ax.set_title(r'{0}$_{{{1}^\circ-{2}^\circ N}}$'.format(ftitle, np.round(np.abs(lats[si]),0), np.round(lats[ni],0)))
    ax.axhline(0, color='black')
    ax.set_ylabel(r'{:s} ({:s})'.format(ftitle, units))
    #ax.legend()
    ax.set_xlabel('time (years)')
    plt.savefig(fout + '{:s}_SST{:s}{:2.0f}Nto{:2.0f}N_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, lats[si], lats[ni], latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
        
    ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
    ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
    ticklbls[ticklbls == -0.0] = 0.0
                     
    sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
    fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
    
    orient = 'horizontal'
    if lonbounds[1] - lonbounds[0] <= 180:
        orient = 'vertical'
        
    
    CTIindex_lagcorr = np.correlate(np.squeeze(CTIstd/len(CTIstd)), np.squeeze(indexstd), 'full')
    lagzero = len(CTIindex_lagcorr)/2
    lagmax = 12
    lags = np.arange(-lagmax, lagmax+1)
    lagoffset = np.diff(lags)[0]/2.
    lags = lags-lagoffset
    
    CTIindex_lagcorr = CTIindex_lagcorr[lagzero-lagmax:lagzero+lagmax+1]
    
    
    
    plt.figure(figsize=(10,8))
    plt.plot(lags, CTIindex_lagcorr)
    plt.ylim(-0.5,0.5)
    plt.axhline(0, color='k')
    plt.axvline(0, color='k')
    plt.xlabel('CTI lag (months)')
    plt.title('Correlation between CTI and SST$_{{{0}^\circ-{1}^\circ N}}$'.format(np.round(np.abs(lats[si]),0), np.round(lats[ni],0)))
    plt.ylabel('Correlation')
    plt.savefig(fout + '{:s}_CTI_SST{:2.0f}Nto{:2.0f}N_lagcorr_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, lats[si], lats[ni], latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    
    x1=-(eloni-wloni)/2.
    y1=slati
    x2=-(eloni-wloni)/2.
    y2=nlati
    x3=(eloni-wloni)/2.
    y3=nlati
    x4=(eloni-wloni)/2.
    y4=slati  
    
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
    #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, fieldcorrs, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())    
    cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
    if drawbox:
        poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
        ax.add_patch(poly)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    plt.title(r'{:s}'.format(ftitle))
    plt.savefig(fout + '{:s}_SST{:2.0f}Nto{:2.0f}N_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    if lterm:
    
    
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
        #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
        plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        if drawbox:
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
            ax.add_patch(poly)
        plt.title(r'Long-term {:s} ({:1.0f}-yr LP)'.format(ftitle, Tn/12.))
        plt.savefig(fout + '{:s}_SST{:2.0f}Nto{:2.0f}N_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
        plt.close()
        
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
        #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
        plot = ax.contourf(x, y, fieldcorrs_st, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        if drawbox:
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
            ax.add_patch(poly)
        plt.title(r'Short-term {:s} ({:1.0f}-yr HP)'.format(ftitle, Tn/12.))
        plt.savefig(fout + '{:s}_SST{:2.0f}Nto{:2.0f}N_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname,  lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
        plt.close()
        
#    np.ma.dump(sstcorrs, fin + 'deltaSST_SST{0}to{1}_detr{2}_{3}x{4}'.format(np.round(lats[si],0), np.round(lats[ni],0), str(detr)[0], latdiff, londiff))
#    if lterm:
#        np.ma.dump(sstcorrs_lt, fin + 'deltaSST_SST{0}to{1}_{2}LP_detr{3}_{4}x{5}'.format(np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff))
#        np.ma.dump(sstcorrs_st, fin + 'deltaSST_SST{0}to{1}_{2}HP_detr{3}_{4}x{5}.npy'.format(np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff))
#    if fsave == 'qv3D':
#        #convert back to kg/kg
#        fieldcorrs = fieldcorrs*1e-3
#        fieldcorrs_lt = fieldcorrs_lt*1e-3
#        fieldcorrs_st = fieldcorrs_st*1e-3
        
#    np.ma.set_fill_value(sstcorrs, np.nan)
#    sstcorrs_save = sstcorrs.getValue()
#    
#    np.ma.set_fill_value(sstcorrs_lt, np.nan)
#    sstcorrs_ltsave = sstcorrs_lt.getValue()
#    
#    np.ma.set_fill_value(sstcorrs_st, np.nan)
#    sstcorrs_stsave = sstcorrs_st.getValue()
        
    #field_mask = fieldcorrs.mask
    np.ma.set_fill_value(fieldcorrs, np.nan)
    fieldcorrs_save = fieldcorrs.getValue()
    #fieldcorrs_save = fieldcorrs_save[field_mask] = np.nan
    
    #field_ltmask = fieldcorrs_lt.mask
    np.ma.set_fill_value(fieldcorrs_lt, np.nan)
    fieldcorrs_ltsave = fieldcorrs_lt.getValue()
    #fieldcorrs_ltsave = fieldcorrs_ltsave[field_ltmask] = np.nan
    
    #field_stmask = fieldcorrs_st.mask
    np.ma.set_fill_value(fieldcorrs_st, np.nan)
    fieldcorrs_stsave = fieldcorrs_st.getValue()
    #fieldcorrs_stsave = fieldcorrs_stsave[field_stmask] = np.nan
    
    np.savez(fin + 'deltaSST_SST{0}to{1}_detr{2}_{3}x{4}'.format(np.round(lats[si],0), np.round(lats[ni],0), str(detr)[0], latdiff, londiff), sst=sstcorrs, lats=lats, lons=lons)
    if lterm:
        np.savez(fin + 'deltaSST_SST{0}to{1}_{2}LP_detr{3}_{4}x{5}'.format(np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), sst=sstcorrs_lt, lats=lats, lons=lons)
        np.savez(fin + 'deltaSST_SST{0}to{1}_{2}HP_detr{3}_{4}x{5}'.format(np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), sst=sstcorrs_st, lats=lats, lons=lons)
    if fsave == 'qv3D':
        #convert back to kg/kg
        fieldcorrs = fieldcorrs*1e-3
        fieldcorrs_lt = fieldcorrs_lt*1e-3
        fieldcorrs_st = fieldcorrs_st*1e-3
    np.save(fin + 'delta{0}_SST{1}to{2}_detr{3}_{4}x{5}'.format(fsave, np.round(lats[si],0), np.round(lats[ni],0), str(detr)[0], latdiff, londiff), fieldcorrs_save)
    if lterm:
        np.save(fin + 'delta{0}_SST{1}to{2}_{3}LP_detr{4}_{5}x{6}'.format(fsave, np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), fieldcorrs_ltsave)
        np.save(fin + 'delta{0}_SST{1}to{2}_{3}HP_detr{4}_{5}x{6}'.format(fsave, np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), fieldcorrs_stsave)
 












































