#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats.stats import pearsonr, linregress
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from ocean_atmosphere.misc_fns import an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common
from matplotlib.patches import Polygon

#sns.palplot(sns.color_palette("coolwarm", 7))

#fin = '/Users/cpatrizio/data/ECMWF/'
fin = '/Users/cpatrizio/data/MERRA2/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

#dataname = 'ERAi'
dataname = 'MERRA2'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')

#ERAi
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')
#fsst = cdms2.open(fin + 'sstslp.197901-201712.nc')
#fuv = cdms2.open(fin + 'uv.197901-201712.nc')


matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
#matplotlib.rcParams.update({'figure.autolayout': True})

maxlat = 70
minlat = -70

maxlon = 360
minlon = 0

tskip = 6

ps = fSLP('SLP') 
#ps = fsst('msl')/1e2
ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
ps = ps[tskip:,:]/1e2

#sst = fsst('sst')
sst = fsst('TSKINWTR')
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:,:]
#cdutil.setTimeBoundsMonthly(sst)
#lhf and shf are accumulated over 12 hours (units of J/m^2)
#lhf = fthf('slhf')/(12*60*60)
lhf = fsst('EFLUXWTR')
lhf = lhf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
lhf = lhf[tskip:,:]
#cdutil.setTimeBoundsMonthly(lhf)
#shf = fthf('sshf')/(12*60*60)
shf = fsst('HFLUXWTR')
shf = shf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
shf = shf[tskip:,:]

u = fuv('U10M')
#u = fuv('u10')
u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
u = u[tskip:,:]

v = fuv('V10M')
#v = fuv('v10')
v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
v = v[tskip:,:]

#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)
thf = lhf + shf
#convert to positive down, to be consistent with surface radiation
#thf = -thf

#thf is positive down in ERAi
#thf = -thf 

#masking continents for SLP field
sst_mask = np.ma.getmaskarray(sst)
ps_mask = np.ma.getmaskarray(ps)
ps_mask = np.ma.mask_or(sst_mask[:ps.shape[0],:,:], ps_mask)
ps = np.ma.array(ps, mask=ps_mask)

#True for detrending data, False for raw data
detr=True
#True for zonally averaged lagged correlation, False for zonally averaged regression
corr=True


lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]
nt = sst.shape[0]
lons[0] = 0
nlat = len(lats)
nlon = len(lons)

t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1980 + t/12.

tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#initial/final indices for base period
baseti = 0
basetf = 10

sst_an = an_ave(sst)
thf_an = an_ave(thf)
ps_an = an_ave(ps)
u_an = an_ave(u)
v_an = an_ave(v)

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst_an, params = detrend_separate(sst_an)
 ps_an, params = detrend_separate(ps_an)
 thf_an, params = detrend_separate(thf_an)
 u_an, params = detrend_separate(u_an)
 v_an, params = detrend_separate(v_an)
 
thf_globe_an = spatial_ave(thf_an, lats)
ps_globe_an = spatial_ave(ps_an, lats)
sst_globe_an = spatial_ave(sst_an, lats)
u_globe_an = spatial_ave(u_an, lats)
v_globe_an = spatial_ave(v_an, lats)

#subtract global annual mean to isolate processes in NA
sstprime = sst_an.T - sst_globe_an
sstprime = sstprime.T
thfprime = thf_an.T - thf_globe_an
thfprime = thfprime.T
psprime = ps_an.T - ps_globe_an
psprime = psprime.T
uprime  = u_an.T - u_globe_an
uprime = uprime.T
vprime = v_an.T - v_globe_an
vprime = vprime.T

#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5
ci = (N_map-1)/2
ltlag = 7
stlag = 1

lagmax=11
lags = np.arange(-lagmax,lagmax+1)

#bounds for AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
#latboundsar = np.array([[0,20], [20,45], [45,60]])
#latboundsar = np.array([[0,60], [0,20], [20,45], [45,60]])
latboundsar = np.array([[0,60]])

for latbounds in latboundsar:


    AMO, sstanom_globe_an, sstanom_na_an = calc_NA_globeanom(sst_an, latbounds, lats, lons, baseti, basetf)
    NAthf2, thfanom_globe_an, thfanom_na_an = calc_NA_globeanom(thf_an, latbounds, lats, lons, baseti, basetf)
     
    
    AMO_lt = running_mean(AMO, N_map)
    AMO_st = AMO[ci:-ci] - AMO_lt
    
    scaler = StandardScaler()
    AMOstd = scaler.fit_transform(AMO.reshape(-1,1))
    AMOstd_lt = scaler.fit_transform(AMO_lt.reshape(-1,1))
    AMOstd_st = scaler.fit_transform(AMO_st.reshape(-1,1))
    
    #need to normalize in this manner in order to get pearson correlation coefficient from np.correlate
    AMOstd2 = (AMO - np.mean(AMO)) / (np.std(AMO) * len(AMO))
    AMOstd_lt2 = (AMO_lt - np.mean(AMO_lt)) / (np.std(AMO_lt) * len(AMO_lt))
    AMOstd_st2 = (AMO_st - np.mean(AMO_st)) / (np.std(AMO_st) * len(AMO_st))
    
        
    sst_lt = running_mean(sstprime, N_map)
    sst_st = sstprime[ci:-ci,:] - sst_lt
    
    thf_lt = running_mean(thfprime, N_map)
    thf_st = thfprime[ci:-ci,:] - thf_lt
    
    ps_lt = running_mean(psprime, N_map)
    ps_st = psprime[ci:-ci,:] - ps_lt
    
    u_lt = running_mean(uprime, N_map)
    u_st = uprime[ci:-ci,:] - u_lt
    
    v_lt = running_mean(vprime, N_map)
    v_st = vprime[ci:-ci,:] - v_lt
    
    nt = sst_an.shape[0]
    nt_lt = thf_lt.shape[0]
    
    
    sstcorrs = MV.zeros((nlat,nlon))
    sstcorrs_lt = MV.zeros((nlat, nlon))
    sstcorrs_st = MV.zeros((nlat, nlon))
    thfcorrs = MV.zeros((nlat, nlon))
    thfcorrs_lt = MV.zeros((nlat,nlon))
    thfcorrs_st = MV.zeros((nlat,nlon))
    pscorrs = MV.zeros((nlat, nlon))
    pscorrs_lt = MV.zeros((nlat,nlon))
    pscorrs_st = MV.zeros((nlat,nlon))
    ucorrs = MV.zeros((nlat,nlon))
    ucorrs_lt = MV.zeros((nlat,nlon))
    ucorrs_st = MV.zeros((nlat,nlon))
    vcorrs = MV.zeros((nlat,nlon))
    vcorrs_lt = MV.zeros((nlat,nlon))
    vcorrs_st = MV.zeros((nlat,nlon))

    sstlagcorrs = np.zeros((len(lags), nlat, nlon))
    sstlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
    sstlagcorrs_st = np.zeros((len(lags), nlat, nlon))
    thflagcorrs = np.zeros((len(lags), nlat, nlon))
    thflagcorrs_lt = np.zeros((len(lags), nlat, nlon))
    thflagcorrs_st = np.zeros((len(lags), nlat, nlon))
    pslagcorrs = np.zeros((len(lags), nlat, nlon))
    pslagcorrs_lt = np.zeros((len(lags), nlat, nlon))
    pslagcorrs_st = np.zeros((len(lags), nlat, nlon))
    
    #compute correlation between long-term/short-term AMO and sst, thf, SLP
    #also compute lagged correlation between THF and AMO
    #todo: compute lagged correlation between SLP and AMO
    print 'calculating correlations between AMO and THF, SLP...'
    for i in range(nlat):         
    
         print 'latitude', lats[i]
       
    
         sstprime_g = sstprime[:,i,:]
         sst_lt_g = sst_lt[:,i,:]
         sst_st_g = sst_st[:,i,:]
         thfprime_g = thfprime[:,i,:]
         psprime_g = psprime[:,i,:]
         thf_lt_g = thf_lt[:,i,:]
         thf_st_g = thf_st[:,i,:]
         u_lt_g = u_lt[:,i,:]
         u_st_g = u_st[:,i,:]
         v_lt_g = v_lt[:,i,:]
         v_st_g = v_st[:,i,:]
         ps_lt_g = ps_lt[:,i,:]
         ps_st_g = ps_st[:,i,:]
         uprime_g = uprime[:,i,:]
         vprime_g = vprime[:,i,:]
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd.reshape(-1,1), sstprime_g)
         sstcorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_lt.reshape(-1,1), sst_lt_g)
         sstcorrs_lt[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_st.reshape(-1,1), sst_st_g)
         sstcorrs_st[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd.reshape(-1,1), uprime_g)
         ucorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_lt.reshape(-1,1), u_lt_g)
         ucorrs_lt[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_st.reshape(-1,1), u_st_g)
         ucorrs_st[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd.reshape(-1,1), vprime_g)
         vcorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_lt.reshape(-1,1), v_lt_g)
         vcorrs_lt[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_st.reshape(-1,1), v_st_g)
         vcorrs_st[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd.reshape(-1,1), psprime_g)
         pscorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_lt.reshape(-1,1), ps_lt_g)
         pscorrs_lt[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_st.reshape(-1,1), ps_st_g)
         pscorrs_st[i,:] = np.squeeze(clf.coef_)
        
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd.reshape(-1,1), thfprime_g)
         thfcorrs[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_lt.reshape(-1,1), thf_lt_g)
         thfcorrs_lt[i,:] = np.squeeze(clf.coef_)
         
         clf = linear_model.LinearRegression()
         clf.fit(AMOstd_st.reshape(-1,1), thf_st_g)
         thfcorrs_st[i,:] = np.squeeze(clf.coef_)
         
         for lag in lags:
             
             scaler = StandardScaler()
             
             if corr:
                 # CALCULATE CORERLATION COEFFICIENT
                 sststd = scaler.fit_transform(sstprime_g)
                 sststd_lt = scaler.fit_transform(sst_lt_g)
                 sststd_st = scaler.fit_transform(sst_st_g)
                 thfstd = scaler.fit_transform(thfprime_g)
                 thfstd_lt = scaler.fit_transform(thf_lt_g)
                 thfstd_st = scaler.fit_transform(thf_st_g)
                 psstd = scaler.fit_transform(psprime_g)
                 psstd_lt = scaler.fit_transform(ps_lt_g)
                 psstd_st = scaler.fit_transform(ps_st_g)
                 #need to divide by length of time series for computing normalized lagged correlation?
             else:
                 # CALCULATE REGRESSION
                 sststd = sstprime_g
                 sststd_lt = sst_lt_g
                 sststd_st = sst_st_g
                 thfstd = thfprime_g
                 thfstd_lt = thf_lt_g
                 thfstd_st = thf_st_g
                 psstd = psprime_g
                 psstd_lt = ps_lt_g
                 psstd_st = ps_st_g
             
             sstclf = linear_model.LinearRegression()
             sstclf_lt = linear_model.LinearRegression()
             sstclf_st = linear_model.LinearRegression()
             thfclf = linear_model.LinearRegression()
             thfclf_lt = linear_model.LinearRegression()
             thfclf_st = linear_model.LinearRegression()
             psclf = linear_model.LinearRegression()
             psclf_lt = linear_model.LinearRegression()
             psclf_st = linear_model.LinearRegression()
             #THF LAGS SST
             if lag > 0:
                sstclf.fit(AMOstd[:-lag], sststd[lag:,:])
                sstclf_lt.fit(AMOstd_lt[:-lag], sststd_lt[lag:,:])
                sstclf_st.fit(AMOstd_lt[:-lag], sststd_st[lag:,:])
                thfclf.fit(AMOstd[:-lag], thfstd[lag:,:])
                thfclf_lt.fit(AMOstd_lt[:-lag], thfstd_lt[lag:,:])
                thfclf_st.fit(AMOstd_st[:-lag], thfstd_st[lag:,:])
                psclf.fit(AMOstd[:-lag], psstd[lag:,:])
                psclf_lt.fit(AMOstd_lt[:-lag], psstd_lt[lag:,:])
                psclf_st.fit(AMOstd_st[:-lag], psstd_st[lag:,:])
            #THF LEADS SST
             elif lag < 0: 
                sstclf.fit(AMOstd[-lag:], sststd[:lag,:])
                sstclf_lt.fit(AMOstd_lt[-lag:], sststd_lt[:lag,:])
                sstclf_st.fit(AMOstd_st[-lag:], sststd_st[:lag,:])
                thfclf.fit(AMOstd[-lag:], thfstd[:lag,:])
                thfclf_lt.fit(AMOstd_lt[-lag:], thfstd_lt[:lag,:])
                thfclf_st.fit(AMOstd_st[-lag:], thfstd_st[:lag,:])
                psclf.fit(AMOstd[-lag:], psstd[:lag,:])
                psclf_lt.fit(AMOstd_lt[-lag:], psstd_lt[:lag,:])
                psclf_st.fit(AMOstd_st[-lag:], psstd_st[:lag,:])
             else:
                sstclf.fit(AMOstd, sststd)
                sstclf_lt.fit(AMOstd_lt, sststd_lt)               
                sstclf_st.fit(AMOstd_st, sststd_st)
                thfclf.fit(AMOstd, thfstd)
                thfclf_lt.fit(AMOstd_lt, thfstd_lt)
                thfclf_st.fit(AMOstd_st, thfstd_st)
                psclf.fit(AMOstd, psstd)
                psclf_lt.fit(AMOstd_lt, psstd_lt)
                psclf_st.fit(AMOstd_st, psstd_st)
                
             sstlagcorrs[lag+lagmax,i,:] = np.squeeze(sstclf.coef_)   
             sstlagcorrs_lt[lag+lagmax,i,:] = np.squeeze(sstclf_lt.coef_)   
             sstlagcorrs_st[lag+lagmax,i,:] = np.squeeze(sstclf_st.coef_)   
             thflagcorrs[lag+lagmax,i,:] = np.squeeze(thfclf.coef_)
             thflagcorrs_lt[lag+lagmax,i,:] = np.squeeze(thfclf_lt.coef_)
             thflagcorrs_st[lag+lagmax,i,:] = np.squeeze(thfclf_st.coef_)
             pslagcorrs[lag+lagmax,i,:] = np.squeeze(psclf.coef_)
             pslagcorrs_lt[lag+lagmax,i,:] = np.squeeze(psclf_lt.coef_)
             pslagcorrs_st[lag+lagmax,i,:] = np.squeeze(psclf_st.coef_)
             
    lonbounds = [280,359.99]
    
    NAminlati = np.where(lats > latbounds[0])[0][0]
    NAmaxlati = np.where(lats > latbounds[1])[0][0]
    NAminloni = np.where(lons > lonbounds[0])[0][0]
    NAmaxloni = np.where(lons > lonbounds[1])[0][0]
    
    NAlats = lats[NAminlati:NAmaxlati]
    #NEED TO AVERAGE OVER NA LONGITUDES
    NAthf = spatial_ave(thfprime[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], NAlats)
    NAps = spatial_ave(psprime[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], NAlats)
    
    windows = np.arange(3,lagmax+1,2)
    ll, ww = np.meshgrid(lags, windows)
    
    NAthflagcorrs = np.zeros((len(windows), len(lags)))
    NAthflagcorrs_lt = np.zeros((len(windows), len(lags)))
    NAthflagcorrs_st = np.zeros((len(windows), len(lags)))
    NApslagcorrs = np.zeros((len(windows), len(lags)))
    NApslagcorrs_lt = np.zeros((len(windows), len(lags)))
    NApslagcorrs_st = np.zeros((len(windows), len(lags)))
    
    print 'calculating lagged correlation between AMO and NA THF, SLP for different RM windows...'
    #commpute lagged correlation between smoothed AMO and NA THF for different RM windows
    for k, N in enumerate(windows):
        
        if corr:
            
            ci = (N-1)/2
            AMO_lt = running_mean(AMO, N)
            AMO_st = AMO[ci:-ci] - AMO_lt
            AMOstd = (AMO - np.mean(AMO)) / (np.std(AMO) * len(AMO))
            AMOstd_lt = (AMO_lt - np.mean(AMO_lt)) / (np.std(AMO_lt) * len(AMO_lt))
            AMOstd_st = (AMO_st - np.mean(AMO_st)) / (np.std(AMO_st) * len(AMO_st))
            
            
            NAthf_lt = running_mean(NAthf, N)
            NAthf_st = NAthf[ci:-ci] - NAthf_lt 
            NAthfstd = (NAthf - np.mean(NAthf)/np.std(NAthf))
            NAthfstd_lt = (NAthf_lt - np.mean(NAthf_lt)) / (np.std(NAthf_lt))
            NAthfstd_st = (NAthf_st - np.mean(NAthf_st)) / (np.std(NAthf_st))
            #NAthflaggedcorr_temp = np.correlate(NAthfstd, AMOstd, 'full')
            NAthflaggedcorr_lt_temp = np.correlate(NAthfstd_lt, AMOstd_lt, 'full')
            NAthflaggedcorr_st_temp = np.correlate(NAthfstd_st, AMOstd_st, 'full')
        
            NAps_lt = running_mean(NAps, N)
            NAps_st = NAps[ci:-ci] - NAps_lt 
            NApsstd = (NAps - np.mean(NAps)/np.std(NAps))
            NApsstd_lt = (NAps_lt - np.mean(NAps_lt)) / (np.std(NAps_lt))
            NApsstd_st = (NAps_st - np.mean(NAps_st)) / (np.std(NAps_st))
            #NApslaggedcorr_temp = np.correlate(NApsstd, AMOstd, 'full')
            NApslaggedcorr_lt_temp = np.correlate(NApsstd_lt, AMOstd_lt, 'full')
            NApslaggedcorr_st_temp = np.correlate(NApsstd_st, AMOstd_st, 'full')
            
            
            lagzero = len(NAthflaggedcorr_lt_temp)/2
            #NAthflagcorrs[k,:] = NAthflaggedcorr_temp[lagzero-lagmax:lagzero+lagmax+1]
            NAthflagcorrs_lt[k,:] = NAthflaggedcorr_lt_temp[lagzero-lagmax:lagzero+lagmax+1]
            NAthflagcorrs_st[k,:] = NAthflaggedcorr_st_temp[lagzero-lagmax:lagzero+lagmax+1]
            #NApslagcorrs[k,:] = NApslaggedcorr_temp[lagzero-lagmax:lagzero+lagmax+1]
            NApslagcorrs_lt[k,:] = NApslaggedcorr_lt_temp[lagzero-lagmax:lagzero+lagmax+1]
            NApslagcorrs_st[k,:] = NApslaggedcorr_st_temp[lagzero-lagmax:lagzero+lagmax+1]
                
        else:
            for j, lag in enumerate(lags):

             ci = (N-1)/2 
             
             scaler = StandardScaler()
             
             AMO_lt = running_mean(AMO, N)
             AMO_st = AMO[ci:-ci] - AMO_lt
             #AMOstd = (AMO - np.mean(AMO) / np.std(AMO)).reshape(-1,1)
             AMOstd_lt = scaler.fit_transform(AMO_lt.reshape(-1,1))
             AMOstd_st = scaler.fit_transform(AMO_st.reshape(-1,1))
             
             NAthf_lt_temp = running_mean(NAthf, N)
             NAthf_st_temp = NAthf[ci:-ci] - NAthf_lt_temp
             NAps_lt_temp = running_mean(NAps, N)
             NAps_st_temp = NAps[ci:-ci] - NAps_lt_temp
             
             #thfstd = NAthf
             thfstd_lt = NAthf_lt_temp.reshape(-1,1)
             thfstd_st = NAthf_st_temp.reshape(-1,1)
             #psstd = NAps
             psstd_lt = NAps_lt_temp.reshape(-1,1)
             psstd_st = NAps_st_temp.reshape(-1,1)
             
             #thfclf = linear_model.LinearRegression()
             thfclf_lt = linear_model.LinearRegression()
             thfclf_st = linear_model.LinearRegression()
             #psclf = linear_model.LinearRegression()
             psclf_lt = linear_model.LinearRegression()
             psclf_st = linear_model.LinearRegression()
            
            #THF LAGS SST
             if lag > 0:
                #thfclf.fit(AMOstd[:-lag], thfstd[lag:,...])
                thfclf_lt.fit(AMOstd_lt[:-lag], thfstd_lt[lag:,...])
                thfclf_st.fit(AMOstd_st[:-lag], thfstd_st[lag:,...])
                #psclf.fit(AMOstd[:-lag], psstd[lag:,...])
                psclf_lt.fit(AMOstd_lt[:-lag], psstd_lt[lag:,...])
                psclf_st.fit(AMOstd_st[:-lag], psstd_st[lag:,...])
            #THF LEADS SST
             elif lag < 0: 
                #thfclf.fit(AMOstd[-lag:], thfstd[:lag,...])
                thfclf_lt.fit(AMOstd_lt[-lag:], thfstd_lt[:lag,...])
                thfclf_st.fit(AMOstd_st[-lag:], thfstd_st[:lag,...])
                #psclf.fit(AMOstd[-lag:], psstd[:lag,...])
                psclf_lt.fit(AMOstd_lt[-lag:], psstd_lt[:lag,...])
                psclf_st.fit(AMOstd_st[-lag:], psstd_st[:lag,...])
             else:
                #thfclf.fit(AMOstd, thfstd)
                thfclf_lt.fit(AMOstd_lt, thfstd_lt)
                thfclf_st.fit(AMOstd_st, thfstd_st)
                #psclf.fit(AMOstd, psstd)
                psclf_lt.fit(AMOstd_lt, psstd_lt)
                psclf_st.fit(AMOstd_st, psstd_st)
            

             #NAthflagcorrs[k,lag+lagmax] = np.squeeze(thfclf.coef_)
             NAthflagcorrs_lt[k,lag+lagmax] = np.squeeze(thfclf_lt.coef_)
             NAthflagcorrs_st[k,lag+lagmax] = np.squeeze(thfclf_st.coef_)
             #NApslagcorrs[k,lag+lagmax] = np.squeeze(psclf.coef_)
             NApslagcorrs_lt[k,lag+lagmax] = np.squeeze(psclf_lt.coef_)
             NApslagcorrs_st[k,lag+lagmax] = np.squeeze(psclf_st.coef_)
       
        
    #Plot AMO
    fig=plt.figure(figsize=(16,14))
    fig.tight_layout()
    ax = fig.add_subplot(311)
    plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
    ax.plot(tyears, sstanom_globe_an)
    if detr:
        ax.set_ylim(-0.5,0.5)
    else:
        ax.set_ylim(-1,1)
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    #ax.fill_between(tyears, 0, sstanom_globe_an, where= sstanom_globe_an>0, color='red')
    #ax.fill_between(tyears, 0, sstanom_globe_an, where= sstanom_globe_an<0, color='blue')
    ax.set_title(r'global mean SST (base period: {:3.0f} to {:3.0f})'.format(tyears[baseti], tyears[basetf]))
    ax.axhline(0, color='black')
    ax = fig.add_subplot(312)
    ax.plot(tyears, sstanom_na_an)
    #ax.fill_between(tyears, 0, sstanom_na_an, where= sstanom_na_an>0, color='red')
    #ax.fill_between(tyears, 0, sstanom_na_an, where= sstanom_na_an<0, color='blue')
    if detr:
        ax.set_ylim(-0.5,0.5)
    else:
        ax.set_ylim(-1,1)
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    ax.axhline(0, color='black')
    ax.set_title(r'NA mean SST')
    #plt.savefig(fout + 'MERRA2_global_NA_SST_anomaly_timeseries.pdf')
    #plt.close()
    
    ci = (N_map-1)/2
    AMO_smooth = running_mean(AMO, N_map)
    AMO_st = AMO[ci:-ci] - AMO_smooth
    ax = fig.add_subplot(313)
    #plt.figure()
    #ax=plt.gcf().gca()
    ax.plot(tyears, AMO, label='AMO')
    ax.plot(tyears[ci:-ci],AMO_smooth, label='{:1.0f}-yr RM'.format(N_map))
    #plt.plot(tyears[ci:-ci], AMO_st, label='short-term residual')
    #ax.fill_between(tyears[ci:-ci], 0, AMO_smooth, where= AMO_smooth>0, color='red')
    #ax.fill_between(tyears[ci:-ci], 0, AMO_smooth, where= AMO_smooth<0, color='blue')
    ax.set_title(r'AMO (NA mean SST - global mean SST)'.format(latbounds[0], latbounds[1]))
    ax.axhline(0, color='black')
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    ax.legend()
    ax.set_xlabel('time (years)')
    plt.savefig(fout + '{:s}_AMO_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    #Plot NA THF
    fig=plt.figure(figsize=(16,14))
    plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
    ax = fig.add_subplot(311)
    ax.plot(tyears, thfanom_globe_an)
    ax.axhline(0, color='black')
    ax.set_ylabel(r'THF (W m$^{{-2}}$)')
    #ax.set_ylim(-1,1)
    ax.set_title(r'global mean THF (base period: {:3.0f} to {:3.0f})'.format(tyears[baseti], tyears[basetf]))
    ax = fig.add_subplot(312)
    ax.plot(tyears, thfanom_na_an)
    ax.set_ylabel(r'THF (W m$^{{-2}}$)')
    #ax.set_ylim(-1,1)
    ax.axhline(0, color='black')
    ax.set_title(r'NA mean THF')
    #plt.savefig(fout + 'MERRA2_global_NA_thf_anomaly_timeseries.pdf')
    #plt.close()
    
    NAthf_smooth = running_mean(NAthf2, N_map)
    ax = fig.add_subplot(313)
    ax.plot(tyears, NAthf2, label = 'NA THF')
    ax.plot(tyears[ci:-ci], NAthf_smooth, label='{:1.0f}-yr RM'.format(N_map))
    ax.set_ylabel(r'THF (W m$^{{-2}}$)')
    plt.title(r'NA mean THF - global mean THF')
    plt.axhline(0, color='black')
    ax.set_xlabel('time (years)')
    ax.legend()
    plt.savefig(fout + '{:s}_thfanom_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    lagoffset = np.diff(lags)[0]/2.
    woffset = np.diff(windows)[0]/2.
    latoffset = np.diff(lats)[0]/2.
    
    ll, ww = np.meshgrid(lags-lagoffset, windows-woffset)
    
    laglabels = np.round(np.arange(-10,15,5))
    
    psmax=0.4
    psmin = -0.4
    psstep = 0.01
    pscbstep = 0.1
    sststep = 0.02
    thfmin=-3
    thfmax=3
    thfstep =0.02
    thfcbstep=1.0
    sstcbstep = 0.2
    SLPlevels = np.arange(psmin, psmax+psstep, psstep)
    sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
    thflevels = np.arange(thfmin,thfmax+thfstep, thfstep)
    
    if corr:
        sstminlag = -1
        sstmaxlag = 1
        thfminlag = -1
        thfmaxlag = 1
        psminlag = -1
        psmaxlag = 1
        thfcbsteplag = 0.2
        pscbsteplag = 0.2
        sstcbsteplag = 0.2
        thfunitslag = ''
        psunitslag = ''
        sstunitslag = ''
    else:
        sstminlag = -0.8
        sstmaxlag = 0.8
        thfminlag = thfmin
        thfmaxlag = thfmax
        psminlag = psmin
        psmaxlag = psmax
        thfcbsteplag = thfcbstep
        pscbsteplag = pscbstep 
        sstcbsteplag = sstcbstep
        thfunitslag = r'W m$^{-2}$'
        psunitslag = 'hPa'
        sstunitslag = 'K'


    thfticks = np.round(np.arange(thfminlag,thfmaxlag+thfcbstep,thfcbsteplag),2)
    thfticklbls = np.round(np.arange(thfminlag,thfmaxlag+thfcbstep,thfcbsteplag), 2)
    thfticklbls[thfticklbls == -0.00] = 0.00
    
    psticks = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag),2)
    psticklbls = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag), 2)
    psticklbls[psticklbls == -0.00] = 0.00
    
    sstticks = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag),2)
    sstticklbls = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag), 2)
    sstticklbls[sstticklbls == -0.00] = 0.00
    
        
    #Plot correlation between smoothed THF and AMO at different lags and different smoothing window lengths
    fig=plt.figure(figsize=(18,14))
#    ax = fig.add_subplot(221)
#    ax.pcolor(ll, ww, NAthflagcorrs, vmin=thfminlag, vmax=thfmaxlag, cmap=plt.cm.RdBu_r)
#    if corr:
#        ax.set_title('correlation of AMO with NA THF')
#    else:
#        ax.set_title('regression of NA THF on AMO')
#    ax.set_xlabel('THF lag (years)')
#    ax.set_ylabel('RM window (years)')
#    ax.axvline(0, color='k')
#    #ax.set_yticks(windows)
#    #ax.set_yticklabels(windows)
#    ax.set_xticks(laglabels)
#    ax.set_xticklabels(laglabels)
#    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(221)
    ax.pcolor(ll, ww, NAthflagcorrs_lt, vmin=thfminlag, vmax=thfmaxlag, cmap=plt.cm.RdBu_r)
    if corr:
        ax.set_title('long-term correlation of AMO with NA THF')
    else:
        ax.set_title('long-term regression of NA THF on AMO')
    ax.set_xlabel('THF lag (years)')
    ax.set_ylabel('RM window (years)')
    ax.axvline(0, color='k')
    #ax.set_yticks(windows)
    #ax.set_yticklabels(windows)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(222)
    h=ax.pcolor(ll, ww, NAthflagcorrs_st, vmin=thfminlag, vmax=thfmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('THF lag (years)')
    #ax.set_yticks(windows)
    #ax.set_yticklabels(windows)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #ax.set_ylabel('smoothing (years)') 
    cb=fig.colorbar(h, ax=ax, orientation="vertical", format='%1.2f', label = r'{:s}'.format(thfunitslag))
    cb.set_ticks(thfticks)
    cb.set_ticklabels(thfticklbls)
    #plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_hist_{:2.0f}Nto{:2.0f}N.pdf'.format(latbounds[0], latbounds[1]))
    #plt.close()
    
    i = np.where(windows>N_map)[0][0]-1
    
    #fig=plt.figure(figsize=(18,6))
    
#    ax = fig.add_subplot(223)
#    ax.plot(lags, NAthflagcorrs[i,:])
#    ax.axhline(0, color='black')
#    ax.axvline(0, color='black')
#    ax.set_ylim(thfminlag,thfmaxlag)
#    ax.set_ylabel(r'{:s}'.format(thfunitslag))
#    if corr:
#        ax.set_title('correlation of AMO with NA THF ({:1.0f}-yr RM)'.format(windows[i]))
#    else:
#        ax.set_title('regression of NA THF on AMO ({:1.0f}-yr RM)'.format(windows[i]))
#    ax.set_xlabel('THF lag (years)')
    ax = fig.add_subplot(223)
    ax.plot(lags, NAthflagcorrs_lt[i,:])
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(thfminlag,thfmaxlag)
    ax.set_ylabel(r'{:s}'.format(thfunitslag))
    if corr:
        ax.set_title('long-term correlation of AMO with NA THF ({:1.0f}-yr RM)'.format(windows[i]))
    else:
        ax.set_title('long-term regression of NA THF on AMO ({:1.0f}-yr RM)'.format(windows[i]))
    ax.set_xlabel('THF lag (years)')
    ax = fig.add_subplot(224)
    ax.plot(lags, NAthflagcorrs_st[i,:])
    ax.set_ylim(thfminlag,thfmaxlag)
    ax.set_ylabel(r'{:s}'.format(thfunitslag))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('THF lag (years)')
    if corr:
        plt.savefig(fout + '{:s}_AMO_thf_lagcorr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_thf_lagregr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    #plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_timeseries_{:1.0f}year_{:2.0f}Nto{:2.0f}N.pdf'.format(windows[i],latbounds[0], latbounds[1]))
    plt.close()
    

    fig=plt.figure(figsize=(18,14))
#    ax = fig.add_subplot(221)
#    ax.pcolor(ll, ww, NApslagcorrs, vmin=psminlag, vmax=psmaxlag, cmap=plt.cm.RdBu_r)
#    if corr:
#        ax.set_title('correlation of AMO with NA SLP')
#    else:
#        ax.set_title('regression of NA SLP on AMO')
#    ax.set_xlabel('SLP lag (years)')
#    ax.set_ylabel('RM window (years)')
#    ax.axvline(0, color='k')
#    #ax.set_yticks(windows)
#    #ax.set_yticklabels(windows)
#    ax.set_xticks(laglabels)
#    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(221)
    ax.pcolor(ll, ww, NApslagcorrs_lt, vmin=psminlag, vmax=psmaxlag, cmap=plt.cm.RdBu_r)
    if corr:
        ax.set_title('long-term correlation of AMO with NA SLP')
    else:
        ax.set_title('long-term regression of NA SLP on AMO')
    ax.set_xlabel('SLP lag (years)')
    ax.set_ylabel('RM window (years)')
    ax.axvline(0, color='k')
    #ax.set_yticks(windows)
    #ax.set_yticklabels(windows)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(222)
    h=ax.pcolor(ll, ww, NApslagcorrs_st, vmin=psminlag, vmax=psmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('SLP lag (years)')
    #ax.set_yticks(windows)
    #ax.set_yticklabels(windows)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #ax.set_ylabel('smoothing (years)') 
    cb=fig.colorbar(h, ax=ax, orientation="vertical", format='%1.2f', label=r'{:s}'.format(psunitslag))
    cb.set_ticks(psticks)
    cb.set_ticklabels(psticklbls)
    #plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_hist_{:2.0f}Nto{:2.0f}N.pdf'.format(latbounds[0], latbounds[1]))
    #plt.close()
    
    i = np.where(windows>N_map)[0][0]-1
    
    #fig=plt.figure(figsize=(18,6))
    
#    ax = fig.add_subplot(234)
#    ax.plot(lags, NApslagcorrs[i,:])
#    ax.axhline(0, color='black')
#    ax.axvline(0, color='black')
#    ax.set_ylim(psminlag,psmaxlag)
#    ax.set_ylabel(r'{:s}'.format(psunitslag))
#    if corr:
#        ax.set_title('correlation of AMO with NA SLP ({:1.0f}-yr RM)'.format(windows[i]))
#    else:
#        ax.set_title('regression of NA SLP on AMO ({:1.0f}-yr RM)'.format(windows[i]))
#    ax.set_xlabel('SLP lag (years)')
    ax = fig.add_subplot(223)
    ax.plot(lags, NApslagcorrs_lt[i,:])
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(psminlag,psmaxlag)
    ax.set_ylabel(r'{:s}'.format(psunitslag))
    if corr:
        ax.set_title('long-term correlation of AMO with NA SLP ({:1.0f}-yr RM)'.format(windows[i]))
    else:
        ax.set_title('long-term regression of NA SLP on AMO ({:1.0f}-yr RM)'.format(windows[i]))
    ax.set_xlabel('SLP lag (years)')
    ax = fig.add_subplot(224)
    ax.plot(lags, NApslagcorrs_st[i,:])
    ax.set_ylim(psminlag,psmaxlag)
    ax.set_ylabel(r'{:s}'.format(psunitslag))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('SLP lag (years)')
    if corr:
        plt.savefig(fout + '{:s}_AMO_SLP_lagcorr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_SLP_lagregr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    #plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_timeseries_{:1.0f}year_{:2.0f}Nto{:2.0f}N.pdf'.format(windows[i],latbounds[0], latbounds[1]))
    plt.close()
    
    psmax=0.5
    psmin = -0.5
    psstep = 0.01
    pscbstep = 0.1
    sststep = 0.02
    thfmin=-6
    thfmax=6
    thfstep =0.05
    thfcbstep=2.0
    sstcbstep = 0.2
    SLPlevels = np.arange(psmin, psmax+psstep, psstep)
    sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
    thflevels = np.arange(thfmin,thfmax+thfstep, thfstep)
    
    if corr:
        sstminlag = -1
        sstmaxlag = 1
        thfminlag = -1
        thfmaxlag = 1
        psminlag = -1
        psmaxlag = 1
        thfcbsteplag = 0.2
        pscbsteplag = 0.2
        sstcbsteplag = 0.2
        thfunitslag = ''
        psunitslag = ''
        sstunitslag = ''
    else:
        sstminlag = -0.8
        sstmaxlag = 0.8
        thfminlag = thfmin
        thfmaxlag = thfmax
        psminlag = psmin
        psmaxlag = psmax
        thfcbsteplag = thfcbstep
        pscbsteplag = pscbstep 
        sstcbsteplag = sstcbstep
        thfunitslag = r'W m$^{-2}$'
        psunitslag = 'hPa'
        sstunitslag = 'K'
        
    thfticks = np.round(np.arange(thfminlag,thfmaxlag+thfcbstep,thfcbsteplag),2)
    thfticklbls = np.round(np.arange(thfminlag,thfmaxlag+thfcbstep,thfcbsteplag), 2)
    thfticklbls[thfticklbls == -0.00] = 0.00
    
    psticks = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag),2)
    psticklbls = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag), 2)
    psticklbls[psticklbls == -0.00] = 0.00
    
    sstticks = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag),2)
    sstticklbls = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag), 2)
    sstticklbls[sstticklbls == -0.00] = 0.00

    weights = np.cos(np.deg2rad(lats))
    #thflagcorrs = np.ma.array(thflagcorrs, mask=~np.isfinite(thflagcorrs))
    #thflagcorrs_lt = np.ma.array(thflagcorrs_lt, mask=~np.isfinite(thflagcorrs_lt))
    #thflagcorrs_st = np.ma.array(thflagcorrs_st, mask=~np.isfinite(thflagcorrs_st))
    thflagcorrs_zonalave = np.ma.average(thflagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    thflagcorrs_lt_zonalave = np.ma.average(thflagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    thflagcorrs_st_zonalave = np.ma.average(thflagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    
    pslagcorrs_zonalave = np.ma.average(pslagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    pslagcorrs_lt_zonalave = np.ma.average(pslagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    pslagcorrs_st_zonalave = np.ma.average(pslagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    
    sstlagcorrs_zonalave = np.ma.average(sstlagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    sstlagcorrs_lt_zonalave = np.ma.average(sstlagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    sstlagcorrs_st_zonalave = np.ma.average(sstlagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    
    
    
    #SHOULDN'T THIS BE EQUIVALENT TO THE CORRELATION BETWEEN SMOOTHED AMO AND NA THF? i.e. NAthf_laggedcorr_lt[i,:]
    #thflagcorrs_test = np.ma.average(thflagcorrs_lt_zonalave, axis=1, weights=weights[NAminlati:NAmaxlati])
    
    lagg, latt = np.meshgrid(lags-lagoffset, NAlats-latoffset)
    
    #Plot zonally-averaged lagged correlation between long-term AMO and SST
    fig=plt.figure(figsize=(22,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(131)
    h = ax.pcolor(lagg, latt, sstlagcorrs_zonalave.T, vmin=sstminlag, vmax=sstmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('correlation between NA SST and AMO')
    else:
        ax.set_title('regression of NA sst on AMO')
    ax.set_xlabel('SST lag (years)')
    ax.set_ylabel('latitude (degrees)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(132)
    ax.pcolor(lagg, latt, sstlagcorrs_lt_zonalave.T, vmin=sstminlag, vmax=sstmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
    else:
        ax.set_title('long-term regression ({:1.0f}-yr RM)'.format(N_map))
    ax.set_xlabel('SST lag (years)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    ax = fig.add_subplot(133)
    h = ax.pcolor(lagg, latt, sstlagcorrs_st_zonalave.T, vmin=sstminlag, vmax=sstmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('SST lag (years)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    cb = fig.colorbar(h, ax=ax, orientation="vertical", format='%1.2f', label=r'{:s}'.format(sstunitslag))
    cb.set_ticks(sstticks)
    cb.set_ticklabels(sstticklbls)
    if corr:
         plt.savefig(fout + '{:s}_AMO_sst_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_sst_lagregr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()

    

    
    #Plot zonally-averaged lagged correlation between long-term AMO and THF
    fig=plt.figure(figsize=(22,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(131)
    h = ax.pcolor(lagg, latt, thflagcorrs_zonalave.T, vmin=thfminlag, vmax=thfmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('correlation between NA THF and AMO')
    else:
        ax.set_title('regression of NA THF on AMO')
    ax.set_xlabel('THF lag (years)')
    ax.set_ylabel('latitude (degrees)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(132)
    ax.pcolor(lagg, latt, thflagcorrs_lt_zonalave.T, vmin=thfminlag, vmax=thfmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
    else:
        ax.set_title('long-term regression ({:1.0f}-yr RM)'.format(N_map))
    ax.set_xlabel('THF lag (years)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    ax = fig.add_subplot(133)
    h = ax.pcolor(lagg, latt, thflagcorrs_st_zonalave.T, vmin=thfminlag, vmax=thfmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('THF lag (years)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    cb = fig.colorbar(h, ax=ax, orientation="vertical", format='%1.2f', label=r'{:s}'.format(thfunitslag))
    cb.set_ticks(thfticks)
    cb.set_ticklabels(thfticklbls)
    if corr:
         plt.savefig(fout + '{:s}_AMO_thf_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_thf_lagregr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    #Plot zonally-averaged lagged correlation between long-term AMO and THF
    fig=plt.figure(figsize=(22,6))
    #plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    ax = fig.add_subplot(131)
    h = ax.pcolor(lagg, latt, pslagcorrs_zonalave.T, vmin=psminlag, vmax=psmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('correlation between NA SLP and AMO')
    else:
        ax.set_title('regression of NA SLP on AMO')
    ax.set_xlabel('SLP lag (years)')
    ax.set_ylabel('latitude (degrees)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(132)
    h = ax.pcolor(lagg, latt, pslagcorrs_lt_zonalave.T, vmin=psminlag, vmax=psmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
    else:
        ax.set_title('long-term regression ({:1.0f}-yr RM)'.format(N_map))
    ax.set_xlabel('SLP lag (years)')
    ax.set_ylim(0,60)
    ax = fig.add_subplot(133)
    ax.pcolor(lagg, latt, pslagcorrs_st_zonalave.T, vmin=psminlag, vmax=psmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('SLP lag (years)')
    ax.set_ylim(0,60)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    cb = fig.colorbar(h, ax=ax, orientation="vertical", format='%1.2f', label='{:s}'.format(psunitslag))
    cb.set_ticks(psticks)
    cb.set_ticklabels(psticklbls)
    if corr:
        plt.savefig(fout + '{:s}_AMO_SLP_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_SLP_lagregr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    
    #Plot maps of SST and THF patterns associated with AMO
    #CHANGE THIS FOR MAP PROJECTION
    prj = cart.crs.PlateCarree()
    bnds = [-90, 0, -30, 70]
    
    #latitude/longitude labels
    par = np.arange(-90.,91.,15.)
    mer = np.arange(-180.,180.,15.)
    
    x, y = np.meshgrid(lons, lats)
    

    
    if dataname == 'ERAi':
        uskip=8
        pstep = 0.2
        sststep = 0.02
        thfstep = 0.5
        thfstep_lt = 5
        thfstep_st =5
        SLPlevels = np.arange(-2, 2+pstep, pstep)
        sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
        thflevels = np.arange(-15, 15+thfstep, thfstep)
    else:
        uskip=12
        pstep = 0.2
        sststep = 0.02
        thfstep = 0.5
        thfstep_lt = 5
        thfstep_st =5
        SLPlevels = np.arange(-2, 2+pstep, pstep)
        sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
        thflevels = np.arange(-10, 10+thfstep, thfstep)
        
    x1=lonbounds[0]-360
    y1=latbounds[0]
    x2=lonbounds[0]-360
    y2=latbounds[1]
    x3=lonbounds[1]-360
    y3=latbounds[1]
    x4=lonbounds[1]-360
    y4=latbounds[0]

    
    plt.figure(figsize=(16,12))
    ax = plt.axes(projection=prj)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=50, edgecolor='k', facecolor='grey')
    ax.set_xticks(mer, crs=prj)
    ax.set_yticks(par, crs=prj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1)
    ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
    plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
    qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs[::uskip,::uskip], vcorrs[::uskip,::uskip], transform=prj, scale_units='inches', scale = 1, width=0.001, headwidth=16, headlength=10, minshaft=4)
    ax.quiverkey(qv1, 0.95, 0.02, 0.5, '0.5 m/s')
    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
    #rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
    ax.add_patch(poly)
    plt.title(r'regression of SST, SLP and 10-m winds on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    #plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + '{:s}_AMO_sstSLPuv_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    #plt.savefig(fout + '{:s}_AMO_sst_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    plt.figure(figsize=(16,12))
    ax = plt.axes(projection=prj)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=50, edgecolor='k', facecolor='grey')
    ax.set_xticks(mer, crs=prj)
    ax.set_yticks(par, crs=prj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    ct = ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', linewidths=1)
    ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
    plot = ax.contourf(x, y, sstcorrs_lt, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
    ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_lt[::uskip,::uskip], vcorrs_lt[::uskip,::uskip], transform=prj, scale_units='inches', scale = 1, width=0.001, headwidth=16, headlength=10, minshaft=4)
    ax.quiverkey(qv1, 0.95, 0.02, 0.5, '0.5 m/s')
    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
    #rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
    ax.add_patch(poly)
    plt.title(r'long-term regression of SST, SLP and 10-m winds ({:1.0f}-yr RM) on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(N_map, latbounds[0], latbounds[1]))
    plt.savefig(fout + '{:s}_AMO_sstSLPuv_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    plt.figure(figsize=(16,12))
    ax = plt.axes(projection=prj)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=50, edgecolor='k', facecolor='grey')
    ax.set_xticks(mer, crs=prj)
    ax.set_yticks(par, crs=prj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    ct = ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', linewidths=1)
    ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
    plot = ax.contourf(x, y, sstcorrs_st, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
    ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_st[::uskip,::uskip], vcorrs_st[::uskip,::uskip], transform=prj, scale_units='inches', scale = 1, width=0.001, headwidth=16, headlength=10, minshaft=4)
    ax.quiverkey(qv1, 0.95, 0.02, 0.5, '0.5 m/s')
    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=1, zorder=100)
    #rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
    ax.add_patch(poly)
    plt.title(r'short-term regression of SST, SLP and 10-m winds on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + '{:s}_AMO_sstSLPuv_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    

    plt.figure(figsize=(16,12))
    ax = plt.axes(projection=prj)
    ax.coastlines()
    ax.add_feature(cart.feature.LAND, zorder=90, edgecolor='k', facecolor='grey')
    ax.set_xticks(mer, crs=prj)
    ax.set_yticks(par, crs=prj)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, thfcorrs, cmap=plt.cm.RdBu_r, levels=thflevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'W m$^{-2}$')
    #rect = patches.Rectangle((latbounds[0],lonbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',fill='False')
    #ax.add_patch(poly)
    plt.title(r'regression of THF on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + '{:s}_AMO_thf_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    #ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, thfcorrs_lt, cmap=plt.cm.RdBu_r, levels=thflevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'W m$^{-2}$')
    plt.title(r'regression of long-term THF on AMO ({:1.0f}-yr RM)'.format(N_map))
    plt.savefig(fout + '{:s}_AMO_thf_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
    ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
    #ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, thfcorrs_st, cmap=plt.cm.RdBu_r, levels=thflevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'W m$^{-2}$')
    plt.title(r'regression of short-term THF on AMO ({:1.0f}-yr RM residual)'.format(N_map))
    plt.savefig(fout + '{:s}_AMO_thf_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    sstcorrs_zonalave = np.ma.average(sstcorrs[:,NAminloni:NAmaxloni], axis=-1)
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    ax.plot(sstcorrs_zonalave, lats)
    ax.axvline(0, color='k')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_xlabel(r'$^{\circ}$C')
    ax.set_ylabel(r'latitude ($^{\circ}$)')
    ax.set_ylim(0, 60)
    ax.set_xlim(-6.5,2.5)
    #ax.set_ylim(50,1000)
    #ax.invert_yaxis()
    #cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    #cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    #cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_sst_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    thfcorrs_zonalave = np.ma.average(thfcorrs[:,NAminloni:NAmaxloni], axis=-1)
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    ax.plot(thfcorrs_zonalave, lats)
    ax.axvline(0, color='k')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_xlabel(r'W m$^{-2}$')
    ax.set_ylabel(r'latitude ($^{\circ}$)')
    ax.set_ylim(0, 60)
    #ax.set_ylim(50,1000)
    #ax.invert_yaxis()
    #cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    #cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    #cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of THF on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_thf_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    pscorrs_zonalave = np.ma.average(pscorrs[:,NAminloni:NAmaxloni], axis=-1)
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    ax.plot(pscorrs_zonalave, lats)
    ax.axvline(0, color='k')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_xlabel(r'hPa')
    ax.set_ylabel(r'latitude ($^{\circ}$)')
    ax.set_ylim(0, 60)
    #ax.set_ylim(50,1000)
    #ax.invert_yaxis()
    #cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    #cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    #cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'regression of SLP on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
    plt.savefig(fout + 'MERRA2_AMO_slp_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    
meansst = np.ma.average(sst, axis=0)
meanps = np.ma.average(ps, axis=0)
meanu = np.ma.average(u, axis=0)
meanv = np.ma.average(v, axis=0)

SLPlevels = np.arange(-1000,1030,4)
sstlevels = np.arange(270,310,1)


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
ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
ct = ax.contour(x, y, meanps, levels=SLPlevels, colors='k', linewidths=1)
ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
#plot = ax.contourf(x, y, meansst, cmap=plt.cm.magma, levels=sstlevels, extend='both', transform=prj)
#cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], meanu[::uskip,::uskip], meanv[::uskip,::uskip], transform=prj,  scale_units='inches', scale = 10, width=0.001, headwidth=16, headlength=10, minshaft=4)
ax.quiverkey(qv1, 0.95, 0.95, 5, '5 m/s')
plt.title(r'mean SLP and 10-m winds') 
plt.savefig(fout + '{:s}_MEAN_SLPuv_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()















































