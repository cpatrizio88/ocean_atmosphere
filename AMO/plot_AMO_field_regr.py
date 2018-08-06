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
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats.stats import pearsonr, linregress
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, calcsatspechum
from thermolib.thermo import w_sat, r_star, q_star, dqstar_dT, drstar_dT
from thermolib.constants import constants

c = constants()


fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
#fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')
#fustar = cdms2.open(fin + 'MERRA2_ustarqstar_monthly1980to2017.nc')
fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
fqvsurf = cdms2.open(fin + 'MERRA2_qvsurf_monthly1980to2017.nc')
ft10m = cdms2.open(fin + 'MERRA2_t10m_monthly1980to2017.nc')



#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201612.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#dataname = 'ERAi'
dataname = 'MERRA2'

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

rho=1.225
#L_v = 2.3e6

ps = fSLP('SLP')
ps = ps/1e2
ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
nt_ps = ps.shape[0]
ps = ps[tskip:,:]

#sst = fsst('skt')
sst = fsst('TSKINWTR')
lats = sst.getLatitude()[:]
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:nt_ps,:]

rho=1.225
L_v = 2.3e6

#cE = fcE('CDQ')
#cE = fcE('CDH')
#cE = cE.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#cE = cE[tskip:nt_ps,:]
#cD = fcD('CN')

#lhf = fsst('EFLUXWTR')
#shf = fsst('HFLUXWTR')
##thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)
#
##lhf = fthf('slhf')
##lhf = lhf/(12*3600)
##shf = fthf('sshf')
##sshf is accumulated 
##shf = shf/(12*3600)
#thf = lhf + shf
#thf = thf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#thf = thf[tskip:nt_ps,:]
#thf is positive down in ERAi, convert to positive up
#thf = thf

#u = fuv('U10M')
#u = fuv('u10')
#u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#u = u[tskip:nt_ps,:]

#v = fuv('V10M')
#v = fuv('v10')
#v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#v = v[tskip:nt_ps,:]

#umag = np.sqrt(np.square(v) + np.square(u))


#cE = fcE('CDQ')
#cE = cE.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#cE = cE[tskip:nt_ps,:]
#
#qv10m = fRH('QV10M')
#qv10m = qv10m.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#qv10m = qv10m[tskip:nt_ps,:]
#
#qvsurf = fqvsurf('QSH')
#qvsurf = qvsurf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#qvsurf = qvsurf[tskip:nt_ps,:]
#
#qvdiff = qvsurf - qv10m
#
#t10m = ft10m('T10M')
#t10m = t10m.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#t10m  = t10m[tskip:nt_ps,:]
#
#tdiff = sst - t10m
#
#Hdiff = c.cpd*tdiff + L_v*qvdiff
#
#THFparam = np.multiply(cE, Hdiff)
#
##
#meant10m = np.ma.average(t10m, axis=0)
#meanps = np.ma.average(ps, axis=0)
##
##qsat = q_star(ps*1e2, t10m)
##
#wsat = r_star(ps*1e2, t10m)
#
#RH10m=100*(qv10m/wsat)
##
#meanRH10m = np.ma.average(RH10m, axis=0)
#meansst = np.ma.average(sst, axis=0)
##
#dqstardt = dqstar_dT(meanps*1e2, meansst)

#dqstardt = np.repeat(dqstardt[np.newaxis,...],sst.shape[0],axis=0)
#meanRH10m = np.repeat(RH10m[np.newaxis,...],sst.shape[0],axis=0)



#LW_net_surf = radfile['LWGNT']
##LW_net_surf_cs = radfile('LWGNTCLR')
SW_net_surf = radfile['SWGNT']
##SW_net_surf_cs = radfile('SWGNTCLR')
##LW_net_TOA = radfile['LWTUP']
##lwnettoa_cs = radfile('LWTUPCLR')
##swnettoa = radfile('SWTNT')
##swnettoa_cs = radfile('SWTNTCLR')
#
##Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs
#
#Q_net_surf = LW_net_surf + SW_net_surf

#CRE_surf = Q_net_surf - Q_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH AMO
#field = umag
#ftitle = r'$|\mathbf{u}_{10m}|$'
#fsave = 'umag'
#units = 'm/s'

#field=qv10m
#ftitle=r'RH$_{10m}$'
#fsave = 'RH10m'
#units = '%'

#field=qv10m
#fsave = 'cE'
#ftitle=r'$c_E$'
#units = ''

#field = cE
#fsave = 'cEH'
#ftitle = r'$C_{E,heat}$'
#units = 'kg m$^{-2}$ s$^{-1}$'

#field = cE*1e3
#fsave = 'cE'
#ftitle = r'$C_{E}$'
#units = 'g m$^{-2}$ s$^{-1}$'

#field= sst 
#fsave = 'qvdiffCC'
#ftitle=r'$\Delta q_{v, sea-air, CC}$'
#units = 'g/kg'

#field = RH10m
#fsave = 'RH10m'
#ftitle = r'RH$_{10m}$'
#units = '%'

#field = t10m
#fsave = 't10m'
#ftitle = r'$T_{10m}$'
#units = 'K'

#field = t10m
#fsave = 'qvdiff'
#ftitle = r'CC $\Delta q_{sea-air}$'
#units = 'g/kg'

#field = tdiff
#fsave = 'tdiff'
#ftitle = r'$\Delta T_{sea-air}$'
#units = 'K'

#field = Q_net_surf
#ftitle = r'$Q_{net}$'
#fsave = 'Qnetsurf'
#units = r'W m$^{-2}$'

#field = LW_net_surf
#ftitle = r'$LW_{net}$'
#fsave = 'LWnetsurf'
#units = r'W m$^{-2}$'

field = SW_net_surf
ftitle = r'$SW_{net}$'
fsave = 'SWnetsurf'
units = r'W m$^{-2}$'


#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

#field = thf
#ftitle = r'THF'
#fsave = 'thf'
#units = r'W m$^{-2}$'

#field = THFparam
#ftitle = r'$(c_E \Delta H_{sea-air})^{\prime}$'
#fsave = 'THFparam'
#units = r'W m$^{-2}$'

#field = Hdiff*1e-3
#ftitle = r'$\Delta H_{sea-air}$'
#fsave = 'Hdiff'
#units = 'kJ/kg'

#field = L_v*qvdiff*1e-3
#ftitle = r'$L_{v}\Delta q_{v,sea-air}$'
#fsave = 'Lvqvdiff'
#units = 'kJ/kg'

#field = c.cpd*tdiff*1e-3
#ftitle = r'$c_{p}\Delta T_{,sea-air}$'
#fsave = 'cptdiff'
#units = 'kJ/kg'

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

#field = sst
#ftitle = r'SST'
#fsave = 'sst'
#units = 'K'

#field = ps
#ftitle = 'SLP'
#fsave = 'SLP'
#units = 'hPa'



#NAmaxlati = np.where(lats > maxlat)[0][0]
#NAminlati = np.where(lats > minlat)[0][0]

#sst = sst.subRegion(longitude=(minlon, maxlon))
#sst = sst[tskip:,NAminlati:NAmaxlati,:]
#field = field[tskip:,NAminlati:NAmaxlati,:]

field = field.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
field = field[tskip:nt_ps,:]

#field = qsat - field

#field = sst - field

#CHANGE FOR DETRENDING/CORRELATION VS. REGRESSION
detr=True
corr=False

if detr: 
 sst = detrend(sst)
 #ps_an, params = detrend_separate(ps_an)
 field = detrend(field)
 
#field=field*1e3

sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
#field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)


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

#initial/final years for base period 
baseti = 0
basetf = 10

sst_an = an_ave(sst)
#CRE_surf_an = an_ave(thf)
#ps_an = an_ave(ps)
field_an = an_ave(field)

if fsave == 'qvdiffCC':
    
    dqstardt = np.repeat(dqstardt[np.newaxis,...],sst_an.shape[0],axis=0)
    meanRH10m = np.repeat(meanRH10m[np.newaxis,...],sst_an.shape[0],axis=0)
    field_an = field_an - np.multiply(meanRH10m/100., an_ave(t10m))
    field_an = 1e3*np.multiply(dqstardt, field_an)


#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
#if detr: 
# sst_an, params = detrend(sst_an)
# #ps_an, params = detrend_separate(ps_an)
# field_an, params = detrend(field_an)
# 

 
#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5
ci = (N_map-1)/2
ltlag = 5
stlag = 1

lagmax=11
lags = np.arange(-lagmax,lagmax+1)

sst_globe_an = spatial_ave(sst_an, lats)
field_globe_an = spatial_ave(field_an, lats)

#subtract global annual mean to isolate processes in NA
sstprime = sst_an.T - sst_globe_an
sstprime = sstprime.T

fieldprime = field_an.T - field_globe_an
fieldprime = fieldprime.T

field_lt = running_mean(fieldprime, N_map)
field_st = fieldprime[ci:-ci,:] - field_lt

nt = sst.shape[0]
nt_lt = field_lt.shape[0]

fieldlagcorrs_merge = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_merge_lt = np.zeros((len(lags), nlat, nlon))
fieldlagcorrs_merge_st = np.zeros((len(lags), nlat, nlon))

latboundar = np.array([[0,20],[20,45],[45,60]])
#latboundar = np.array([[0,20]])
#latboundar = np.array([[0,60]])

#EDIT BOUNDS FOR AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
for latbounds in latboundar:
    
    fieldlagcorrs = np.zeros((len(lags), nlat, nlon))
    fieldlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
    fieldlagcorrs_st = np.zeros((len(lags), nlat, nlon))
    
    sstcorrs = MV.zeros((nlat,nlon))
    sstpvals = MV.zeros((nlat,nlon))
    fieldcorrs = MV.zeros((nlat, nlon))
    fieldcorrs_lt = MV.zeros((nlat,nlon))
    fieldcorrs_st = MV.zeros((nlat,nlon))

    
    AMO, sstanom_globe_an, sstanom_na_an = calc_NA_globeanom(sst_an, latbounds, lats, lons, baseti, basetf)
    NAfield2, fieldanom_globe_an, fieldanom_na_an = calc_NA_globeanom(field_an, latbounds, lats, lons, baseti, basetf)
    
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
    
    
    #TODO: can eliminate loop over latitudes with these lines
    #sstprime_g = sstprime.reshape(nt, nlat*nlon)
    #clf = linear_model.LinearRegression()
    #clf.fit(AMOstd.reshape(-1,1), sstprime_g)
    #sstcorrs = clf.coef_.reshape(nlat, nlon)
    
    #fieldprime_g = fieldprime.reshape(nt, nlat*nlon)
    #clf = linear_model.LinearRegression()
    #clf.fit(AMOstd.reshape(-1,1), fieldprime_g)
    #fieldcorrs = clf.coef_.reshape(nlat, nlon)
    
    #field_lt_g = field_lt.reshape(nt_lt, nlat*nlon)
    #clf = linear_model.LinearRegression()
    #clf.fit(AMOstd.reshape(-1,1), field_lt_g)
    #fieldcorrs_lt = clf.coef_.reshape(nlat, nlon)
    
    #field_st_g = field_lt.reshape(nt_lt, nlat*nlon)
    #clf = linear_model.LinearRegression()
    #clf.fit(AMOstd.reshape(-1,1), field_st_g)
    #fieldcorrs_st = clf.coef_.reshape(nlat, nlon)
    
    #compute correlation between long-term/short-term AMO and 2D field
    print r'calculating correlations between AMO and {:s}...'.format(ftitle)
    for i in range(nlat):         
    
     print 'latitude', lats[i]
       
     sstprime_g = sstprime[:,i,:]
     fieldprime_g = fieldprime[:,i,:]
     field_lt_g = field_lt[:,i,:]
     field_st_g = field_st[:,i,:]
    
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd.reshape(-1,1), sstprime_g)
     sstcorrs[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd.reshape(-1,1), fieldprime_g)
     fieldcorrs[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd_lt.reshape(-1,1), field_lt_g)
     fieldcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd_st.reshape(-1,1), field_st_g)
     fieldcorrs_st[i,:] = np.squeeze(clf.coef_)
     
     for lag in lags:
         
         scaler = StandardScaler()
         if corr:
             fieldstd = scaler.fit_transform(fieldprime_g)
             fieldstd_lt = scaler.fit_transform(field_lt_g)
             fieldstd_st = scaler.fit_transform(field_st_g)
         else:
             fieldstd = fieldprime_g
             fieldstd_lt = field_lt_g
             fieldstd_st = field_st_g 
         
         fieldclf = linear_model.LinearRegression()
         fieldclf_lt = linear_model.LinearRegression()
         fieldclf_st = linear_model.LinearRegression()
    
         #THF LAGS SST
         if lag > 0:
            fieldclf.fit(AMOstd[:-lag], fieldstd[lag:,:])
            fieldclf_lt.fit(AMOstd_lt[:-lag], fieldstd_lt[lag:,:])
            fieldclf_st.fit(AMOstd_st[:-lag], fieldstd_st[lag:,:])
    
        #THF LEADS SST
         elif lag < 0: 
            fieldclf.fit(AMOstd[-lag:], fieldstd[:lag,:])
            fieldclf_lt.fit(AMOstd_lt[-lag:], fieldstd_lt[:lag,:])
            fieldclf_st.fit(AMOstd_st[-lag:], fieldstd_st[:lag,:])
    
         else:
            fieldclf.fit(AMOstd, fieldstd)
            fieldclf_lt.fit(AMOstd_lt, fieldstd_lt)
            fieldclf_st.fit(AMOstd_st, fieldstd_st)
    
            
            
         fieldlagcorrs[lag+lagmax,i,:] = np.squeeze(fieldclf.coef_)
         fieldlagcorrs_lt[lag+lagmax,i,:] = np.squeeze(fieldclf_lt.coef_)
         fieldlagcorrs_st[lag+lagmax,i,:] = np.squeeze(fieldclf_st.coef_)
         
     #fieldcorrs = fieldcorrs.reshape(nlat, nlon)
     #fieldcorrs_lt = fieldcorrs_lt.reshape(nlat, nlon)
     #fieldcorrs_st = fieldcorrs_st.reshape(nlat, nlon)
         
     #fieldlagcorrs = fieldlagcorrs.reshape(len(lags), nlat, nlon)
     #fieldlagcorrs_lt = fieldlagcorrs_lt.reshape(len(lags), nlat, nlon)
     #fieldlagcorrs_st = fieldlagcorrs_st.reshape(len(lags), nlat, nlon)
    
    
         
    lonbounds = [280,359.99]
    
    NAminlati = np.where(lats > latbounds[0])[0][0]
    NAmaxlati = np.where(lats > latbounds[1])[0][0]
    NAminloni = np.where(lons > lonbounds[0])[0][0]
    NAmaxloni = np.where(lons > lonbounds[1])[0][0]
    
    fieldlagcorrs_merge[:,NAminlati:NAmaxlati,:] = fieldlagcorrs[:,NAminlati:NAmaxlati,:]
    fieldlagcorrs_merge_lt[:,NAminlati:NAmaxlati,:] = fieldlagcorrs_lt[:,NAminlati:NAmaxlati,:]
    fieldlagcorrs_merge_st[:,NAminlati:NAmaxlati,:] = fieldlagcorrs_st[:,NAminlati:NAmaxlati,:]
    
    NAlats = lats[NAminlati:NAmaxlati]
    NAfield = spatial_ave(fieldprime[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], NAlats)
    
    
    windows = np.arange(3,lagmax+1,2)
    ll, ww = np.meshgrid(lags, windows)
    
    #NAfieldlagcorrs = np.zeros((len(windows), len(lags)))
    NAfieldlagcorrs_lt = np.zeros((len(windows), len(lags)))
    NAfieldlagcorrs_st = np.zeros((len(windows), len(lags)))
    
    
    print 'calculating lagged correlation between AMO and {:s} for different RM windows...'.format(ftitle)
    #commpute lagged correlation between smoothed AMO and NA THF for different RM windows
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
                
                
                NAfield_lt = running_mean(NAfield, N)
                NAfield_st = NAfield[ci:-ci] - NAfield_lt 
                NAfieldstd = (NAfield - np.mean(NAfield)/np.std(NAfield))
                NAfieldstd_lt = (NAfield_lt - np.mean(NAfield_lt)) / (np.std(NAfield_lt))
                NAfieldstd_st = (NAfield_st - np.mean(NAfield_st)) / (np.std(NAfield_st))
                NAfieldlaggedcorr_temp = np.correlate(NAfieldstd, AMOstd, 'full')
                NAfieldlaggedcorr_lt_temp = np.correlate(NAfieldstd_lt, AMOstd_lt, 'full')
                NAfieldlaggedcorr_st_temp = np.correlate(NAfieldstd_st, AMOstd_st, 'full')
     
                
                
                lagzero = len(NAfieldlaggedcorr_lt_temp)/2
                #NAfieldlagcorrs[k,:] = NAfieldlaggedcorr_temp[lagzero-lagmax:lagzero+lagmax+1]
                NAfieldlagcorrs_lt[k,:] = NAfieldlaggedcorr_lt_temp[lagzero-lagmax:lagzero+lagmax+1]
                NAfieldlagcorrs_st[k,:] = NAfieldlaggedcorr_st_temp[lagzero-lagmax:lagzero+lagmax+1]
    
                    
            else:
                
                for j, lag in enumerate(lags):
            
        
                     scaler = StandardScaler()
                    
                     ci = (N-1)/2 
                     
                     AMO_lt = running_mean(AMO, N)
                     AMO_st = AMO[ci:-ci] - AMO_lt
                     AMOstd = scaler.fit_transform(AMO.reshape(-1,1))
                     AMOstd_lt = scaler.fit_transform(AMO_lt.reshape(-1,1))
                     AMOstd_st = scaler.fit_transform(AMO_st.reshape(-1,1))
                    
                     NAfield_lt_temp = running_mean(NAfield, N)
                     NAfield_st_temp = NAfield[ci:-ci] - NAfield_lt_temp
                     
                     fieldstd_lt = NAfield_lt_temp.reshape(-1,1)
                     fieldstd_st = NAfield_st_temp.reshape(-1,1)
                    
                     fieldclf = linear_model.LinearRegression()
                     fieldclf_lt = linear_model.LinearRegression()
                     fieldclf_st = linear_model.LinearRegression()
                     
                    
                    #field LAGS SST
                     if lag > 0:
                        #fieldclf.fit(AMOstd[:-lag], fieldstd[lag:,:])
                        fieldclf_lt.fit(AMOstd_lt[:-lag], fieldstd_lt[lag:,:])
                        fieldclf_st.fit(AMOstd_st[:-lag], fieldstd_st[lag:,:])
                    
                    #field LEADS SST
                     elif lag < 0: 
                        #fieldclf.fit(AMOstd[-lag:], fieldstd[:lag,:])
                        fieldclf_lt.fit(AMOstd_lt[-lag:], fieldstd_lt[:lag,:])
                        fieldclf_st.fit(AMOstd_st[-lag:], fieldstd_st[:lag,:])
                    
                     else:
                        #fieldclf.fit(AMOstd, fieldstd)
                        fieldclf_lt.fit(AMOstd_lt, fieldstd_lt)
                        fieldclf_st.fit(AMOstd_st, fieldstd_st)
                       
                    
                     #NAfieldlagcorrs[k,lag+lagmax] = np.squeeze(fieldclf.coef_)
                     NAfieldlagcorrs_lt[k,lag+lagmax] = np.squeeze(fieldclf_lt.coef_)
                     NAfieldlagcorrs_st[k,lag+lagmax] = np.squeeze(fieldclf_st.coef_)
   
    
    ##Plot AMO
    fig=plt.figure(figsize=(16,14))
    fig.tight_layout()
    ax = fig.add_subplot(311)
    plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
    ax.plot(tyears, sstanom_globe_an)
    if detr:
        ax.set_ylim(-0.6,0.6)
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
        ax.set_ylim(-0.6,0.6)
    else:
        ax.set_ylim(-1,1)
    ax.set_ylabel(r'SST ($^{{\circ}}$C)')
    ax.axhline(0, color='black')
    ax.set_title(r'NA mean SST')
    #plt.savefig(fout + 'MERRA2_global_NA_SST_anomaly_timeseries.pdf')
    #plt.close()
    
    AMO_smooth = running_mean(AMO, N_map)
    ci = (N_map-1)/2
    AMO_st = AMO[ci:-ci] - AMO_smooth
    ax = fig.add_subplot(313)
    #plt.figure()
    #ax=plt.gcf().gca()
    ax.plot(tyears, AMO, label='AMO')
    ax.plot(tyears[ci:-ci],AMO_smooth,label='{:1.0f}-yr RM'.format(N_map))
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
    
    #Plot NA field
    fig=plt.figure(figsize=(16,14))
    plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
    ax = fig.add_subplot(311)
    ax.plot(tyears, fieldanom_globe_an)
    ax.set_ylabel(r'{:s} ({:s})'.format(ftitle, units))
    #ax.set_ylim(ymin,ymax)
    ax.set_title(r'global mean {:s} (base period: {:3.0f} to {:3.0f})'.format(ftitle, tyears[baseti], tyears[basetf]))
    ax.axhline(0, color='black')
    ax = fig.add_subplot(312)
    ax.plot(tyears, fieldanom_na_an)
    ax.set_ylabel(r'{:s}  ({:s})'.format(ftitle, units))
    #ax.set_ylim(ymin,ymax)
    ax.axhline(0, color='black')
    ax.set_title(r'NA mean {:s}'.format(ftitle))
    #plt.savefig(fout + 'MERRA2_global_NA_thf_anomaly_timeseries.pdf')
    #plt.close()
    
    NAfield_smooth = running_mean(NAfield2, N_map)
    ax = fig.add_subplot(313)
    ax.plot(tyears, NAfield2, label = 'NA {:s}'.format(ftitle))
    ax.plot(tyears[ci:-ci], NAfield_smooth, label = '{:1.0f}-yr RM'.format(N_map))
    ax.set_ylabel(r'{:s}  ({:s})'.format(ftitle, units))
    plt.title(r'NA mean {:s} - global mean {:s}'.format(ftitle, ftitle))
    #ax.set_ylim(ymin, ymax)
    plt.axhline(0, color='black')
    ax.set_xlabel('time (years)')
    ax.legend()
    plt.savefig(fout + '{:s}_{:s}_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    lagoffset = np.diff(lags)[0]/2.
    woffset = np.diff(windows)[0]/2.
    latoffset = np.diff(lats)[0]/2.
    
    ll, ww = np.meshgrid(lags-lagoffset, windows-woffset)
    
    laglabels = np.round(np.arange(-10,15,5))
    
    if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
        fieldmin=-5
        fieldmax=5
        fieldstep = 0.05
        cbstep = 1.0
    elif fsave == 'sst':
        fieldmin=-0.5
        fieldmax=0.5
        fieldstep = 0.02
        cbstep = 0.25
    elif fsave == 'Qnetsurf':
        fieldmin=-4
        fieldmax=4
        fieldstep = 0.1
        cbstep=1.0
    elif fsave == 'thf':
        fieldmin=-4
        fieldmax=4
        fieldstep=0.1
        cbstep=1.0
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
    elif fsave == 'cE' or fsave == 'cEH':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep = 0.1
    elif fsave == 'cD':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep = 0.1
    elif fsave == 'Hdiff' or fsave == 'Lvqvdiff' or fsave == 'cptdiff':
        fieldmin = -0.8
        fieldmax = 0.8
        fieldstep = 0.02
        cbstep = 0.2
    else:
        fieldmin=-4
        fieldmax=4
        fieldstep =0.1
        cbstep=1.0
        
    if corr:
        fieldminlag = -1 
        fieldmaxlag = 1
        cbsteplag = 0.2
        fieldunitslag = ''
    
    else:
        fieldminlag = fieldmin
        fieldmaxlag = fieldmax
        cbsteplag = cbstep 
        fieldunitslag = units
    
    ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
    ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
    ticklbls[ticklbls == -0.00] = 0.00
      

    
    #Plot correlation between field and AMO at different lags and different smoothing window lengths
    fig=plt.figure(figsize=(18,14))
    #ax = fig.add_subplot(221)
#    ax.pcolor(ll, ww, NAfieldlagcorrs, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#    if corr:
#        ax.set_title('correlation of AMO with NA {:s}'.format(ftitle))
#    else:
#        ax.set_title('regression of NA {:s} on AMO'.format(ftitle))
#    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
#    ax.set_ylabel('RM window (years)')
#    ax.axvline(0, color='k')
#    #ax.set_yticks(windows)
#    #ax.set_yticklabels(windows)
#    ax.set_xticks(laglabels)
#    ax.set_xticklabels(laglabels)
#    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(221)
    ax.pcolor(ll, ww, NAfieldlagcorrs_lt, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
    if corr:
        ax.set_title('long-term correlation of AMO with NA {:s}'.format(ftitle))
    else:
        ax.set_title('long-term regression of NA {:s} on AMO'.format(ftitle))
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    ax.set_ylabel('RM window (years)')
    ax.axvline(0, color='k')
    #ax.set_yticks(windows)
    #ax.set_yticklabels(windows)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
    ax = fig.add_subplot(222)
    h=ax.pcolor(ll, ww, NAfieldlagcorrs_st, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
    ax.axvline(0, color='k')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    #ax.set_yticks(windows)
    #ax.set_yticklabels(windows)
    ax.set_xticks(laglabels)
    ax.set_xticklabels(laglabels)
    #ax.set_ylabel('smoothing (years)') 
    cb=fig.colorbar(h, ax=ax, orientation="vertical", format='%1.2f', label=r'{:s}'.format(fieldunitslag))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    #plt.savefig(fout + 'MERRA2_AMO_field_lagcorr_hist_{:2.0f}Nto{:2.0f}N.pdf'.format(latbounds[0], latbounds[1]))
    #plt.close()
    
    i = np.where(windows>N_map)[0][0]-1
    
    #fig=plt.figure(figsize=(18,6))
    
#    ax = fig.add_subplot(324)
#    ax.plot(lags, NAfieldlagcorrs[i,:])
#    ax.axhline(0, color='black')
#    ax.axvline(0, color='black')
#    ax.set_ylim(fieldminlag,fieldmaxlag)
#    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
#    if corr:
#        ax.set_title('correlation of AMO with NA {:s} ({:1.0f}-yr RM)'.format(ftitle, windows[i]))
#    else:
#        ax.set_title('regression of NA {:s} on AMO ({:1.0f}-yr RM)'.format(ftitle, windows[i]))
#    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    ax = fig.add_subplot(223)
    ax.plot(lags, NAfieldlagcorrs_lt[i,:])
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    if corr:
        ax.set_title('long-term correlation of {:s} with AMO ({:1.0f}-yr RM)'.format(ftitle, windows[i]))
    else:
        ax.set_title('long-term regression of {:s} on AMO ({:1.0f}-yr RM)'.format(ftitle, windows[i]))
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    ax = fig.add_subplot(224)
    ax.plot(lags, NAfieldlagcorrs_st[i,:])
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    if corr:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagcorr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagregr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    #plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_timeseries_{:1.0f}year_{:2.0f}Nto{:2.0f}N.pdf'.format(windows[i],latbounds[0], latbounds[1]))
    plt.close()
    
    #Plot maps of SST and THF patterns associated with AMO
    #CHANGE THIS FOR MAP PROJECTION
    prj = cart.crs.PlateCarree()
    bnds = [-90, 0, 0, 60]
    
        #latitude/longitude labels
    par = np.arange(-90.,91.,15.)
    mer = np.arange(-180.,180.,15.)
    
    lstep = 0.01
    levels = np.arange(-1.0, 1.0+lstep, lstep)
    x, y = np.meshgrid(lons, lats)
    pstep = 0.2
    sststep = 0.02
    
    if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
        fieldmin=-5
        fieldmax=5
        fieldstep = 0.05
        cbstep = 1.0
    elif fsave == 'sst' or fsave == 't10m':
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
    elif fsave == 'qvdiff' or fsave == 'qvdiffCC':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep =0.1
    elif fsave == 'cE' or fsave == 'cEH':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep = 0.1
    elif fsave == 'cD':
        fieldmin = -0.5
        fieldmax = 0.5
        fieldstep = 0.01
        cbstep = 0.1
    elif fsave == 'tdiff':
        fieldmin=-0.5
        fieldmax=0.5
        fieldstep=0.01
        cbstep=0.1
    elif fsave == 'Hdiff' or fsave == 'Lvqvdiff' or fsave == 'cptdiff':
        fieldmin = -0.8
        fieldmax = 0.8
        fieldstep = 0.02
        cbstep = 0.2
    else:
        fieldmin=-10
        fieldmax=10
        fieldstep =0.2
        cbstep=2.5
        
    #dqstardt = dqstar_dT(meanps*1e2, meansst)
    #CCfieldcorrs = 1e3*np.multiply(dqstardt, fieldcorrs)
    
            
    ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),2)
    ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 2)
    ticklbls[ticklbls == -0.0] = 0.0
                     
    sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
    fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
        
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
    #ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
    plot = ax.contourf(x, y, fieldcorrs, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    plt.title(r'regression of {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$'.format(ftitle, latbounds[0], latbounds[1]))
    plt.savefig(fout + '{:s}_AMO_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    #plt.savefig(fout + '{:s}_AMO_CCQV_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
    plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    plt.title(r'regression of long-term {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle,  latbounds[0], latbounds[1], N_map))
    plt.savefig(fout + '{:s}_AMO_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
    plot = ax.contourf(x, y, fieldcorrs_st, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=prj)
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklbls)
    plt.title(r'regression of short-term {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM residual)'.format(ftitle, latbounds[0], latbounds[1], N_map))
    plt.savefig(fout + '{:s}_AMO_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    
fieldcorrs_zonalave = np.ma.average(fieldcorrs[:,NAminloni:NAmaxloni], axis=-1)
fig = plt.figure(figsize=(12,8))
ax = fig.gca()
ax.plot(fieldcorrs_zonalave, lats)
ax.axvline(0, color='k')
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.set_xlabel(r'{:s}'.format(units))
ax.set_ylabel(r'latitude ($^{\circ}$)')
ax.set_ylim(0, 60)
if fsave == 'thf' or fsave == 'Qnetsurf':
    ax.set_xlim(-2.5,6.5)
if fsave == 'RH10m':
    ax.set_xlim(-0.6,0.6)
#ax.set_ylim(50,1000)
#ax.invert_yaxis()
#cb = plt.colorbar(plot, label=r'{:s}'.format(units))
#cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
#cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
plt.title(r'regression of {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$'.format(ftitle, latbounds[0], latbounds[1]))
plt.savefig(fout + '{:s}_AMO_{:s}_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
    

lonbounds = [280,359.99]
latbounds = [0,60]
    
NAminlati = np.where(lats > latbounds[0])[0][0]
NAmaxlati = np.where(lats > latbounds[1])[0][0]
NAminloni = np.where(lons > lonbounds[0])[0][0]
NAmaxloni = np.where(lons > lonbounds[1])[0][0]

NAlats = lats[NAminlati:NAmaxlati]

    
if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst' or fsave == 't10m':
    fieldmin=-0.3
    fieldmax=0.3
    fieldstep = 0.02
    cbstep = 0.1
elif fsave == 'umag':
    fieldmin=-0.3
    fieldmax=0.3
    fieldstep=0.01
    cbstep = 0.1
elif fsave == 'RH10m':
    fieldmin = -2
    fieldmax = 2
    fieldstep = 0.01
    cbstep = 0.5
elif fsave == 'cE' or fsave == 'cEH':
    fieldmin = -0.5
    fieldmax = 0.5
    fieldstep = 0.01
    cbstep = 0.1
elif fsave == 'cD':
    fieldmin = -0.5
    fieldmax = 0.5
    fieldstep = 0.01
    cbstep = 0.1
elif fsave == 'Hdiff' or fsave == 'Lvqvdiff' or fsave == 'cptdiff':
    fieldmin = -0.8
    fieldmax = 0.8
    fieldstep = 0.02
    cbstep = 0.2
else:
    fieldmin=-3
    fieldmax=3
    fieldstep =0.1
    cbstep=1.0
    
if corr:
    fieldminlag = -1.0
    fieldmaxlag = 1.0
    cbsteplag = 0.2
    fieldunitslag = ''

else:
    fieldminlag = fieldmin
    fieldmaxlag = fieldmax
    cbsteplag = cbstep 
    fieldunitslag = units
    
ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
ticklbls[ticklbls == -0.00] = 0.00
                 
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

weights = np.cos(np.deg2rad(lats))
#CRE_surflagcorrs = np.ma.array(CRE_surflagcorrs, mask=~np.isfinite(CRE_surflagcorrs))
#CRE_surflagcorrs_lt = np.ma.array(CRE_surflagcorrs_lt, mask=~np.isfinite(CRE_surflagcorrs_lt))
#CRE_surflagcorrs_st = np.ma.array(CRE_surflagcorrs_st, mask=~np.isfinite(CRE_surflagcorrs_st))
fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs_merge[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_merge_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_merge_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)

#pslagcorrs_zonalave = np.ma.average(pslagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#pslagcorrs_lt_zonalave = np.ma.average(pslagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
#pslagcorrs_st_zonalave = np.ma.average(pslagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)#SHOULDN'T THIS BE EQUIVALENT TO THE CORRELATION BETWEEN SMOOTHED AMO AND NA THF? i.e. NACRE_surf_laggedcorr_lt[i,:]
#thflagcorrs_test = np.ma.average(thflagcorrs_lt_zonalave, axis=1, weights=weights[NAminlati:NAmaxlati])

lagg, latt = np.meshgrid(lags-lagoffset, NAlats-latoffset)

#Plot zonally-averaged lagged correlation between long-term AMO and THF
fig=plt.figure(figsize=(22,6))
#plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
ax = fig.add_subplot(131)
h = ax.pcolor(lagg, latt, fieldlagcorrs_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
if corr:
    ax.set_title('correlation of {:s} with AMO'.format(ftitle))
else:
    ax.set_title('regression of {:s} on AMO'.format(ftitle))
ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
ax.set_ylabel('latitude (degrees)')
ax.set_ylim(0,60)
ax.set_xticks(laglabels)
ax.set_xticklabels(laglabels)
if len(latboundar) > 1:
    ax.axhline(20, color='grey', linewidth=1)
    ax.axhline(45, color='grey', linewidth=1)
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
ax = fig.add_subplot(132)
ax.pcolor(lagg, latt, fieldlagcorrs_lt_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
if corr:
    ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
else:
    ax.set_title('long-term regression ({:1.0f}-yr RM)'.format(N_map))
ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
ax.set_ylim(0,60)
ax.set_xticks(laglabels)
ax.set_xticklabels(laglabels)
if len(latboundar) > 1:
    ax.axhline(latboundar[0][1], color='grey', linewidth=1)
    ax.axhline(latboundar[1][1], color='grey', linewidth=1)
ax = fig.add_subplot(133)
h = ax.pcolor(lagg, latt, fieldlagcorrs_st_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
if corr:
    ax.set_title('short-term correlation')
else:
    ax.set_title('short-term regression')
ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
ax.set_ylim(0,60)
ax.set_xticks(laglabels)
ax.set_xticklabels(laglabels)
if len(latboundar) > 1:
    ax.axhline(latboundar[0][1], color='grey', linewidth=1)
    ax.axhline(latboundar[1][1], color='grey', linewidth=1)
cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
cb.set_ticks(ticks)
cb.set_ticklabels(ticklbls)
if corr:
    if len(latboundar) > 1:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagcorr_zonalavelocal_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
else:
    if len(latboundar) > 1:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagregr_zonalavelocal_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagregr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close() 

#calculate average of lagged correlation within defined latbounds
lats = lats[NAminlati:NAmaxlati]
weights = np.cos(np.deg2rad(lats))
tli = np.where(lats > latboundar[0][0])[0][0]
tui = np.where(lats > latboundar[0][1]-0.001)[0][0]

tfieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
tfieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])

fig=plt.figure(figsize=(18,7))
ax = fig.add_subplot(121)
ax.plot(lags, tfieldlagcorrs_lt_ave)
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_ylim(fieldminlag,fieldmaxlag)
ax.set_ylabel(r'{:s}'.format(fieldunitslag))
if corr:
    ax.set_title('long-term correlation of {:s} with AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle, latboundar[0][0], latboundar[0][1], windows[i]))
else:
    ax.set_title('long-term regression of {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle, latboundar[0][0], latboundar[0][1], windows[i]))
ax.set_xlabel('{:s} lag (years)'.format(ftitle))
ax = fig.add_subplot(122)
ax.plot(lags, tfieldlagcorrs_st_ave)
ax.set_ylim(fieldminlag,fieldmaxlag)
ax.set_ylabel(r'{:s}'.format(fieldunitslag))
ax.axhline(0, color='black')
ax.axvline(0, color='black')
if corr:
    ax.set_title('short-term correlation')
else:
    ax.set_title('short-term regression')
ax.set_xlabel('{:s} lag (years)'.format(ftitle))
if corr:
    plt.savefig(fout + '{:s}_AMO_{:s}_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
else:
    plt.savefig(fout + '{:s}_AMO_{:s}_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
plt.close() 

if len(latboundar) > 1:
    mui = np.where(lats > latboundar[1][1]-0.001)[0][0]
    eui = np.where(lats > latboundar[2][1]-0.001)[0][0]
    mfieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,tui:mui],axis=1,weights=weights[tui:mui])
    mfieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,tui:mui],axis=1,weights=weights[tui:mui])

    efieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,mui:eui],axis=1,weights=weights[mui:eui])
    efieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,mui:eui],axis=1,weights=weights[mui:eui])

    fig=plt.figure(figsize=(18,7))
    ax = fig.add_subplot(121)
    ax.plot(lags, mfieldlagcorrs_lt_ave)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    if corr:
        ax.set_title('long-term correlation of {:s} with AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle, latboundar[0][1], latboundar[1][1], windows[i]))
    else:
        ax.set_title('long-term regression of {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle, latboundar[0][1], latboundar[1][1], windows[i]))
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    ax = fig.add_subplot(122)
    ax.plot(lags, mfieldlagcorrs_st_ave)
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    if corr:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][1], latboundar[1][1], str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][1], latboundar[1][1],  str(detr)[0]))
    plt.close() 
    
    
    fig=plt.figure(figsize=(18,7))
    ax = fig.add_subplot(121)
    ax.plot(lags, efieldlagcorrs_lt_ave)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    if corr:
        ax.set_title('long-term correlation of {:s} with AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle, latboundar[1][1], latboundar[2][1], windows[i]))
    else: 
        ax.set_title('long-term regression of {:s} on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(ftitle, latboundar[1][1], latboundar[2][1], windows[i]))
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    ax = fig.add_subplot(122)
    ax.plot(lags, efieldlagcorrs_st_ave)
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('{:s} lag (years)'.format(ftitle))
    if corr:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[1][1], latboundar[2][1],  str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_{:s}_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[1][1], latboundar[2][1], str(detr)[0]))
    plt.close() 














































