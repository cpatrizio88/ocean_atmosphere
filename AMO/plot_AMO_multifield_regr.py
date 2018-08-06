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
from ocean_atmosphere.misc_fns import an_ave, spatial_ave, calc_AMO, running_mean, calc_NA_globeanom, detrend_separate, detrend_common
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
fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')
#fustar = cdms2.open(fin + 'MERRA2_ustarqstar_monthly1980to2017.nc')
fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
fqvsurf = cdms2.open(fin + 'MERRA2_qvsurf_monthly1980to2017.nc')
ft10m = cdms2.open(fin + 'MERRA2_t10m_monthly1980to2017.nc')


#ERA-interim
#fsst = cdms2.open(fin + 'sstslp.197901-201612.nc')
#fthf = cdms2.open(fin + 'thf.197901-201712.nc')

#dataname = 'ERAi's
dataname = 'MERRA2'

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'figure.figsize': (10,8)})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'legend.fontsize': 16})
matplotlib.rcParams.update({'mathtext.fontset': 'cm'})
matplotlib.rcParams.update({'ytick.major.size': 3})
#matplotlib.rcParams.update({'figure.autolayout': True})

rho = 1.225
L_v = 2.3e6

maxlat = 70
minlat = -70

maxlon = 360
minlon = 0

tskip = 6

ps = fSLP('SLP')
ps = ps/1e2
ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
nt_ps = ps.shape[0]
ps = ps[tskip:,:]

#sst = fsst('skt')
sst = fsst('TSKINWTR')
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:nt_ps,:]

lhf = fsst('EFLUXWTR')
#lhf = lhf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#lhf = lhf[tskip:nt_ps,:]
shf = fsst('HFLUXWTR')
#shf = shf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#shf = shf[tskip:nt_ps,:]
#thf = thf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#thf = thf[tskip:nt_ps,:]
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

#lhf = fthf('slhf')
#lhf = lhf/(12*3600)
#shf = fthf('sshf')
#sshf is accumulated 
#shf = shf/(12*3600)
thf = lhf + shf
#thf = thf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#thf = thf[tskip:nt_ps,:]
#thf is positive down in ERAi, convert to positive up
#thf = -thf



#cE = fcE('CDQ')
#cE = cE.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#cE = cE[tskip:nt_ps,:]
#
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
#meant10m = np.ma.average(t10m, axis=0)
#meanps = np.ma.average(ps, axis=0)
#meansst = np.ma.average(sst, axis=0)
#
#qsat = q_star(ps*1e2, t10m)
#
#wsat = r_star(ps*1e2, t10m)

#FIX CALCULATION OF RH?
#RH10m=100*(qv10m/wsat)
#
#meanRH10m = np.ma.average(RH10m, axis=0)
#
#dqstardt = dqstar_dT(meanps*1e2, meansst)
#
#Hdiff = c.cpd*tdiff + L_v*qvdiff


#ps = fSLP('SLP')
#ps = ps/1e2
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#ps = ps[tskip:,:]/1e2

LW_net_surf = radfile['LWGNT']
#LW_net_surf_cs = radfile('LWGNTCLR')
SW_net_surf = radfile['SWGNT']
#SW_net_surf_cs = radfile('SWGNTCLR')
#LW_net_TOA = radfile['LWTUP']
#lwnettoa_cs = radfile('LWTUPCLR')
#swnettoa = radfile('SWTNT')
#swnettoa_cs = radfile('SWTNTCLR')

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

Q_net_surf = LW_net_surf + SW_net_surf

#CRE_surf = Q_net_surf - Q_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH AMO
#field1 = Q_net_surf
#ftitle = r'$Q_{net}$'
#fsave = 'Qnetsurf'
#units = r'W m$^{-2}$'

#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

field1 = -thf
ftitle1 = r'-THF'
fsave1 = 'thf'
units = r'W m$^{-2}$'

#field2 = cE
#fsave2 = 'LHFdecomp_cE'
#ftitle2 = r'LHF$^{\prime}_{c_E}$'
#units = 'W m$^{-2}$'

#field3= qvdiff
#fsave3 = 'LHFdecomp_qvdiff'
#ftitle3=r'LHF$^{\prime}_{q_v}$'
#units = 'W m$^{-2}$'

#field1= qvdiff*1e3
#fsave1 = 'qvdiff'
#ftitle1=r'$\Delta q_{v,sea-air}$'
#units = 'g/kg'

#field2 = sst
#fsave2 = 'qvdiffCC'
#ftitle2=r'$\Delta q_{v, sea-air, CC}$'
#units2 = 'g/kg'

#field3= sst 
#fsave3 = 'LHFdecomp_qvdiffCC'
#ftitle3=r'LHF$^{\prime}_{q_v,CC}$'
#units = 'W m$^{-2}$'

#field = tdiff
#fsave = 'SHFdecomp_tdiff'
#ftitle =r'SHF$^{\prime}_{T}$'
#units = 'W m$^{-2}$'
#
#field = cE
#fsave = 'SHFdecomp_cE'
#ftitle =r'SHF$^{\prime}_{c_E}$'
#units = 'W m$^{-2}$'

#field2 = Hdiff
#fsave2 = 'THFdecomp_Hdiff'
#ftitle2 =r'THF$^{\prime}_{H}$'
#units = 'W m$^{-2}$'
#
#field3 = cE
#fsave3 = 'THFdecomp_cE'
#ftitle3 =r'THF$^{\prime}_{c_E}$'
#units = 'W m$^{-2}$'

#field1 = thf
#ftitle1 = r'THF'
#units = r'W m$^{-2}$'
#fsave2 = 'thf'
#units2 = r'W m$^{-2}$'

#field2= lhf
#ftitle2 = r'LHF'
#fsave2 = 'lhf'
#units = r'W m$^{-2}$'

#field3 = shf
#ftitle2 = r'SHF'
#fsave2 = 'shf'
#units = r'W m$^{-2}$'


#cf = cffile['MDSCLDFRCTTL']
#cf = cf[tskip:,:]

#field = cf*100.
#ftitle = r'$f_{total}$'
#fsave = 'ftotal'
#units = '%'

#field1 = Q_net_surf
#ftitle1 = r'$Q_{net}$'
#fsave1 = 'Qnetsurf'
#units = r'W m$^{-2}$'

field3 = LW_net_surf
ftitle3 = r'$LW_{net}$'
fsave3 = 'LWnetsurf'
units = r'W m$^{-2}$'

field2 = SW_net_surf
ftitle2 = r'$SW_{net}$'
fsave2 = 'SWnetsurf'
units = r'W m$^{-2}$'


#field1 = Hdiff*1e-3
#ftitle1 = r'$\Delta H_{sea-air}$'
#fsave1 = 'Hdiff'
#units = 'kJ/kg'

#field1 = L_v*1e-3*qvdiff
#ftitle1 = r'$L_{v}\Delta q_{v,sea-air}$'
#fsave1 = 'Lvqvdiff'
#units = 'kJ/kg'
#
#field2 = c.cpd*1e-3*tdiff
#ftitle2 = r'$c_{p}\Delta T_{,sea-air}$'
#fsave2 = 'cptdiff'
#units = 'kJ/kg'
#
#field3 = sst
#ftitle3 = r'$L_{v}\Delta q_{v,sea-air,CC}$'
#fsave3 = 'LvqvdiffCC'
#units = 'kJ/kg'

field1 = field1.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
field1 = field1[tskip:nt_ps,:]

field2 = field2.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
field2 = field2[tskip:nt_ps,:]

field3 = field3.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
field3 = field3[tskip:nt_ps,:]




#field = ps
#ftitle = 'SLP'
#fsave = 'SLP'
#units = 'hPa'

#fields = [field1, field3, field2]
#ftitles = [ftitle1, ftitle3, ftitle2]å
#fsaves = [fsave1, fsave3, fsave2]

fields = [field1, field2, field3]
ftitles = [ftitle1, ftitle2, ftitle3]
fsaves = [fsave1, fsave2, fsave3]
colors = ['C0', 'C1', 'C2']

#colors = ['r']
numfields = len(fields)
for counter, field in enumerate(fields):
    
    ftitle = ftitles[counter]
    fsave = fsaves[counter]

    #field = field.subRegion(latitude=(minlat, maxlat), longitude=(0,360))
    #field = field[tskip:nt_ps,:]
    
    sst_mask = np.ma.getmaskarray(sst)
    field_mask = np.ma.getmaskarray(field)
    field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
    field = np.ma.array(field, mask=field_mask)
    
    #True for detrending data, False for raw data
    detr=True
    corr=False
    #latitude bounds for SST averaging
    #latboundar = np.array([[0,60],[0,20],[20,45],[45,60]])
    #latboundar = np.array([[0,60]])
    
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
    
    if fsave == 'qvdiffCC' or fsave == 'LHFdecomp_qvdiffCC' or fsave == 'LvqvdiffCC':
    
        dqstardt = np.repeat(dqstardt[np.newaxis,...],sst_an.shape[0],axis=0)
        meanRH10m = np.repeat(meanRH10m[np.newaxis,...],sst_an.shape[0],axis=0)
        field_an = field_an - np.multiply(meanRH10m/100., an_ave(t10m))
        if fsave == 'qvdiffCC':
            field_an = 1e3*np.multiply(dqstardt, field_an)
        elif fsave == 'LvqvdiffCC':
           field_an = L_v*1e-3*np.multiply(dqstardt, field_an)
        else: 
            field_an = np.multiply(dqstardt, field_an)

    
    #detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
    if detr: 
     sst_an, params = detrend_separate(sst_an)
     #ps_an, params = detrend_separate(ps_an)
     field_an, params = detrend_separate(field_an)
     
    #CHANGE THIS TO MODIFY RM WINDOW LENGTH
    N_map=5
    ci = (N_map-1)/2
    
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
    
    #latboundar = np.array([[0,20],[20,45],[45,60]])
    #latboundar = np.array([[0,60]])
    latboundar = np.array([[0,20]])
    #latboundar = np.array([[45,60]])
    #latboundar = np.array([[20,45]])
    
    #EDIT BOUNDS FOR AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
    for latcount, latbounds in enumerate(latboundar):
        
        fieldlagcorrs = np.zeros((len(lags), nlat, nlon))
        fieldlagcorrs_lt = np.zeros((len(lags), nlat, nlon))
        fieldlagcorrs_st = np.zeros((len(lags), nlat, nlon))
        
        sstcorrs = MV.zeros((nlat,nlon))
        sstpvals = MV.zeros((nlat,nlon))
        fieldcorrs = MV.zeros((nlat, nlon))
        fieldcorrs_lt = MV.zeros((nlat,nlon))
        fieldcorrs_st = MV.zeros((nlat,nlon))
        
        #CHANGE THIS TO MODIFY RM WINDOW LENGTH
        N_map=5
        ci = (N_map-1)/2

        
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
            fieldmin=-0.8
            fieldmax=0.8
            fieldstep = 0.02
            cbstep = 0.2
        else:
            fieldmin=-3
            fieldmax=3
            fieldstep =0.1
            cbstep=0.5
            
        if corr:
            fieldminlag = -0.6
            fieldmaxlag = 0.6
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
        
          
        
        #CHANGE MEAN STATE ACCORDING TO VARIABLE OF INTEREST
        
#        if fsave == 'LvqvdiffCC' or fsave == 'Lvqvdiff':
#            fieldcorrs = L_v*1e-3*fieldcorrs
#            fieldcorrs_lt = L_v*1e-3*fieldcorrs_lt
#            fieldcorrs_st = L_v*1e-3*fieldcorrs_st
#        
#        if fsave == 'cptdiff':
#            fieldcorrs = c.cpd*1e-3*fieldcorrs
#            fieldcorrs_lt = c.cpd*1e-3*fieldcorrs_lt
#            fieldcorrs_st = c.cpd*1e-3*fieldcorrs_st
     
        if fsave == 'LHFdecomp_qvdiff':
            #umagbar = np.ma.average(umag,axis=0)
            #cDbar = np.ma.average(cD,axis=0)
            cEbar = np.ma.average(cE,axis=0)
            meanstate = L_v*cEbar
        elif fsave == 'LHFdecomp_cE':
            qvdiffbar = np.ma.average(qvdiff, axis=0)
            meanstate = L_v*qvdiffbar
        elif fsave == 'SHFdecomp_tdiff':
            cEbar = np.ma.average(cE,axis=0)
            meanstate = c.cpd*cEbar
        elif fsave == 'SHFdecomp_cE':
            tdiffbar = np.ma.average(tdiff, axis=0)
            meanstate = c.cpd*tdiffbar
        elif fsave == 'THFdecomp_Hdiff':
            cEbar = np.ma.average(cE,axis=0)
            meanstate = cEbar
        elif fsave == 'THFdecomp_cE':
            Hdiffbar = np.ma.average(Hdiff, axis=0)
            meanstate = Hdiffbar
        else:
            meanstate=1
        
        
        if fsave == 'THFdecomp_qvdiff' or fsave == 'SHFdecomp_cE' or fsave == 'SHFdecomp_tdiff' or fsave == 'THFdecomp_cE' or fsave == 'THFdecomp_Hdiff':
            fieldcorrs_zonalave = np.ma.average(np.multiply(meanstate[:,NAminloni:NAmaxloni], fieldcorrs[:,NAminloni:NAmaxloni]), axis=-1)
        else:
            fieldcorrs_zonalave = np.ma.average(fieldcorrs[:,NAminloni:NAmaxloni], axis=-1)
            
        fig = plt.figure(3*(2+latcount), figsize=(12,8))
        ax = fig.gca()
        if fsave == 'LvqvdiffCC' or fsave == 'qvdiffCC':
            ax.plot(fieldcorrs_zonalave, lats, color=colors[counter], linewidth=1, label='{:s}'.format(ftitle))
        else:
            ax.plot(fieldcorrs_zonalave, lats, color=colors[counter], label='{:s}'.format(ftitle))
        ax.axvline(0, color='k')
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.set_xlabel(r'{:s}'.format(units))
        ax.set_ylabel(r'latitude ($^{\circ}$)')
        ax.set_ylim(0, 60)
        #ax.set_xlim(-2.5,5.5)
        if fsave == 'qvdiff' or fsave == 'qvdiffCC':
            ax.set_xlim(-.05,0.15)
        elif fsave == 'Hdiff' or fsave == 'Lvqvdiff' or fsave == 'cptdiff' or fsave == 'LvqvdiffCC':
            ax.set_xlim(-0.5,0.5)
        else:
            ax.set_xlim(-6.5,6.5)
        #ax.set_ylim(50,1000)
        #ax.invert_yaxis()
        #cb = plt.colorbar(plot, label=r'{:s}'.format(units))
        #cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
        #cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
        plt.title(r'regression on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$'.format(latbounds[0], latbounds[1]))
        if counter == numfields-1:
            #ax.plot(fieldcorrs_zonalave+SAVE, lats, color='C0', linewidth=1, label='sum')
            #ax.plot(fieldcorrs_zonalave, lats, color=colors[counter], linewidth=1, label='{:s}'.format(ftitle))
            plt.legend()
        #plt.savefig(fout + '{:s}_AMO_surfenergy_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
        plt.savefig(fout + '{:s}_AMO_THFdecomp_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
        if counter == numfields-1:
            plt.close()
            #plt.close()
        
    i = np.where(windows>N_map)[0][0]-1
    
    #CHANGE MEAN STATE ACCORDING TO VARIABLE OF INTEREST
    if fsave == 'LHFdecomp_qvdiff':
        #umagbar = np.ma.average(umag,axis=0)
        #cDbar = np.ma.average(cD,axis=0)
        cEbar = np.ma.average(cE,axis=0)
        meanstate = L_v*cEbar
    elif fsave == 'LHFdecomp_cE':
        qvdiffbar = np.ma.average(qvdiff, axis=0)
        meanstate = L_v*qvdiffbar
    elif fsave == 'SHFdecomp_tdiff':
        cEbar = np.ma.average(cE,axis=0)
        meanstate = c.cpd*cEbar
    elif fsave == 'SHFdecomp_cE':
        tdiffbar = np.ma.average(tdiff, axis=0)
        meanstate = c.cpd*tdiffbar
    elif fsave == 'THFdecomp_Hdiff':
        cEbar = np.ma.average(cE,axis=0)
        meanstate = cEbar
    elif fsave == 'THFdecomp_cE':
        Hdiffbar = np.ma.average(Hdiff, axis=0)
        meanstate = Hdiffbar
    else:
        meanstate=1
        
    if fsave == 'THFdecomp_qvdiff' or fsave == 'THFdecomp_cE' or fsave == 'SHFdecomp_cE' or fsave == 'SHFdecomp_tdiff' or fsave == 'THFdecomp_cE' or fsave == 'THFdecomp_Hdiff':
        meanstate_temp = np.repeat(meanstate[np.newaxis,...], fieldlagcorrs_merge.shape[0], axis=0)
        meanstate_temp = meanstate_temp[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni]
        meanstate_lt_temp = np.repeat(meanstate[np.newaxis,...], fieldlagcorrs_merge_lt.shape[0], axis=0)
        meanstate_lt_temp = meanstate_lt_temp[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni]
        meanstate_st_temp = np.repeat(meanstate[np.newaxis,...], fieldlagcorrs_merge_st.shape[0], axis=0)
        meanstate_st_temp = meanstate_st_temp[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni]
        fieldlagcorrs_zonalave = np.ma.average(np.multiply(meanstate_temp, fieldlagcorrs_merge[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni]), axis=2)
        fieldlagcorrs_lt_zonalave = np.ma.average(np.multiply(meanstate_lt_temp, fieldlagcorrs_merge_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni]), axis=2)
        fieldlagcorrs_st_zonalave = np.ma.average(np.multiply(meanstate_st_temp, fieldlagcorrs_merge_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni]), axis=2)
    else:
        fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs_merge[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
        fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_merge_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
        fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_merge_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
    
    lats = lats[NAminlati:NAmaxlati]
    weights = np.cos(np.deg2rad(lats))
    tli = np.where(lats > latboundar[0][0])[0][0]
    tui = np.where(lats > latboundar[0][1]-0.001)[0][0]
    
    tfieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
    tfieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
    
    fig=plt.figure(1, figsize=(18,7))
    ax = fig.add_subplot(121)
    ax.plot(lags, tfieldlagcorrs_lt_ave, color=colors[counter], label='{:s}'.format(ftitle))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    if corr:
        ax.set_title('long-term correlation with AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(latboundar[0][0], latboundar[0][1], windows[i]))
    else:
        ax.set_title('long-term regression on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(latboundar[0][0], latboundar[0][1], windows[i]))
    ax.set_xlabel('lag (years)')
    ax = fig.add_subplot(122)
    ax.plot(lags, tfieldlagcorrs_st_ave, color=colors[counter], label='{:s}'.format(ftitle))
    ax.set_ylim(fieldminlag,fieldmaxlag)
    ax.set_ylabel(r'{:s}'.format(fieldunitslag))
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if corr:
        ax.set_title('short-term correlation')
    else:
        ax.set_title('short-term regression')
    ax.set_xlabel('lag (years)')
    if counter == numfields-1:
           plt.legend()
    if corr:
        plt.savefig(fout + '{:s}_AMO_THFdecomp_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
        #plt.savefig(fout + '{:s}_AMO_surfenergy_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
    else:
        plt.savefig(fout + '{:s}_AMO_THFdecomp_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
        #plt.savefig(fout + '{:s}_AMO_surfenergy_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
    if counter == numfields-1:
            plt.close()
    
    if len(latboundar) > 1:
        mui = np.where(lats > latboundar[1][1]-0.001)[0][0]
        eui = np.where(lats > latboundar[2][1]-0.001)[0][0]
        mfieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,tui:mui],axis=1,weights=weights[tui:mui])
        mfieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,tui:mui],axis=1,weights=weights[tui:mui])
    
        efieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,mui:eui],axis=1,weights=weights[mui:eui])
        efieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,mui:eui],axis=1,weights=weights[mui:eui])
    
        fig=plt.figure(2, figsize=(18,7))
        ax = fig.add_subplot(121)
        ax.plot(lags, mfieldlagcorrs_lt_ave, color=colors[counter], label='{:s}'.format(ftitle))
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.set_ylim(fieldminlag,fieldmaxlag)
        ax.set_ylabel(r'{:s}'.format(fieldunitslag))
        if corr:
            ax.set_title('long-term correlation with AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(latboundar[0][1], latboundar[1][1], windows[i]))
        else:
            ax.set_title('long-term regression on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(latboundar[0][1], latboundar[1][1], windows[i]))
        ax.set_xlabel('{:s} lag (years)'.format(ftitle))
        ax = fig.add_subplot(122)
        ax.plot(lags, mfieldlagcorrs_st_ave, color=colors[counter], label='{:s}'.format(ftitle))
        ax.set_ylim(fieldminlag,fieldmaxlag)
        ax.set_ylabel(r'{:s}'.format(fieldunitslag))
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        if corr:
            ax.set_title('short-term correlation')
        else:
            ax.set_title('short-term regression')
        ax.set_xlabel('lag (years)')
        if counter == numfields-1:
           plt.legend()
        if corr:
            plt.savefig(fout + '{:s}_AMO_THFdecomp_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][1], latboundar[1][1], str(detr)[0]))
            #plt.savefig(fout + '{:s}_AMO_surfenergy_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][1], latboundar[1][1], str(detr)[0]))
        else:
            plt.savefig(fout + '{:s}_AMO_THFdecomp_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][1], latboundar[1][1],  str(detr)[0]))
            #plt.savefig(fout + '{:s}_AMO_surfenergy_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[0][1], latboundar[1][1],  str(detr)[0]))
        if counter == numfields-1:
            plt.close()
        
        
        fig=plt.figure(3, figsize=(18,7))
        ax = fig.add_subplot(121)
        ax.plot(lags, efieldlagcorrs_lt_ave, color=colors[counter], label='{:s}'.format(ftitle))
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        ax.set_ylim(fieldminlag,fieldmaxlag)
        ax.set_ylabel(r'{:s}'.format(fieldunitslag))
        if corr:
            ax.set_title('long-term correlation with AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(latboundar[1][1], latboundar[2][1], windows[i]))
        else:
            ax.set_title('long-term regression on AMO$_{{{:1.0f}^{{\circ}}-{:2.0f}^{{\circ}}}}$ ({:1.0f}-yr RM)'.format(latboundar[1][1], latboundar[2][1], windows[i]))
        ax.set_xlabel('lag (years)')
        ax = fig.add_subplot(122)
        ax.plot(lags, efieldlagcorrs_st_ave, color=colors[counter], label='{:s}'.format(ftitle))
        ax.set_ylim(fieldminlag,fieldmaxlag)
        ax.set_ylabel(r'{:s}'.format(fieldunitslag))
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        if corr:
            ax.set_title('short-term correlation')
        else:
            ax.set_title('short-term regression')
        ax.set_xlabel('lag (years)')
        if counter == numfields-1:
           plt.legend()
        if corr:
            plt.savefig(fout + '{:s}_AMO_THFdecomp_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[1][1], latboundar[2][1],  str(detr)[0]))
            #plt.savefig(fout + '{:s}_AMO_surfenergy_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[1][1], latboundar[2][1],  str(detr)[0]))
        else:
            plt.savefig(fout + '{:s}_AMO_THFdecomp_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[1][1], latboundar[2][1], str(detr)[0]))
            #plt.savefig(fout + '{:s}_AMO_surfenergy_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latboundar[1][1], latboundar[2][1], str(detr)[0]))
        if counter == numfields-1:
            plt.close()
        













































