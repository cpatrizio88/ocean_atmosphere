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
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, regressout_x, butter_lowpass_filter
from matplotlib.patches import Polygon
from palettable.cubehelix import Cubehelix

cx4 = Cubehelix.make(reverse=True, start=0.3, rotation=-0.5)

#cx3 = Cubehelix.cmap(reverse=True, start=0.3, rot=-0.5)

fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/NA index/'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_isccp_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')

#fthf = cdms2.open(fin + 'thf.197901-201712.nc')
#fsst = cdms2.open(fin + 'sstslp.197901-201712.nc')
#fuv = cdms2.open(fin + 'uv.197901-201712.nc')


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

print 'loading fields...'

#era-i fields
#ps = fsst('msl')/1e2
#sst = fsst('sst')
#lhf = fthf('slhf')/(12*60*60)
#shf = fthf('sshf')/(12*60*60)
#u = fuv('u10')
#v = fuv('v10')
#
#thf = lhf + shf
#thf = -thf


#MERRA-2 fields
sst = fsst('TSKINWTR')
lhf = fsst('EFLUXWTR')
shf = fsst('HFLUXWTR')
u = fuv('U10M')
v = fuv('V10M')

ps = fSLP('SLP')
ps = ps/1e2

cflo = cffile('MDSCLDFRCLO')
cfhi = cffile('MDSCLDFRCHI')
cfttl = cffile('MDSCLDFRCTTL')
#cf = cffile('ISCCPCLDFRC')


#thf is positive down in ERAi, convert to positive up
#thf = -thf 

#sst_index = sst.subRegion(latitude=(indexlatbounds[0], indexlatbounds[1]), longitude=(indexlonbounds[0], indexlatbounds[1]))
#
#indexlats = sst_index.getLatitude()[:]
#
#index = spatial_ave(sst_index, indexlats)

#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)

thf = lhf + shf




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

Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

Q_net_surf = LW_net_surf + SW_net_surf

CRE_surf = Q_net_surf - Q_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH index
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

#field = LW_net_surf
#ftitle = r'$LW_{net}$'
#fsave = 'LWnetsurf'
#units = r'W m$^{-2}$'
#
#field = SW_net_surf
#ftitle = r'$SW_{net}$'
#fsave = 'SWnetsurf'
#units = r'W m$^{-2}$'


field2 = CRE_surf
ftitle2 = r'CRE$_{surf}$'
fsave2 = 'CREsurf'
units2 = r'W m$^{-2}$'

field1 = thf
ftitle1 = r'THF'
fsave1 = 'thf'
units1 = r'W m$^{-2}$'

#field = lhf
#ftitle = r'LHF'
#fsave = 'lhf'
#units = r'W m$^{-2}$'

#field = shf
#ftitle = r'SHF'
#fsave = 'shf'
#units = r'W m$^{-2}$'

#field1 = cfttl*100.
#ftitle1 = r'$f_{total}$'
#fsave1 = 'ftotal'
#units1 = '%'

#field2 = cflo*100.
#ftitle2 = r'$f_{low}$'
#fsave2 = 'flow'
#units2 = '%'

#field3 = cfhi*100.
#ftitle3 = r'$f_{high}$'
#fsave3 = 'fhigh'
#units3 = '%'

fname = 'CREsurfTHF'
sstsave = 'dSSTdt'

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


    

ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))

if field1.shape[0] < ps.shape[0]:
    nt_ps = field1.shape[0]
    ps = ps[tskip:nt_ps,:]
else:
    nt_ps = ps.shape[0]
    ps = ps[tskip:,:]
    
    
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:nt_ps,:]

u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
u = u[tskip:nt_ps,:]


v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
v = v[tskip:nt_ps,:]

sst_mask = np.ma.getmaskarray(sst)
ps_mask = np.ma.getmaskarray(ps)
ps_mask = np.ma.mask_or(sst_mask[:ps.shape[0],:,:], ps_mask)
ps = np.ma.array(ps, mask=ps_mask)
    
#fieldlist = [field1, field2, field3]
#titlelist = [ftitle1, ftitle2, ftitle3]
#fsavelist = [fsave1, fsave2, fsave3]
#unitslist = [units1, units2, units3]

fieldlist = [field1, field2]
titlelist = [ftitle1, ftitle2]
fsavelist = [fsave1, fsave2]
unitslist = [units1, units2]

#True for detrending data, False for raw data
detr=True
corr=False
lterm=False
drawbox=True
rENSO=True
plotwinds=True
plotSLP=True


#EDIT THIS FOR BOUNDS
lonbounds = [280.,360.]
latbounds = [0, 70.]


print 'detrending...'
#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
 ps = detrend(ps)
 u = detrend(u)
 v = detrend(v)
 
 #subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(u)
cdutil.setTimeBoundsMonthly(v)
cdutil.setTimeBoundsMonthly(ps)

u = cdutil.ANNUALCYCLE.departures(u)
v = cdutil.ANNUALCYCLE.departures(v)
ps = cdutil.ANNUALCYCLE.departures(ps)
sst = cdutil.ANNUALCYCLE.departures(sst)

lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]

CTIminlati = np.argmin(np.abs(lats - (-6)))
CTImaxlati = np.argmin(np.abs(lats - 6))
CTIminloni = np.argmin(np.abs(lons - 0))
CTImaxloni = np.argmin(np.abs(lons - 90))
 
CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])

# CTI Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 3.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
 
CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])


CTI = butter_lowpass_filter(CTI, cutoff, fs, order)

sst = sst.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
u = u.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
v = v.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
ps = ps.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))


lats = sst.getLatitude()[:]
lons = sst.getLongitude()[:]

nlat = len(lats)
nlon = len(lons)

t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
tyears = 1980 + t/12.


if rENSO:
    CTIlag=2
    sst = regressout_x(sst[CTIlag:,...], CTI[:-CTIlag])
    #field = regressout_x(field[CTIlag:,...], CTI[:-CTIlag])
    u = regressout_x(u[CTIlag:,...], CTI[:-CTIlag])
    v = regressout_x(v[CTIlag:,...], CTI[:-CTIlag])
    ps = regressout_x(ps[CTIlag:,...], CTI[:-CTIlag])
    #u = u[CTIlag:,...]
    #v = v[CTIlag:,...]
    #ps = ps[CTIlag:,...]
    tyears = tyears[CTIlag:,...]
        
    print 'max SST', np.ma.max(sst)
    print 'max u,v', np.ma.max(u), np.ma.max(v)  
    print 'max ps', np.ma.max(ps)
    
    
sstprime = sst
uprime = u
vprime = v
psprime = ps

# CTI Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12*7.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

# apply the filter
sst_lt = butter_lowpass_filter(sstprime, cutoff, fs, order)
u_lt = butter_lowpass_filter(uprime, cutoff, fs, order)
v_lt = butter_lowpass_filter(vprime, cutoff, fs, order)
ps_lt = butter_lowpass_filter(psprime, cutoff, fs, order)

sst_st = sstprime - sst_lt
u_st = uprime - u_lt
v_st = vprime - v_lt
ps_st = psprime - ps_lt

if sstsave == 'dSSTdt':
   sstprime = (sstprime[2:,...]-sstprime[:-2,...])/2.
   sst_lt = (sst_lt[2:,...]-sst_lt[:-2,...])/2.
   sst_st = (sst_st[2:,...]-sst_st[:-2,...])/2.
   uprime = uprime[1:-1,...]
   u_lt = u_lt[1:-1,...]
   u_st = u_st[1:-1,...]
   vprime = vprime[1:-1,...]
   v_lt = v_lt[1:-1,...]
   v_st = v_st[1:-1,...]
   psprime = psprime[1:-1,...]
   ps_lt = ps_lt[1:-1,...]
   ps_st = ps_st[1:-1,...]
   #sstprime = sstprime/(3600*24*30.)
   #sstprime = sstprime/(3600*24*30.)
   #sstprime = sstprime/(3600*24*30.)
   
   tyears = tyears[1:-1,...]


sstcorrs = MV.zeros((nlat,nlon))
sstcorrs_lt = MV.zeros((nlat, nlon))
sstcorrs_st = MV.zeros((nlat, nlon))
pscorrs = MV.zeros((nlat, nlon))
pscorrs_lt = MV.zeros((nlat,nlon))
pscorrs_st = MV.zeros((nlat,nlon))
ucorrs = MV.zeros((nlat,nlon))
ucorrs_lt = MV.zeros((nlat,nlon))
ucorrs_st = MV.zeros((nlat,nlon))
vcorrs = MV.zeros((nlat,nlon))
vcorrs_lt = MV.zeros((nlat,nlon))
vcorrs_st = MV.zeros((nlat,nlon))

fieldend = len(fieldlist)-1

for fieldcount, field in enumerate(fieldlist):
    
    units = unitslist[fieldcount]
    ftitle = titlelist[fieldcount]
    fsave = fsavelist[fieldcount]
    

    if not(fsave == 'sst'):
        field = field.subRegion(latitude=(minlat, maxlat), longitude=(minlon,maxlon))
        field = field[tskip:nt_ps,:]

    #field = field[:ps.shape[0],...]/qsat

    field_mask = np.ma.getmaskarray(field)
    field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
    ##field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
    field = np.ma.array(field, mask=field_mask)
    
    fieldsave = field
    
    print 'detrending...'
    #detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
    if detr: 
     field = detrend(field)
     
    print 'max field', np.ma.max(field)
     
    lats = field.getLatitude()[:]
    lons = field.getLongitude()[:]
     
    
    
    t = field.getTime().asRelativeTime("months since 1980")
    t = np.array([x.value for x in t])
    tyears = 1980 + t/12.
    
    
    
    cdutil.setTimeBoundsMonthly(field)
    
    print 'subtracting seasonal cycle...'
    field = cdutil.ANNUALCYCLE.departures(field)

    
    print 'max field', np.ma.max(field)
    
    print 'getting subregion...'

    field = field.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))

    
    
    
    lats = field.getLatitude()[:]
    lons = field.getLongitude()[:]
    nlat = len(lats)
    nlon = len(lons)
    nt = sst.shape[0]
    
    
    print 'regressing out CTI...'
    
    if rENSO:
        CTIlag=2
        field = field[CTIlag:,...]
        
    print 'max field', np.ma.max(field)
    

    fieldprime = field
    
    
    
    print 'computing long-term/short-term fields...'
    # apply the filter
    field_df = pd.DataFrame(fieldprime.reshape(nt, nlat*nlon))
    #sst_df = sst_df.interpolate()
    field_df = field_df.interpolate()
    #sstprime = sst_df.values.reshape(nt, nlat, nlon)
    fieldprime = field_df.values.reshape(nt, nlat, nlon)
    
    field_lt = butter_lowpass_filter(fieldprime, cutoff, fs, order)
    
    fieldprime = np.ma.masked_array(fieldprime, mask=~np.isfinite(fieldprime))
    field_lt = np.ma.masked_array(field_lt, mask=~np.isfinite(field_lt))
    
    field_st =  fieldprime - field_lt
    
    if sstsave == 'dSSTdt':
           fieldprime = fieldprime[1:-1,...]
           field_lt = field_lt[1:-1,...]
           field_st = field_st[1:-1,...]


    fieldcorrs = MV.zeros((nlat, nlon))
    fieldcorrs_lt = MV.zeros((nlat,nlon))
    fieldcorrs_st = MV.zeros((nlat,nlon))
    
    #interpolate to coarser grid to speed up 
    #sst= sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
    #field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
    latw = 20  
    slats = np.arange(0,latw,latw)   
        
    for slati in slats:  
        
        nlati = slati+latw
        wloni = 290
        eloni = 350
        
        si = np.argmin(np.abs(lats - slati))
        ni = np.argmin(np.abs(lats - nlati))
        wi = np.argmin(np.abs(lons - wloni))
        ei = np.argmin(np.abs(lons - eloni))
        
        index = spatial_ave(sst[:,si:ni,wi:ei], lats[si:ni])
    
        
        index_lt = butter_lowpass_filter(index, cutoff, fs, order)
        index_st = index - index_lt
    
       
        
        
        nt = field.shape[0]
        nt_lt = sst_lt.shape[0]
        
        
        #lats = lats[NAminlati:NAmaxlati]
        #lons = lons[NAminloni:NAmaxloni]
        #
        #nlat = len(lats)
        #nlon = len(lons)
        
        scaler = StandardScaler()
        indexstd = scaler.fit_transform(index.reshape(-1,1))
        indexstd_lt = scaler.fit_transform(index_lt.reshape(-1,1))
        indexstd_st = scaler.fit_transform(index_st.reshape(-1,1))
        
        indexstdrep = np.squeeze(np.repeat(indexstd[:,np.newaxis], nlon, axis=1))
        indexstd_ltrep = np.squeeze(np.repeat(indexstd_lt[:,np.newaxis], nlon, axis=1))
        indexstd_strep = np.squeeze(np.repeat(indexstd_st[:,np.newaxis], nlon, axis=1))
        
        
        
        #compute correlation between long-term/short-term index and 2D field
        print r'calculating correlations between index and {:s}...'.format(ftitle)
        for i in range(nlat):   
                print 'latitude', lats[i]
             
             #for j in range(nlon):
                 
                sstprime_g = sstprime[:,i,:]
                fieldprime_g = fieldprime[:,i,:]
                uprime_g = uprime[:,i,:]
                vprime_g = vprime[:,i,:]
                psprime_g = psprime[:,i,:]
                
                field_lt_g = field_lt[:,i,:]
                field_st_g = field_st[:,i,:]
                sst_lt_g = sst_lt[:,i,:]
                sst_st_g = sst_st[:,i,:]
                
                u_lt_g = u_lt[:,i,:]
                u_st_g = u_st[:,i,:]
                v_lt_g = v_lt[:,i,:]
                v_st_g = v_st[:,i,:]
                ps_lt_g = ps_lt[:,i,:]
                ps_st_g = ps_st[:,i,:]
                
                
                N = fieldprime_g.shape[1]
                
            
                clf = linear_model.LinearRegression()
                clf.fit(indexstd.reshape(-1,1), sstprime_g)
                sstcorrs[i,:] = np.squeeze(clf.coef_)
                
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_lt.reshape(-1,1), sst_lt_g)
                sstcorrs_lt[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_st.reshape(-1,1), sst_st_g)
                sstcorrs_st[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd.reshape(-1,1), uprime_g)
                ucorrs[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_lt.reshape(-1,1), u_lt_g)
                ucorrs_lt[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_st.reshape(-1,1), u_st_g)
                ucorrs_st[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd.reshape(-1,1), vprime_g)
                vcorrs[i,:] = np.squeeze(clf.coef_)
                 
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_lt.reshape(-1,1), v_lt_g)
                vcorrs_lt[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_st.reshape(-1,1), v_st_g)
                vcorrs_st[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd.reshape(-1,1), psprime_g)
                pscorrs[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_lt.reshape(-1,1), ps_lt_g)
                pscorrs_lt[i,:] = np.squeeze(clf.coef_)
             
                clf = linear_model.LinearRegression()
                clf.fit(indexstd_st.reshape(-1,1), ps_st_g)
                pscorrs_st[i,:] = np.squeeze(clf.coef_)
                
                
                coefs = np.diag(np.ma.cov(indexstdrep, fieldprime_g, rowvar=False)[:N,N:])
                coefs_lt = np.diag(np.ma.cov(indexstd_ltrep, field_lt_g, rowvar=False)[:N,N:])
                coefs_st = np.diag(np.ma.cov(indexstd_strep, field_st_g, rowvar=False)[:N,N:])
                
                fieldcorrs[i,:] = coefs
                fieldcorrs_lt[i,:] = coefs_lt
                fieldcorrs_st[i,:] = coefs_st
                
             
    #            clf = linear_model.LinearRegression()
    #            clf.fit(indexstd.reshape(-1,1), fieldprime_g)
    #            fieldcorrs[i,:] = np.squeeze(clf.coef_)
    #         
    #            clf = linear_model.LinearRegression()
    #            clf.fit(indexstd_lt.reshape(-1,1), field_lt_g)
    #            fieldcorrs_lt[i,:] = np.squeeze(clf.coef_)
    #         
    #            clf = linear_model.LinearRegression()
    #            clf.fit(indexstd_st.reshape(-1,1), field_st_g)
    #            fieldcorrs_st[i,:] = np.squeeze(clf.coef_)
               
        #lonbounds = [0.1,359.99]
        #latbounds = [minlat, maxlat]
        #    
        #NAminlati = np.where(lats >= latbounds[0])[0][0]
        #NAmaxlati = np.where(lats >= latbounds[1])[0][0]
        #NAminloni = np.where(lons >= lonbounds[0])[0][0]
        #NAmaxloni = np.where(lons >= lonbounds[1])[0][0]
        
        
        #fieldlagcorrs_zonalave = np.ma.average(fieldlagcorrs[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
        #fieldlagcorrs_lt_zonalave = np.ma.average(fieldlagcorrs_lt[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
        #fieldlagcorrs_st_zonalave = np.ma.average(fieldlagcorrs_st[:,NAminlati:NAmaxlati,NAminloni:NAmaxloni], axis=2)
        
        
              
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
            fieldstep = 0.1
            cbstep = 1
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
        else:
            fieldmin=-10
            fieldmax=10
            fieldstep =0.2
            cbstep=2.5
            
        sstmin = -0.8
        sstmax = 0.8   
        sststep =0.02
        sstcbstep = 0.2
        psmax=1.0
        psmin = -1.0
        psstep = 0.1
        pscbstep = 0.1
        thfmin=-3
        thfmax=3
        thfstep =0.02
        thfcbstep=1.0
        SLPlevels = np.arange(psmin, psmax+psstep, psstep)
        sstlevels = np.arange(sstmin, sstmax+sststep, sststep)
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
            sstminlag = sstmin
            sstmaxlag = sstmax
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
            
        psticks = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag),1)
        psticklbls = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag), 1)
        psticklbls[psticklbls == -0.00] = 0.00
        
        sstticks = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag),1)
        sstticklbls = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag), 1)
        sstticklbls[sstticklbls == -0.00] = 0.00
        
        
            
        ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
        ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
        ticklbls[ticklbls == -0.0] = 0.0
                         
        #sstlevels = np.arange8, 0.8+sststep, sststep)
        fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
        
        orient = 'horizontal'
        if lonbounds[1] - lonbounds[0] <= 180:
            orient = 'vertical'
            
        uskip=8
        
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
        ax.add_feature(cart.feature.LAND, zorder=50, edgecolor='k', facecolor='grey')
        ax.set_xticks(mer, crs=prj)
        ax.set_yticks(par, crs=prj)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
        ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
        #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
        if np.any(np.round(ct.levels, 5) == 0):
            ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
        #ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
        plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
        cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
        cb.set_ticks(sstticks)
        cb.set_ticklabels(sstticklbls)
        for line in ct.collections:
            if line.get_linestyle() != [(None, None)]:
                line.set_linestyle([(0, (8.0, 8.0))])
        if plotwinds:
            qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs[::uskip,::uskip], vcorrs[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
            ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
        if drawbox:
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
            ax.add_patch(poly)
        plt.title(r'SST, SLP and 10-m Winds')
        #plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
        plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_sstSLPuv_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], latbounds[0], latbounds[1], str(detr)[0]))
        #plt.savefig(fout + '{:s}_AMO_sst_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
        plt.close()
        
        if lterm:
        
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
            ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
            ct = ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
            #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
            if np.any(np.round(ct.levels, 5) == 0):
                ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
            #ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
            for line in ct.collections:
                if line.get_linestyle() != [(None, None)]:
                    line.set_linestyle([(0, (8.0, 8.0))])
            plot = ax.contourf(x, y, sstcorrs_lt, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
            cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
            cb.set_ticks(sstticks)
            cb.set_ticklabels(sstticklbls)
            if plotwinds:
                ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_lt[::uskip,::uskip], vcorrs_lt[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
                ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
            if drawbox:
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
                ax.add_patch(poly)
            plt.title(r'Long-term SST, SLP and 10-m Winds ({:1.0f}-yr LP)'.format(Tn/12.))
            plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_sstSLPuv_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], latbounds[0], latbounds[1], str(detr)[0]))
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
            ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
            ct = ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
            #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
            for line in ct.collections:
                if line.get_linestyle() != [(None, None)]:
                    line.set_linestyle([(0, (8.0, 8.0))])
            if np.any(np.round(ct.levels, 5) == 0):
                ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
            #ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
            plot = ax.contourf(x, y, sstcorrs_st, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
            cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
            cb.set_ticks(sstticks)
            cb.set_ticklabels(sstticklbls)
            if plotwinds:
                ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_st[::uskip,::uskip], vcorrs_st[::uskip,::uskip], transform=cart.crs.PlateCarree(),  scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
                ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
            if drawbox:
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
                ax.add_patch(poly)
            plt.title(r'Short-term SST, SLP and 10-m Winds ({:1.0f}-yr HP)'.format(Tn/12.))
            plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_sstSLPuv_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], latbounds[0], latbounds[1], str(detr)[0]))
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
        plot = ax.contourf(x, y, fieldcorrs, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())    
        cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
        if drawbox:
            poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
            ax.add_patch(poly)
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklbls)
        plt.title(r'{:s}'.format(ftitle))
        plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
            plt.title(r'Long-term {:s} and SLP ({:1.0f}-yr LP)'.format(ftitle, Tn/12.))
            plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
            if drawbox:
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
                ax.add_patch(poly)
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklbls)
            plt.title(r'Short-term {:s} and SLP ({:1.0f}-yr HP)'.format(ftitle, Tn/12.))
            plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
            plt.close()
            
        
    
        if plotSLP:    
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
            ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
            #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
            for line in ct.collections:
                if line.get_linestyle() != [(None, None)]:
                    line.set_linestyle([(0, (8.0, 8.0))])
            if np.any(np.round(ct.levels, 5) == 0):
                ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
            if drawbox:
                poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
                ax.add_patch(poly)
            cb.set_ticks(ticks)
            cb.set_ticklabels(ticklbls)
            plt.title(r'{:s}'.format(ftitle))
            plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}SLP_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
                ct = ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
                #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
                for line in ct.collections:
                    if line.get_linestyle() != [(None, None)]:
                        line.set_linestyle([(0, (8.0, 8.0))])
                if np.any(np.round(ct.levels, 5) == 0):
                    ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
                plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
                cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklbls)
                if drawbox:
                    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
                    ax.add_patch(poly)
                plt.title(r'Long-term {:s} and SLP ({:1.0f}-yr LP)'.format(ftitle, Tn/12.))
                plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}SLP_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
                ct = ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
                #ax.clabel(ct, ct.levels[np.abs(np.round(ct.levels, 5)) != 0], fontsize=9, inline=1, fmt='%1.1f')
                for line in ct.collections:
                    if line.get_linestyle() != [(None, None)]:
                        line.set_linestyle([(0, (8.0, 8.0))])
                if np.any(np.round(ct.levels, 5) == 0):
                    ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
                if drawbox:
                    poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='black',linewidth=3,zorder=100)
                    ax.add_patch(poly)
                cb.set_ticks(ticks)
                cb.set_ticklabels(ticklbls)
                plt.title(r'Short-term {:s} and SLP ({:1.0f}-yr HP)'.format(ftitle, Tn/12.))
                plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}SLP_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
                plt.close()
        
        
    fieldcorrs_zonalave = np.ma.average(fieldcorrs, axis=-1)            
    #plt zonal average of fieldcorrs
    plt.figure(20, figsize=(14,10))           
    plt.plot(fieldcorrs_zonalave, lats, label=r'{:s}'.format(ftitle))
    plt.axvline(0, color='k')
    plt.xlabel('{:s}'.format(units))  
    plt.ylabel('Latitude (degrees)')
    if fieldcount == fieldend:
       plt.legend()
       plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fname, latbounds[0], latbounds[1], str(detr)[0]))
       plt.close()
    else:
       plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fname, latbounds[0], latbounds[1], str(detr)[0]))
       
    fieldcorrs_lt_zonalave = np.ma.average(fieldcorrs_lt, axis=-1)            
    #plt zonal average of fieldcorrs
    plt.figure(21, figsize=(14,10))           
    plt.plot(fieldcorrs_lt_zonalave, lats, label=r'{:s}'.format(ftitle))
    plt.axvline(0, color='k')
    plt.xlabel('{:s}'.format(units))  
    plt.ylabel('Latitude (degrees)')
    plt.title('Long-term ({:1.0f}-yr LP)'.format(Tn/12.))
    if fieldcount == fieldend:
       plt.legend()
       plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_ltzonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fname, latbounds[0], latbounds[1], str(detr)[0]))
       plt.close()
    else:
       plt.savefig(fout + '{:s}_{:s}{:2.0f}Nto{:2.0f}N_{:s}_ltzonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, sstsave, lats[si], lats[ni], fname, latbounds[0], latbounds[1], str(detr)[0]))
 
       
##plot mean field 
##            
#if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
#    fieldmin=20
#    fieldmax=80
#    fieldstep = 1
#    cbstep = 20
#elif fsave == 'cldfrac3D':
#    fieldmin = 0
#    fieldmax = 100
#    fieldstep = 0.5
#    cbstep = 20
#elif fsave == 'qv3D':
#    fieldmin=0
#    fieldmax=5
#    fieldstep=0.01
#    cbstep=0.1
#elif fsave == 'RH':
#    fieldmin= 0
#    fieldmax= 100
#    fieldstep= 5
#    cbstep = 20
#elif fsave == 't3D':
#    fieldmin = 200
#    fieldmax = 300
#    fieldstep=5
#    cbstep = 25
#elif fsave == 'omega':
#    fieldmin = -10
#    fieldmax = 10
#    fieldstep = 0.1
#    cbstep = 2.5
#else:
#    fieldmin=-10
#    fieldmax=10
#    fieldstep =0.2
#    cbstep=2.5
#            
#fieldmean = fieldsave.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
#fieldmean = np.ma.average(fieldmean,axis=0)
#
#ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
#ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
#ticklbls[ticklbls == -0.0] = 0.0
#                 
##sstlevels = np.arange8, 0.8+sststep, sststep)
#fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
#    
#plt.figure(figsize=(16,12))
#ax = plt.axes(projection=prj)
#ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=99, edgecolor='k', facecolor='grey')
#ax.set_xticks(mer, crs=prj)
#ax.set_yticks(par, crs=prj)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)
#ax.get_yaxis().set_tick_params(direction='out')
#ax.get_xaxis().set_tick_params(direction='out')
#ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
##ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
#plot = ax.contourf(x, y, fieldmean, levels=fieldlevels, cmap=cx4.mpl_colormap, extend='both', transform=cart.crs.PlateCarree())    
#cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
##cb.set_ticks(ticks)
##cb.set_ticklabels(ticklbls)
#plt.title(r'{:s}'.format(ftitle))
#plt.savefig(fout + '{:s}_SST{:2.0f}Nto{:2.0f}N_MEAN{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()














































