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
from sklearn.decomposition import PCA
from ocean_atmosphere.misc_fns import detrend, an_ave, spatial_ave, running_mean, calc_NA_globeanom, detrend_separate, detrend_common, regressout_x
from matplotlib.patches import Polygon


fin = '/Users/cpatrizio/data/MERRA2/'
#fin = '/Users/cpatrizio/data/ECMWF/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/MM/'

#MERRA-2
fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')
fuv = cdms2.open(fin + 'MERRA2_uv_monthly1980to2017.nc')
#fRH = cdms2.open(fin + 'MERRA2_qv10m_monthly1980to2017.nc')
fcE = cdms2.open(fin + 'MERRA2_cE_monthly1980to2017.nc')
#fcD = cdms2.open(fin + 'MERRA2_cD_monthly1980to2017.nc')

MMf = np.loadtxt(fin + 'AMM.txt', skiprows=1)

NAOf = np.loadtxt(fin + 'NAO.monthly.txt')

NAO = NAOf[:,2]
NAOyr = NAOf[:,0]
NAOmo = NAOf[:,1]

NAOt = NAOyr + NAOmo/12.

MM = MMf[:,2]
MMyr = MMf[:,0]
MMmo = MMf[:,1]

MMt = MMyr + MMmo/12.

MMwind = MMf[:,3]


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

ps = fSLP('SLP')
ps = ps/1e2
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
nt_ps = ps.shape[0]
ps = ps[tskip:,:]

#sst = fsst('skt')
sst = fsst('TSKINWTR')
#lats = sst.getLatitude()[:]
#sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:nt_ps,:]

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

#ps = fSLP('SLP')
#ps = ps/1e2
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#ps = ps[tskip:,:]/1e2

u = fuv('U10M')
#u = u.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
u = u[tskip:nt_ps,:]

v = fuv('V10M')

#v = v.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
v = v[tskip:nt_ps,:]

#umag = np.sqrt(np.square(v) + np.square(u))

#qv10m = fRH('QV10M')
#LW_net_surf = radfile['LWGNT']
##LW_net_surf_cs = radfile('LWGNTCLR')
#SW_net_surf = radfile['SWGNT']
#SW_net_surf_cs = radfile('SWGNTCLR')
#LW_net_TOA = radfile['LWTUP']
#lwnettoa_cs = radfile('LWTUPCLR')
#swnettoa = radfile('SWTNT')
#swnettoa_cs = radfile('SWTNTCLR')

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

#Q_net_surf = LW_net_surf + SW_net_surf

#CRE_surf = Q_net_surf - Q_net_surf_cs

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

#field = LW_net_surf
#ftitle = r'$LW_{net}$'
#fsave = 'LWnetsurf'
#units = r'W m$^{-2}$'
#
#field = SW_net_surf
#ftitle = r'$SW_{net}$'
#fsave = 'SWnetsurf'
#units = r'W m$^{-2}$'


#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

field = thf
ftitle = r'THF'
fsave = 'thf'
units = r'W m$^{-2}$'

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

#field = field.subRegion(latitude=(minlat, maxlat), longitude=(minlon,maxlon))
field = field[tskip:nt_ps,:]

#field = field[:ps.shape[0],...]/qsat

sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
##field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)
ps_mask = np.ma.getmaskarray(ps)
ps_mask = np.ma.mask_or(sst_mask[:ps.shape[0],:,:], ps_mask)
ps = np.ma.array(ps, mask=ps_mask)

#True for detrending data, False for raw data
detr=True
corr=False

lterm=False

#EDIT THIS FOR BOUNDS
lonbounds = [280.,360.]
latbounds = [-30, 70.]

print 'detrending...'
#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
 ps = detrend(ps)
 u = detrend(u)
 v = detrend(v)
 field = detrend(field)
 
lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
nlat = len(lats)
nlon = len(lons)
nt = field.shape[0]


t = field.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
tyears = 1980 + t/12.

MMstarti = np.where(MMt == tyears[0])[0][0]
MMendi = np.where(MMt == tyears[-1])[0][0]

MM = MM[MMstarti:MMendi+1]
MMwind = MMwind[MMstarti:MMendi+1]

NAOstarti = np.where(NAOt == tyears[0])[0][0]
NAOendi = np.where(NAOt == tyears[-1])[0][0]

NAO = NAO[NAOstarti:NAOendi+1]


#tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(field)
cdutil.setTimeBoundsMonthly(u)
cdutil.setTimeBoundsMonthly(v)
cdutil.setTimeBoundsMonthly(ps)
print 'subtracting seasonal cycle...'
field = cdutil.ANNUALCYCLE.departures(field)
u = cdutil.ANNUALCYCLE.departures(u)
v = cdutil.ANNUALCYCLE.departures(v)
ps = cdutil.ANNUALCYCLE.departures(ps)
sst = cdutil.ANNUALCYCLE.departures(sst)

#ps_NAO = ps.subRegion(latitude = (0,90), longitude=(lonbounds[0], lonbounds[1]))
#latNAO = ps_NAO.getLatitude()[:]
#lonNAO = ps_NAO.getLongitude()[:]
#
#nlatNAO = len(latNAO)
#nlonNAO = len(lonNAO)
#
#ps_NAO = ps_NAO.reshape(nt, nlatNAO*nlonNAO)
#
#pca = PCA(n_components=1)
#pca.fit(ps_NAO)
#
#pca.fit_transform(ps_NAO)
#
#eof1 = pca.components_
#
#eof1 = eof1.reshape(nlatNAO, nlonNAO)

NAOminlati = np.argmin(np.abs(lats - (0)))
NAOmaxlati = np.argmin(np.abs(lats - 90))
midlati = np.argmin(np.abs(lats - 40))
NAOminloni = np.argmin(np.abs(lons + 180 - lonbounds[0]))
NAOmaxloni = np.argmin(np.abs(lons + 180 - lonbounds[1]))

NAON = spatial_ave(ps[:,midlati:NAOmaxlati,NAOminloni:NAOmaxloni], lats[midlati:NAOmaxlati])
NAOS = spatial_ave(ps[:,NAOminlati:midlati,NAOminloni:NAOmaxloni], lats[NAOminlati:midlati])

NAO = NAON - NAOS

scaler = StandardScaler()
NAO = scaler.fit_transform(NAO.reshape(-1,1))
NAO = np.squeeze(NAO)


 
CTIminlati = np.argmin(np.abs(lats - (-6)))
CTImaxlati = np.argmin(np.abs(lats - 6))
CTIminloni = np.argmin(np.abs(lons - 0))
CTImaxloni = np.argmin(np.abs(lons - 90))
 
CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])

##coarse grid lat/lon spacing
#cstep=2
#lats = np.arange(minlat,maxlat+cstep,cstep)
#lons = np.arange(minlon,maxlon+cstep,cstep)
#
#
#cgrid = cdms2.createGenericGrid(lats,lons)
##regridfunc = Regridder(ingrid, cgrid)
#sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#u = u.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#v = v.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#ps = ps.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

print 'getting subregion...'
sst = sst.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
field = field.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
u = u.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
v = v.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
ps = ps.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))


lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
nlat = len(lats)
nlon = len(lons)
nt = sst.shape[0]
#interpolate to coarser grid to speed up 
#sst= sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

print 'regressing out CTI...'



sst_MM = regressout_x(sst, CTI)
#field = regressout_x(field, CTI)
#u = regressout_x(u, CTI)
#v = regressout_x(v, CTI)
#ps = regressout_x(ps, CTI)

MMlatbounds = [-30,30]
MMlonbounds = [290,360]

#MMlatboundmini = np.argmin(np.abs(MMlatbounds[0]-lats))
#eqi = np.argmin(np.abs(0 - lats))
#MMlatboundmaxi = np.argmin(np.abs(MMlatbounds[1]-lats))
#MMlonboundmini = np.argmin(np.abs(MMlonbounds[0]-lons))
#MMlonboundmaxi = np.argmin(np.abs(MMlonbounds[1]-lons))
#
#sst_MMNH = sst_MM[:,eqi:MMlatboundmaxi,MMlonboundmini:MMlonboundmaxi]
#sst_MMSH = sst_MM[:,MMlatboundmini:eqi,MMlonboundmini:MMlonboundmaxi]
#
#MMlatsNH = lats[eqi:MMlatboundmaxi]
#MMlatsSH = lats[MMlatboundmini:eqi]
#
#MM = spatial_ave(sst_MMNH, MMlatsNH) - spatial_ave(sst_MMSH, MMlatsSH)



#sstcorrs[i,:] = np.squeeze(clf.coef_)
 
#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5*12 + 1
ci = (N_map-1)/2
ltlag = 5
stlag = 1

#lagmax=3
lagmax = 120
lagstep = 24
lags = np.arange(-lagmax,lagmax+2*lagstep, lagstep)


#lags = np.arange(-3,0,6,3)

#sst_globe = spatial_ave(sst, lats)
field_globe = spatial_ave(field, lats)

#subtract global annual mean to isolate local processes
#sstprime = sst.T - sst_globe
#sstprime = sstprime.T
sstprime = sst


#fieldprime = field.T - field_globe
#fieldprime = fieldprime.T
fieldprime = field

uprime = u
vprime = v
psprime = ps


print 'computing long-term/short-term fields...'
field_lt = running_mean(fieldprime, N_map)
field_st =  fieldprime[ci:-ci,:] - field_lt

sst_lt = running_mean(sstprime, N_map)
sst_st = sstprime[ci:-ci,:] - sst_lt

u_lt = running_mean(uprime, N_map)
u_st = uprime[ci:-ci,:] - u_lt

v_lt = running_mean(vprime, N_map)
v_st = vprime[ci:-ci,:] - v_lt

ps_lt = running_mean(psprime, N_map)
ps_st = psprime[ci:-ci,:] - ps_lt




field_lt = np.ma.masked_array(field_lt, mask=np.abs(field_lt>1e3))
field_st = np.ma.masked_array(field_st, mask=np.abs(field_st>1e3))
sst_lt = np.ma.masked_array(sst_lt, mask=np.abs(sst_lt>1e3))
sst_st = np.ma.masked_array(sst_st, mask=np.abs(sst_st>1e3))


MM_lt = running_mean(MM, N_map)
MM_st = MM[ci:-ci] - MM_lt

MMwind_lt = running_mean(MMwind, N_map)
MMwind_st = MMwind[ci:-ci] - MMwind_lt

nt = field.shape[0]
nt_lt = sst_lt.shape[0]


#lats = lats[NAminlati:NAmaxlati]
#lons = lons[NAminloni:NAmaxloni]
#
#nlat = len(lats)
#nlon = len(lons)


scaler = StandardScaler()
MMstd = scaler.fit_transform(MM.reshape(-1,1))
MMstd_lt = scaler.fit_transform(MM_lt.reshape(-1,1))
MMstd_st = scaler.fit_transform(MM_st.reshape(-1,1))


MMwindstd = scaler.fit_transform(MMwind.reshape(-1,1))
MMwindstd_lt = scaler.fit_transform(MMwind_lt.reshape(-1,1))
MMwindstd_st = scaler.fit_transform(MMwind_st.reshape(-1,1))

sstcorrs = MV.zeros((nlat,nlon))
sstcorrs_lt = MV.zeros((nlat, nlon))
sstcorrs_st = MV.zeros((nlat, nlon))
fieldcorrs = MV.zeros((nlat, nlon))
fieldcorrs_lt = MV.zeros((nlat,nlon))
fieldcorrs_st = MV.zeros((nlat,nlon))

pscorrs = MV.zeros((nlat, nlon))
pscorrs_lt = MV.zeros((nlat,nlon))
pscorrs_st = MV.zeros((nlat,nlon))
ucorrs = MV.zeros((nlat,nlon))
ucorrs_lt = MV.zeros((nlat,nlon))
ucorrs_st = MV.zeros((nlat,nlon))
vcorrs = MV.zeros((nlat,nlon))
vcorrs_lt = MV.zeros((nlat,nlon))
vcorrs_st = MV.zeros((nlat,nlon))

NAOcorrs = MV.zeros((nlat,nlon))

CTIsstcorrs = MV.zeros((nlat, nlon))

CTIstd = scaler.fit_transform(CTI.reshape(-1,1))


#compute correlation between long-term/short-term MM and 2D field
print r'calculating correlations between MM and {:s}...'.format(ftitle)
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
        
        
        clf = linear_model.LinearRegression()
        
        
#        clf.fit(CTIstd.reshape(-1,1), sstprime_g)
#        CTIsstcorrs[i,:] = np.squeeze(clf.coef_)
        
        clf.fit(NAO.reshape(-1,1), psprime_g)
        NAOcorrs[i,:] = np.squeeze(clf.coef_)
        
    

        clf.fit(MMstd.reshape(-1,1), sstprime_g)
        sstcorrs[i,:] = np.squeeze(clf.coef_)
        
        #clf = linear_model.LinearRegression()
        clf.fit(MMstd_lt.reshape(-1,1), sst_lt_g)
        sstcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMstd_st.reshape(-1,1), sst_st_g)
        sstcorrs_st[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd.reshape(-1,1), uprime_g)
        ucorrs[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd_lt.reshape(-1,1), u_lt_g)
        ucorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd_st.reshape(-1,1), u_st_g)
        ucorrs_st[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd.reshape(-1,1), vprime_g)
        vcorrs[i,:] = np.squeeze(clf.coef_)
         
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd_lt.reshape(-1,1), v_lt_g)
        vcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd_st.reshape(-1,1), v_st_g)
        vcorrs_st[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd.reshape(-1,1), psprime_g)
        pscorrs[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd_lt.reshape(-1,1), ps_lt_g)
        pscorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMwindstd_st.reshape(-1,1), ps_st_g)
        pscorrs_st[i,:] = np.squeeze(clf.coef_)
        
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMstd.reshape(-1,1), fieldprime_g)
        fieldcorrs[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMstd_lt.reshape(-1,1), field_lt_g)
        fieldcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
        #clf = linear_model.LinearRegression()
        clf.fit(MMstd_st.reshape(-1,1), field_st_g)
        fieldcorrs_st[i,:] = np.squeeze(clf.coef_)
       
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


      
#Plot maps of SST and THF patterns associated with MM
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
    fieldmin=-5
    fieldmax=5
    fieldstep = 0.05
    cbstep = 1.0
elif fsave == 'sst':
    fieldmin=-0.5
    fieldmax=0.5
    fieldstep = 0.02
    cbstep = 0.25
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
    
    
psmax=0.5
psmin = -0.5
psstep = 0.1
pscbstep = 0.1
sststep = 0.02
thfmin=-3
thfmax=3
sstmin = -0.5
sstmax = 0.5
thfstep =0.02
thfcbstep=1.0
sstcbstep = 0.25
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
    sstminlag = -0.5
    sstmaxlag = 0.5
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
    
psticks = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag),2)
psticklbls = np.round(np.arange(psminlag,psmaxlag+pscbstep,pscbsteplag), 2)
psticklbls[psticklbls == -0.00] = 0.00

sstticks = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag),2)
sstticklbls = np.round(np.arange(sstminlag,sstmaxlag+sstcbstep,sstcbsteplag), 2)
sstticklbls[sstticklbls == -0.00] = 0.00


#NAlats = lats[NAminlati:NAmaxlati]
#NEED TO AVERAGE OVER NA LONGITUDES
NAsst = spatial_ave(sst, lats)
NAsst_lt = spatial_ave(sst_lt, lats)

NAfield = spatial_ave(field, lats)
NAfield_lt = spatial_ave(field_lt, lats)

MMlatbounds = [-21, 32]
MMlonbounds = [-75, 15]

CTIlatbounds = [-6,6]
CTIlonbounds = [0, 90]

x1=CTIlonbounds[0]
y1=CTIlatbounds[0]
x2=CTIlonbounds[0]
y2=CTIlatbounds[1]
x3=CTIlonbounds[1]
y3=CTIlatbounds[1]
x4=CTIlonbounds[1]
y4=CTIlatbounds[0]



#PLOT MM INDEX AND FIELD
ci = (N_map-1)/2
fig=plt.figure(figsize=(14,22))
ax = fig.add_subplot(311)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, MM)
ax.plot(tyears[ci:-ci], MM_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'Atlantic Meridional Mode')
ax.axhline(0, color='black')
ax.set_ylabel(r'SST ($^{{\circ}}$C)')
ax.set_xlabel('time (years)')
ax.legend(loc='upper right')
ax = fig.add_subplot(312)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, MMwind)
ax.plot(tyears[ci:-ci], MMwind_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'Atlantic Meridional Mode')
ax.legend(loc='upper right')
ax.axhline(0, color='black')
ax.set_ylabel(r'Wind (m/s)')
ax.set_xlabel('time (years)')
ax = fig.add_subplot(313)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, NAfield)
ax.plot(tyears[ci:-ci], NAfield_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
ax.legend(loc='upper right')
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'{:s}'.format(ftitle))
ax.axhline(0, color='black')
ax.set_ylabel(r'{:s} ({:s})'.format(ftitle, units))
#ax.legend()
ax.set_xlabel('time (years)')
plt.savefig(fout + '{:s}_MM{:s}_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#PLOT MM INDEX AND FIELD
ci = (N_map-1)/2
fig=plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
#plt.figure()
#ax=plt.gcf().gca()
ax.plot(tyears, MMwindstd, label='Wind')
ax.plot(tyears, MMstd, label = 'SST')
#plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
ax.set_title(r'Atlantic Meridional Mode')
ax.axhline(0, color='black')
#ax.set_ylabel(r'SST ($^{{\circ}}$C)')
ax.legend()
ax.set_xlabel('time (years)')
plt.savefig(fout + '{:s}_stdMM_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

MMstd = np.squeeze(MMstd)
MMstd_lt = np.squeeze(MMstd_lt)
MMwindstd = np.squeeze(MMwindstd)
MMwindstd_lt = np.squeeze(MMwindstd_lt)

NAOMMcorr = np.corrcoef(MMstd, NAO)[0,1]
NAOMMwindcorr = np.corrcoef(MMwindstd, NAO)[0,1]

NAO_lt = running_mean(NAO, N_map)

NAOMMcorr_lt = np.corrcoef(MMstd_lt, NAO_lt)[0,1]
NAOMMwindcorr_lt = np.corrcoef(MMwindstd_lt, NAO_lt)[0,1]

NAOMM_laggedcorr = np.correlate(NAO, MMstd/len(MMstd), 'full')
lagzero = len(NAOMM_laggedcorr)/2

lagmax=12
lagstep=1
lags = np.arange(-lagmax, lagmax+lagstep, lagstep)

NAOMM_laggedcorr = NAOMM_laggedcorr[lagzero-lagmax:lagzero+lagmax+1]

fig=plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.bar(lags, NAOMM_laggedcorr, align='center')
ax.set_xlabel('NAO lag (months)')
ax.set_ylabel('Correlation with AMM')
plt.savefig(fout + '{:s}_MM_NAO_laggedcorr_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


#PLOT NAO AND MM time series
#ci = (N_map-1)/2
#fig=plt.figure(figsize=(14,22))
#ax = fig.add_subplot(311)
##plt.figure()
##ax=plt.gcf().gca()
#ax.plot(tyears, MMstd)
#ax.plot(tyears[ci:-ci], MMstd_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
##plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
##ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
##ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
#ax.set_title(r'Atlantic Meridional Mode [{:3.2f}, {:3.2f}]'.format(NAOMMcorr, NAOMMcorr_lt))
#ax.axhline(0, color='black')
#ax.set_ylabel(r'SST index')
#ax.set_xlabel('time (years)')
#ax.legend(loc='upper right')
#ax = fig.add_subplot(312)
##plt.figure()
##ax=plt.gcf().gca()
#ax.plot(tyears, MMwindstd)
#ax.plot(tyears[ci:-ci], MMwindstd_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
##plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
##ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
##ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
#ax.set_title(r'Atlantic Meridional Mode [{:3.2f}, {:3.2f}]'.format(NAOMMwindcorr, NAOMMwindcorr_lt))
#ax.legend(loc='upper right')
#ax.axhline(0, color='black')
#ax.set_ylabel(r'Wind index')
#ax.set_xlabel('time (years)')
#ax = fig.add_subplot(313)
##plt.figure()
##ax=plt.gcf().gca()
#ax.plot(tyears, NAO)
#ax.plot(tyears[ci:-ci], NAO_lt, label='{:1.0f}-yr RM'.format((N_map-1)/12.))
##plt.plot(tyears[ci:-ci], MM_st, label='short-term residual')
##ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth>0, color='red')
##ax.fill_between(tyears[ci:-ci], 0, MM_smooth, where= MM_smooth<0, color='blue')
#ax.set_title(r'NAO')
#ax.legend(loc='upper right')
#ax.axhline(0, color='black')
##ax.set_ylabel(r'Wind (m/s)')
#ax.set_xlabel('time (years)')
#plt.savefig(fout + '{:s}_MM_NAO_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()


    
ticks = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep),1)
ticklbls = np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 1)
ticklbls[ticklbls == -0.0] = 0.0
                 
#sstlevels = np.arange8, 0.8+sststep, sststep)
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

orient = 'horizontal'
if lonbounds[1] - lonbounds[0] <= 180:
    orient = 'vertical'
    
uskip=8

#plt.figure(figsize=(16,12))
#ax = plt.axes(projection=prj)
#ax.coastlines()
#ax.add_feature(cart.feature.LAND, zorder=50, edgecolor='k', facecolor='grey')
#ax.set_xticks(mer, crs=prj)
#ax.set_yticks(par, crs=prj)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#ax.xaxis.set_major_formatter(lon_formatter)
#ax.yaxis.set_major_formatter(lat_formatter)
#ax.get_yaxis().set_tick_params(direction='out')
#ax.get_xaxis().set_tick_params(direction='out')
#ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
##ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
##ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
#plot = ax.contourf(x, y, CTIsstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
#cb = plt.colorbar(plot, label=r'$^{{\circ}}$C', orientation=orient)
##qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs[::uskip,::uskip], vcorrs[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
##ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
#ax.add_patch(poly)
#plt.title(r'SST regressed on CTI')
##plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
#plt.savefig(fout + '{:s}_CTI_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
##plt.savefig(fout + '{:s}_AMO_sst_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

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
ct = ax.contour(x, y, NAOcorrs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
#plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
#cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
#qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs[::uskip,::uskip], vcorrs[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
#ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
#ax.add_patch(poly)
plt.title(r'SLP')
#plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + '{:s}_NAO_SLP_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs[::uskip,::uskip], vcorrs[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
#ax.add_patch(poly)
plt.title(r'SST, SLP and 10-m winds')
#plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + '{:s}_AMM_sstSLPuv_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
ct = ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs_lt, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_lt[::uskip,::uskip], vcorrs_lt[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
##rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
#ax.add_patch(poly)
plt.title(r'long-term SST, SLP and 10-m winds ({:1.0f}-yr RM)'.format((N_map-1)/12.))
plt.savefig(fout + '{:s}_AMM_sstSLPuv_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs_st, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_st[::uskip,::uskip], vcorrs_st[::uskip,::uskip], transform=cart.crs.PlateCarree(),  scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=1, zorder=100)
##rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
#ax.add_patch(poly)
plt.title(r'short-term SST, SLP and 10-m winds (residual)')
plt.savefig(fout + '{:s}_AMM_sstSLPuv_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
#ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
#ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
qv1 = ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs[::uskip,::uskip], vcorrs[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
##rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
#ax.add_patch(poly)
plt.title(r'SST and 10-m winds')
#plt.title(r'regression of SST on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + '{:s}_AMM_sstuv_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
#ct = ax.contour(x, y, pscorrs_lt, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
#ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs_lt, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_lt[::uskip,::uskip], vcorrs_lt[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=2,zorder=100)
##rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
#ax.add_patch(poly)
plt.title(r'long-term SST and 10-m winds ({:1.0f}-yr RM'.format((N_map-1)/12.))
plt.savefig(fout + '{:s}_AMM_sstuv_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
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
#ct = ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', linewidths=1, transform=cart.crs.PlateCarree())
#ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs_st, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())   
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
ax.quiver(x[::uskip,::uskip], y[::uskip,::uskip], ucorrs_st[::uskip,::uskip], vcorrs_st[::uskip,::uskip], transform=cart.crs.PlateCarree(), scale_units='inches', scale = 1, width=0.0015, headwidth=12, headlength=8, minshaft=2)
ax.quiverkey(qv1, 0.90, -.13, 0.5, '0.5 m/s')
#poly = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)],facecolor='none',edgecolor='red',linewidth=1, zorder=100)
##rect = patches.Rectangle((lonbounds[0], latbounds[0]),lonbounds[1]-lonbounds[0],latbounds[1]-latbounds[0],linewidth=1,edgecolor='grey',facecolor='none',zorder=100)
#ax.add_patch(poly)
plt.title(r'short-term SST and 10-m winds (residual)')
plt.savefig(fout + '{:s}_AMM_sstuv_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
    
    

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
#plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=cart.crs.PlateCarree())    
#cb = plt.colorbar(plot, orientation = orient, label='K')
#cb.set_ticks(sstticks)
#cb.set_ticklabels(sstticklbls)
#plt.title(r'regression of SST on AMM'.format(ftitle))
#plt.savefig(fout + '{:s}_AMM_sst_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()    
    

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
cb.set_ticks(ticks)
cb.set_ticklabels(ticklbls)
plt.title(r'{:s}'.format(ftitle))
plt.savefig(fout + '{:s}_AMM_{:s}_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
plot = ax.contourf(x, y, fieldcorrs_lt, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
cb = plt.colorbar(plot, orientation = orient, label=r'{:s}'.format(units))
cb.set_ticks(ticks)
cb.set_ticklabels(ticklbls)
plt.title(r'long-term {:s} ({:1.0f}-month RM)'.format(ftitle, N_map-1))
plt.savefig(fout + '{:s}_AMM_{:s}_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
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
plt.title(r'short-term {:s} (residual)'.format(ftitle))
plt.savefig(fout + '{:s}_AMM_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
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
#ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]))
##ax.contour(x, y, pscorrs_st, levels=SLPlevels, colors='k', inline=1, linewidths=1)
#plot = ax.contourf(x, y, fieldcorrs_st, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both', transform=cart.crs.PlateCarree())
#cb = plt.colorbar(plot, orientation = 'horizontal', label=r'{:s}'.format(units))
#cb.set_ticks(ticks)
#cb.set_ticklabels(ticklbls)
#plt.title(r'regression of short-term {:s} on SST ({:1.0f}-yr RM residual)'.format(ftitle, N_map))
#plt.savefig(fout + '{:s}_localSST_{:s}_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()

#i = np.where(windows>N_map)[0][0]-1
#    
#    
#fieldcorrs_zonalave = np.ma.average(fieldcorrs[:,NAminloni:NAmaxloni], axis=-1)
#fig = plt.figure(figsize=(12,8))
#ax = fig.gca()
#ax.plot(fieldcorrs_zonalave, lats)
#ax.axvline(0, color='k')
#ax.get_yaxis().set_tick_params(direction='out')
#ax.get_xaxis().set_tick_params(direction='out')
#ax.set_xlabel(r'{:s}'.format(units))
#ax.set_ylabel(r'latitude ($^{\circ}$)')
#ax.set_ylim(0, 60)
#if fsave == 'thf' or fsave == 'Qnetsurf':
#    ax.set_xlim(-5.5,5.5)
##ax.set_ylim(50,1000)
##ax.invert_yaxis()
##cb = plt.colorbar(plot, label=r'{:s}'.format(units))
##cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
##cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
#plt.title(r'regression of {:s} on SST'.format(ftitle))
#plt.savefig(fout + '{:s}_localSST_{:s}_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close()
#    
#
#lonbounds = [280,359.99]
#latbounds = [0,60]
    
#NAminlati = np.where(lats >= latbounds[0])[0][0]
#NAmaxlati = np.where(lats >= latbounds[1])[0][0]
#NAminloni = np.where(lons >= lonbounds[0])[0][0]
#NAmaxloni = np.where(lons >= lonbounds[1])[0][0]
#
#NAlats = lats[NAminlati:NAmaxlati]
#
#    
#if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
#    fieldmin=-5
#    fieldmax=5
#    fieldstep = 0.05
#    cbstep = 1.0
#elif fsave == 'sst':
#    fieldmin=-0.3
#    fieldmax=0.3
#    fieldstep = 0.02
#    cbstep = 0.1
#elif fsave == 'umag':
#    fieldmin=-0.3
#    fieldmax=0.3
#    fieldstep=0.01
#    cbstep = 0.1
#elif fsave == 'RH10m':
#    fieldmin = -2
#    fieldmax = 2
#    fieldstep = 0.01
#    cbstep = 0.5
#elif fsave == 'cE':
#    fieldmin = -0.5
#    fieldmax = 0.5
#    fieldstep = 0.01
#    cbstep = 0.1
#elif fsave == 'cD':
#    fieldmin = -0.5
#    fieldmax = 0.5
#    fieldstep = 0.01
#    cbstep = 0.1
#else:
#    fieldmin=-3
#    fieldmax=3
#    fieldstep =0.1
#    cbstep=1.0
#    
#if corr:
#    fieldminlag = -1.0 
#    fieldmaxlag = 1.0
#    cbsteplag = 0.2
#    fieldunitslag = ''
#
#else:
#    fieldminlag = fieldmin
#    fieldmaxlag = fieldmax
#    cbsteplag = cbstep 
#    fieldunitslag = units
#    
#ticks = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag),2)
#ticklbls = np.round(np.arange(fieldminlag,fieldmaxlag+cbstep,cbsteplag), 2)
#ticklbls[ticklbls == -0.00] = 0.00
#                 
#fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)
#
#weights = np.cos(np.deg2rad(lats))
#
#lagg, latt = np.meshgrid(lags-lagoffset, NAlats-latoffset)
#
##Plot zonally-averaged lagged correlation between SST and THF
#fig=plt.figure(figsize=(12,6))
##plt.suptitle('MM ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
#ax = fig.add_subplot(111)
#h = ax.pcolor(lagg, latt, fieldlagcorrs_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#ax.axvline(0, color='k')
#if corr:
#    ax.set_title('correlation of {:s} with SST'.format(ftitle))
#else:
#    ax.set_title('regression of {:s} on SST'.format(ftitle))
#ax.set_xlabel(r'{:s} lag (months)'.format(ftitle))
#ax.set_ylabel('latitude (degrees)')
#ax.set_ylim(NAlats[0],NAlats[-1])
#ax.set_xticks(laglabels)
#ax.set_xticklabels(laglabels)
#if len(latboundar) > 1:
#    ax.axhline(20, color='grey', linewidth=1)
#    ax.axhline(45, color='grey', linewidth=1)
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
#ax = fig.add_subplot(132)
#ax.pcolor(lagg, latt, fieldlagcorrs_lt_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#ax.axvline(0, color='k')
#if corr:
#    ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
#else:
#    ax.set_title('long-term regression ({:1.0f}-yr RM)'.format(N_map))
#ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
#ax.set_ylim(NAlats[0],NAlats[-1])
#ax.set_xticks(laglabels)
#ax.set_xticklabels(laglabels)
#if len(latboundar) > 1:
#    ax.axhline(latboundar[0][1], color='grey', linewidth=1)
#    ax.axhline(latboundar[1][1], color='grey', linewidth=1)
#ax = fig.add_subplot(133)
#h = ax.pcolor(lagg, latt, fieldlagcorrs_st_zonalave.T, vmin=fieldminlag, vmax=fieldmaxlag, cmap=plt.cm.RdBu_r)
#ax.axvline(0, color='k')
#if corr:
#    ax.set_title('short-term correlation')
#else:
#    ax.set_title('short-term regression')
#ax.set_xlabel(r'{:s} lag (years)'.format(ftitle))
#ax.set_ylim(NAlats[0],NAlats[-1])
#ax.set_xticks(laglabels)
#ax.set_xticklabels(laglabels)
#if len(latboundar) > 1:
#    ax.axhline(latboundar[0][1], color='grey', linewidth=1)
#    ax.axhline(latboundar[1][1], color='grey', linewidth=1)
#cb = fig.colorbar(h, ax=ax, orientation="vertical", label=r'{:s}'.format(fieldunitslag))
#cb.set_ticks(ticks)
#cb.set_ticklabels(ticklbls)
#if corr:
#    if len(latboundar) > 1:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_zonalavelocal_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#    else:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#else:
#    if len(latboundar) > 1:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_zonalavelocal_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#    else:
#        plt.savefig(fout + '{:s}_monthlylocalSST_{:s}_lagregr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latbounds[0], latbounds[1], str(detr)[0]))
#plt.close() 
##
##calculate average of lagged correlation within defined latbounds
#lats = lats[NAminlati:NAmaxlati]
#weights = np.cos(np.deg2rad(lats))
#tli = np.where(lats > latboundar[0][0])[0][0]
#tui = np.where(lats > latboundar[0][1]-0.001)[0][0]
#
#tfieldlagcorrs_lt_ave = np.ma.average(fieldlagcorrs_lt_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
#tfieldlagcorrs_st_ave = np.ma.average(fieldlagcorrs_st_zonalave[:,tli:tui],axis=1,weights=weights[tli:tui])
#
#fig=plt.figure(figsize=(18,7))
#ax = fig.add_subplot(121)
#ax.plot(lags, tfieldlagcorrs_lt_ave)
#ax.axhline(0, color='black')
#ax.axvline(0, color='black')
#ax.set_ylim(fieldminlag,fieldmaxlag)
#ax.set_ylabel(r'{:s}'.format(fieldunitslag))
#if corr:
#    ax.set_title('long-term correlation of {:s} with SST ({:1.0f}-yr RM)'.format(ftitle, N_map))
#else:
#    ax.set_title('long-term regression of {:s} on SST ({:1.0f}-yr RM)'.format(ftitle, N_map))
#ax.set_xlabel('{:s} lag (years)'.format(ftitle))
#ax = fig.add_subplot(122)
#ax.plot(lags, tfieldlagcorrs_st_ave)
#ax.set_ylim(fieldminlag,fieldmaxlag)
#ax.set_ylabel(r'{:s}'.format(fieldunitslag))
#ax.axhline(0, color='black')
#ax.axvline(0, color='black')
#if corr:
#    ax.set_title('short-term correlation')
#else:
#    ax.set_title('short-term regression')
#ax.set_xlabel('{:s} lag (years)'.format(ftitle))
#if corr:
#    plt.savefig(fout + '{:s}_localSST_{:s}_lagcorr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
#else:
#    plt.savefig(fout + '{:s}_localSST_{:s}_lagregr_ave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(dataname, fsave, latboundar[0][0], latboundar[0][1],  str(detr)[0]))
#plt.close() 

#













































