#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:00:06 2018

@author: cpatrizio
"""

import sys
sys.path.append("/Users/cpatrizio/repos/")
import cdms2 as cdms2
#from netCDF4 import Dataset
import matplotlib
import MV2 as MV
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy as cart
from scipy import signal
import cdutil
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.stats.stats import pearsonr, linregress
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from ocean_atmosphere.misc_fns import an_ave, spatial_ave, running_mean, calc_NA_globeanom, calc_NA_globeanom3D, detrend, butter_lowpass_filter, regressout_x, cov2_coeff
from palettable.cubehelix import Cubehelix
import pandas as pd

cx4 = Cubehelix.make(reverse=True, start=0.3, rotation=-0.5)

fin2 = '/Users/cpatrizio/data/ECMWF/'
fin = '/Users/cpatrizio/data/MERRA2/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/NA index/'



fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fcf3D = cdms2.open(fin + 'MERRA2_cldfrac3D_monthly1980to2017.nc')
fv3D = cdms2.open(fin + 'MERRA2_v3D_monthly1980to2017.nc')
fu3D = cdms2.open(fin + 'MERRA2_u3D_monthly1980to2017.nc')
fomega = cdms2.open(fin + 'MERRA2_omega_monthly1980to2017.nc')
fqv3D = cdms2.open(fin + 'MERRA2_qv3D_monthly1980to2017.nc')
ft3D = cdms2.open(fin + 'MERRA2_t3D_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#cffile = cdms2.open(fin + 'MERRA2_modis_cldfrac_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')

#test ERAi
#fsst = cdms2.open(fin2 + 'sstslp.197901-201712.nc')
#fcf3D = cdms2.open(fin2 + 'cc.197901-201512.nc')
#fv3D = cdms2.open(fin2 + 'v3D.197901-201712.nc')

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

ps = fSLP['SLP']
nt_ps = ps.shape[0]

#sst = fsst('sst')
sst = fsst('TSKINWTR')
nt_sst = sst.shape[0]
#sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:,:]

cf = fcf3D['CLOUD']

qv3D = fqv3D['QV'][:]
#qv3D = qv3D.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#qv3D = qv3D[tskip:,:]


T3D = ft3D['T'][:]
#t3D = t3D.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#t3D = t3D[tskip:,:]




#p4D = np.zeros((nplev, nlat, nlon)).T
#p4D[:,:,:] = p*1e2
#p4D = p4D.T
#p4D = np.repeat(p4D[np.newaxis,...], nt, axis=0)

#RH = qv3D/r_star(p4D, t3D)


#v3D = fv3D['V']
#u3D = fu3D['U']


#omega = fomega['OMEGA']



#ctfield = ctfield*(3600/1e2)

#lhf = fsst('EFLUXWTR')
#shf = fsst('HFLUXWTR')
#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)
#thf = lhf + shf

#ps = fSLP('SLP')
#ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
#ps = ps[tskip:,:]/1e2

#LW_net_surf = radfile['LWGNT']
#LW_net_surf_cs = radfile('LWGNTCLR')
#SW_net_surf = radfile['SWGNT']
#SW_net_surf_cs = radfile('SWGNTCLR')
#LW_net_TOA = radfile['LWTUP']
#lwnettoa_cs = radfile('LWTUPCLR')
#swnettoa = radfile('SWTNT')
#swnettoa_cs = radfile('SWTNTCLR')

#Q_net_surf_cs = LW_net_surf_cs + SW_net_surf_cs

#Q_net_surf = LW_net_surf + SW_net_surf

#CRE_surf = Q_net_surf - Q_net_surf_cs

##### EDIT FIELD FOR CORRELATION WITH index
#field = Q_net_surf
#ftitle = r'Q$_{net,surf}$'
#fsave = 'Qnetsurf'

#field = CRE_surf
#ftitle = r'CRE$_{surf}$'
#fsave = 'CREsurf'
#units = r'W m$^{-2}$'

#field = thf
#ftitle = r'THF'
#fsave = 'thf'
#units = r'W m$^{-2}$'


#field = Q_net_surf
#ftitle = r'Q$_{net,surf}$'
#fsave = 'Qnetsurf'

#cf = cffile['MDSCLDFRCTTL']
#cf = cf[tskip:,:]

#field = cf*100.
#ftitle = r'$f_{total}$'
#fsave = 'ftotal'
#units = '%'

#field = cf*100.
#ftitle = r'$f$'
#fsave = 'cldfrac3D'
#units = '%'

field1 = qv3D*1e3
ftitle1 = r'$q_v$'
fsave1 = 'qv3D'
units1 = 'g/kg'

#field = RH*100
#ftitle = 'RH'
#fsave = 'RH'
#units = '%'

#field1 = T3D
#ftitle1 = '$T$'
#fsave1 = 't3D'
#units1 = 'K'

#ctfield = u3D
#ftitle2 = r'$u$'
#fsave2 = 'u3D'
#units2 = 'm/s'

#ctfield = v3D
#ftitle2 = r'$\Psi$'
#fsave2 = 'streamfn'
#units2 = r'10$^9$ kg s$^{-1}$'

#field = -omega*(3600/1e2)
#ftitle = r'$-\omega$'
#fsave = 'omega'
#units = 'hPa/day'

#nt_sst=sst.shape[0]

#fields = [field1]
#ftitles = [ftitle1]
#fsaves = [fsave1]
#unitslist = [units1]

p = field.getLevel()[:]
nplev= len(p)


#EDIT THIS FOR BOUNDS
lonbounds = [280.,360.]
latbounds = [-20, 70.]

#True for detrending data, False for raw data
detr=True
rENSO=True
lterm=True

field = field.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
field = field[tskip:nt_sst,:]


lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
lons[0] = 0
nlat = len(lats)
nlon = len(lons)
nplev = field.shape[1]

#p = field.getLevel()[:]
#nplev = len(p)
#lats = field.getLatitude()[:]
#lons = field.getLongitude()[:]
nt = field.shape[0]
#nlat = len(lats)
#nlon = len(lons)

fieldsave = field

#grid = cdms2.createGenericGrid(lats,lons)

#coarse grid lat/lon spacing
cstep=2
lats = np.arange(minlat,maxlat+cstep,cstep)
lons = np.arange(minlon,maxlon+cstep,cstep)

cgrid = cdms2.createGenericGrid(lats,lons)
#regridfunc = Regridder(ingrid, cgrid)
#sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#u = u.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#v = v.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
#ps = ps.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

#horizontally interpolate SST to coarser 3D field grid 
sst = sst.regrid(cgrid, regridTool="esmf", regridMethod = "linear")
field = field.regrid(cgrid, regridTool="esmf", regridMethod = "linear")

#mask land from 3D fields
sst_mask = np.ma.getmaskarray(sst)
field_mask = np.ma.getmaskarray(field)
sst_mask = np.repeat(sst_mask[:,np.newaxis,...],nplev,axis=1)
field_mask = np.ma.mask_or(sst_mask, field_mask)
#field_mask = np.ma.mask_or(sst_mask[:field.shape[0],:,:], field_mask)
field = np.ma.array(field, mask=field_mask)

t = sst.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
t = 1980 + t/12.

tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))


nt = sst.shape[0]

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst = detrend(sst)
 #ps_an, params = detrend_separate(ps_an)
 field = detrend(field)


lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
 


t = field.getTime().asRelativeTime("months since 1980")
t = np.array([x.value for x in t])
tyears = 1980 + t/12.



#tyears = np.arange(np.ceil(t[0]), np.round(t[-1]))

#subtract seasonal cycle from fields
cdutil.setTimeBoundsMonthly(sst)
cdutil.setTimeBoundsMonthly(field)

print 'subtracting seasonal cycle...'
field = cdutil.ANNUALCYCLE.departures(field)
sst = cdutil.ANNUALCYCLE.departures(sst)

#field = field.reshape(nt, nplev, nlat, nlon)
#ctfield = ctfield.reshape(nt, nplev, nlat, nlon)
#sst = sst.reshape(nt, nplev, nlat, nlon)

CTIminlati = np.argmin(np.abs(lats - (-6)))
CTImaxlati = np.argmin(np.abs(lats - 6))
CTIminloni = np.argmin(np.abs(lons - 0))
CTImaxloni = np.argmin(np.abs(lons - 90))
 
#CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])

# CTI Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 3.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)
 
CTI = spatial_ave(sst[:,CTIminlati:CTImaxlati,CTIminloni:CTImaxloni], lats[CTIminlati:CTImaxlati])


CTI = butter_lowpass_filter(CTI, cutoff, fs, order)


print 'getting subregion...'
sst = sst.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))
field = field.subRegion(latitude=(latbounds[0], latbounds[1]), longitude=(lonbounds[0], lonbounds[1]))


lats = field.getLatitude()[:]
lons = field.getLongitude()[:]
nlat = len(lats)
nlon = len(lons)
nt = sst.shape[0]

latdiff = np.diff(lats)[0]
londiff = np.diff(lons)[0]

print 'regressing out CTI...'

if rENSO:
    CTIlag=2
    sst = regressout_x(sst[CTIlag:,...], CTI[:-CTIlag])
    #field = regressout_x(field[CTIlag:,...], CTI[:-CTIlag])
    #ctfield = regressout_x(ctfield[CTIlag:,...], CTI[:-CTIlag])
    field = field[CTIlag:,...]
    #u = u[CTIlag:,...]
    #v = v[CTIlag:,...]
    #ps = ps[CTIlag:,...]
    tyears = tyears[CTIlag:,...]
    #field = np.ma.masked_array(field, mask=np.abs(field) > 1e4)
    #ctfield = np.ma.masked_array(ctfield, mask=np.abs(field) > 1e4)

#sst_globe = spatial_ave(sst, lats)
#field_globe = spatial_ave(field, lats)

#subtract global annual mean to isolate local processes
#sstprime = sst.T - sst_globe
#sstprime = sstprime.T
sstprime = sst


#fieldprime = field.T - field_globe
#fieldprime = fieldprime.T
fieldprime = field

nt = sst.shape[0]



# CTI Filter requirements.
order = 5
fs = 1     # sample rate, (cycles per month)
Tn = 12*7.
cutoff = 1/Tn  # desired cutoff frequency of the filter (cycles per month)

if lterm: 

    print 'computing long-term/short-term fields...'
    # apply the filter
    sst_lt = butter_lowpass_filter(sstprime, cutoff, fs, order)
    
    #INTERPOLATION MIGHT BE IMPORTANT?
    #need to fill missing field data by interpolation... THF blows up otherwise 
    #sst_df = pd.DataFrame(sst.reshape(nt, nlat*nlon))
    field_df = pd.DataFrame(field.reshape(nt, nplev*nlat*nlon))
    #sst_df = sst_df.interpolate()
    field_df = field_df.interpolate()

    #sstprime = sst_df.values.reshape(nt, nlat, nlon)
    fieldprime = field_df.values.reshape(nt, nplev, nlat, nlon)

    
    
    #THIS SEEMS TO WORK
    field_lt = butter_lowpass_filter(fieldprime, cutoff, fs, order)


    fieldprime = np.ma.masked_array(fieldprime, mask=~np.isfinite(fieldprime))
    field_lt = np.ma.masked_array(field_lt, mask=~np.isfinite(field_lt))
    field_lt = np.ma.masked_array(field_lt, mask=np.abs(field_lt)>1e4)


    
    field_st =  fieldprime - field_lt
    sst_st = sstprime - sst_lt
    
    nt_lt = sst_lt.shape[0]



slati=5
nlati=20
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


scaler = StandardScaler()
indexstd = scaler.fit_transform(index.reshape(-1,1))
indexstd_lt = scaler.fit_transform(index_lt.reshape(-1,1))
indexstd_st = scaler.fit_transform(index_st.reshape(-1,1))

CTIstd = scaler.fit_transform(CTI.reshape(-1,1))


sstcorrs = MV.zeros((nlat,nlon))
sstcorrs_lt = MV.zeros((nlat,nlon))
sstcorrs_st = MV.zeros((nlat,nlon))
fieldcorrs = MV.zeros((nplev, nlat*nlon))
fieldcorrs_lt = MV.zeros((nplev, nlat*nlon))
fieldcorrs_st = MV.zeros((nplev, nlat*nlon))


#calculate SST pattern of index without looping
#sstprime_g = sstprime.reshape(nt, nlat*nlon)
#clf = linear_model.LinearRegression()
#clf.fit(indexstd.reshape(-1,1), sstprime_g)
#sstcorrs = clf.coef_.reshape(nlat, nlon)

fieldprime_temp = fieldprime.reshape(nt, nplev, nlat*nlon)
if lterm:
    field_lt_temp = field_lt.reshape(nt_lt, nplev, nlat*nlon)
    field_st_temp = field_st.reshape(nt_lt, nplev, nlat*nlon)
    indexstd_ltrep = np.squeeze(np.repeat(indexstd_lt[:,np.newaxis], nlon*nlat, axis=1))
    indexstd_strep = np.squeeze(np.repeat(indexstd_st[:,np.newaxis], nlon*nlat, axis=1))



indexstdrep = np.squeeze(np.repeat(indexstd[:,np.newaxis], nlon*nlat, axis=1))





#compute correlation between long-term/short-term index and 3D field
print r'calculating correlations between index and {:s}...'.format(ftitle)
for i in range(nplev):         

     print 'pressure', p[i]
       
     #sstprime_g = sstprime[:,i,:]
     fieldprime_g = fieldprime_temp[:,i,:]
    


     coefs = np.diag(cov2_coeff(indexstdrep.T, fieldprime_g.T))
     fieldcorrs[i,:] = coefs

     
     
     if lterm:
         
              
         field_lt_g = field_lt_temp[:,i,:]
         field_st_g = field_st_temp[:,i,:]

         coefs_lt = np.diag(cov2_coeff(indexstd_ltrep.T, field_lt_g.T))
         coefs_st = np.diag(cov2_coeff(indexstd_strep.T, field_st_g.T))

         
         fieldcorrs_lt[i,:] = coefs_lt
         fieldcorrs_st[i,:] = coefs_st

            
         

    
     #clf = linear_model.LinearRegression()
     #clf.fit(indexstd.reshape(-1,1), sstprime_g)
     #sstcorrs[i,:] = np.squeeze(clf.coef_)
     
#     clf = linear_model.LinearRegression()
#     clf.fit(indexstd.reshape(-1,1), fieldprime_g)
#     fieldcorrs[i,:] = np.squeeze(clf.coef_)
     
#     clf = linear_model.LinearRegression()
#     clf.fit(indexstd.reshape(-1,1), ctfieldprime_g)
#     ctfieldcorrs[i,:] = np.squeeze(clf.coef_)
#     


            
     #ctfieldcorrs[i,:] = ctcoefs
     #fieldcorrs_lt[i,:] = coefs_lt
     #fieldcorrs_st[i,:] = coefs_st
            
     

     

fieldcorrs = fieldcorrs.reshape(nplev, nlat, nlon)
fieldcorrs_lt = fieldcorrs_lt.reshape(nplev, nlat, nlon)
fieldcorrs_st = fieldcorrs_st.reshape(nplev, nlat, nlon)
         
    
        
    
#Plot horizontally averaged NA 3D field vs time.
 
tt, pp = np.meshgrid(tyears, p)


cflevels = np.arange(-3.0, 3.0+0.25, 0.1)
cfanomlevels = np.arange(-1.5, 1.5+0.05, 0.05)
qvlevels = np.arange(-1.0,1.0+0.05,0.05)
qvanomlevels = np.arange(-0.5,0.5+0.01,0.01)
RHlevels = np.arange(0, 105,5)
RHanomlevels = np.arange(-5,5.1,0.1)
omegalevels = np.arange(-1,1,0.01)
omegaanomlevels = np.arange(-0.5,0.5,0.005)
#ctfieldlevels 
if fsave == 'cldfrac3D':
    fieldlevels = cflevels
    anomlevels = cfanomlevels
elif fsave == 'qv3D':
    fieldlevels=qvlevels
    anomlevels=qvanomlevels
elif fsave == 'RH':
    fieldlevels = RHlevels
    anomlevels = RHanomlevels
elif fsave == 't3D':
    fieldlevels = qvlevels
    anomlevels = qvanomlevels
elif fsave == 'omega':
    fieldlevels = omegalevels
    anomlevels = omegaanomlevels
    
    
else:
    fieldlevels = 30
    anomlevels = 30

    

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
elif fsave == 'cldfrac3D':
    fieldmin = -1.0
    fieldmax = 1.0
    fieldstep = 0.02
    cbstep = 0.25
elif fsave == 'qv3D':
    fieldmin= -0.3
    fieldmax= 0.3
    fieldstep= 0.01
    cbstep = 0.1
elif fsave == 'RH':
    fieldmin= -2
    fieldmax= 2
    fieldstep= 0.1
    cbstep = 1
elif fsave == 't3D':
    fieldmin = -0.8
    fieldmax = 0.8
    fieldstep=0.02
    cbstep = 0.2
    if detr: 
        fieldmin = -0.8
        fieldmax = 0.8
        fieldstep=0.005
        cbstep = 0.2
elif fsave == 'omega':
    fieldmin = -0.3
    fieldmax = 0.3
    fieldstep = 0.01
    cbstep = 0.1
    
else:
    fieldmin=-10
    fieldmax=10
    fieldstep =0.2
    cbstep=2.5

sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)


#Plot zonal averages of 3D fieldcorrs, fieldcorrs_lt, and fieldcorrs_st. plot as contourf (p, latitude)

sstcorrs_zonalave = np.ma.average(sstcorrs, axis=-1)
#ctfieldcorrs_zonalave = np.ma.average(ctfieldcorrs[:,NAminloni:NAmaxloni], axis=-1)

fieldcorrs_zonalave = np.ma.average(fieldcorrs, axis=-1)
fieldcorrs_lt_zonalave = np.ma.average(fieldcorrs_lt, axis=-1)
fieldcorrs_st_zonalave = np.ma.average(fieldcorrs_st, axis=-1)

    
#we can calculate the stream function here.
#integrate the zonal mean meridional velocity from 0 to p
#multiple by 2pi*radius of earth*cos(latitude)/g.


x, y = np.meshgrid(lats, p)





    
fig = plt.figure(figsize=(12,8))
ax = fig.gca()
#ct = ax.contour(xct, yct, ctfieldcorrs_zonalave, levels=ctfieldlevels, colors='k', linewidths=1)
#ct.collections[np.where(ct.levels==0)[0][0]].set_linewidth(2)
#ax.clabel(ct, ct.levels[ct.levels != 0], fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, fieldcorrs_zonalave, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both')
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.set_ylabel('Pressure (hPa)')
ax.set_xlabel(r'Latitude ($^{\circ}$)')
ax.set_xlim(-15, 65)
ax.set_ylim(50,1000)
ax.invert_yaxis()
cb = plt.colorbar(plot, label=r'{:s}'.format(units))
cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
cb.set_ticklabels(np.round(np.arange(fieldmin,fieldmax+cbstep,cbstep), 2))
plt.title(r'{:s}'.format(ftitle))
#plt.title(r'regression of {:s} and {:s} on index ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(ftitle, ftitle2, latbounds[0], latbounds[1]))
plt.savefig(fout + 'MERRA2_SST{:2.0f}Nto{:2.0f}N_{:s}_corr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()


if lterm:
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()

    #ax.clabel(ct, ct.levels[ct.levels != 0], fontsize=9, inline=1, fmt='%1.1f')
    plot = ax.contourf(x, y, fieldcorrs_lt_zonalave, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel(r'Latitude ($^{\circ}$)')
    ax.set_xlim(-15, 60)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'Long-term {:s}'.format(ftitle))
    plt.savefig(fout + 'MERRA2_SST{:2.0f}Nto{:2.0f}N_{:s}_ltcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca()
    #ax.clabel(ct, ct.levels[ct.levels != 0], fontsize=9, inline=1, fmt='%1.1f')
    plot = ax.contourf(x, y, fieldcorrs_st_zonalave, cmap=plt.cm.RdBu_r, levels=fieldlevels, extend='both')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_ylabel('pressure (hPa)')
    ax.set_xlabel(r'latitude ($^{\circ}$)')
    ax.set_xlim(-15, 60)
    ax.set_ylim(50,1000)
    ax.invert_yaxis()
    cb = plt.colorbar(plot, label=r'{:s}'.format(units))
    cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
    plt.title(r'Short-term {:s}'.format(ftitle))
    plt.savefig(fout + 'MERRA2_SST{:2.0f}Nto{:2.0f}N_{:s}_stcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
    plt.close()
    
#SAVE SSTCORRS, FIELDCORRS, FIELDCORRS_LT, AND FIELDCORRD_ST

#np.ma.set_fill_value(sstcorrs, np.nan)
#sstcorrs_save = sstcorrs.getValue()

#np.ma.set_fill_value(sstcorrs_lt, np.nan)
#sstcorrs_ltsave = sstcorrs_lt.getValue()

#np.ma.set_fill_value(sstcorrs_st, np.nan)
#sstcorrs_stsave = sstcorrs_st.getValue()
    
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


#np.save(fin + 'deltaSST_SST{0}to{1}_detr{2}_{3}x{4}'.format(np.round(lats[si],0), np.round(lats[ni],0), str(detr)[0], latdiff, londiff), sstcorrs)
#if lterm:
#    np.save(fin + 'deltaSST_SST{0}to{1}_{2}LP_detr{3}_{4}x{5}'.format(np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), sstcorrs_lt)
#    np.save(fin + 'deltaSST_SST{0}to{1}_{2}HP_detr{3}_{4}x{5}'.format(np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), sstcorrs_st)
#if fsave == 'qv3D':
#    #convert back to kg/kg
#    fieldcorrs = fieldcorrs*1e-3
#    fieldcorrs_lt = fieldcorrs_lt*1e-3
#    fieldcorrs_st = fieldcorrs_st*1e-3
np.save(fin + 'delta{0}_SST{1}to{2}_detr{3}_{4}x{5}'.format(fsave, np.round(lats[si],0), np.round(lats[ni],0), str(detr)[0], latdiff, londiff), fieldcorrs_save)
if lterm:
    np.save(fin + 'delta{0}_SST{1}to{2}_{3}LP_detr{4}_{5}x{6}'.format(fsave, np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), fieldcorrs_ltsave)
    np.save(fin + 'delta{0}_SST{1}to{2}_{3}HP_detr{4}_{5}x{6}'.format(fsave, np.round(lats[si],0), np.round(lats[ni],0), Tn/12., str(detr)[0], latdiff, londiff), fieldcorrs_stsave)
    

#plot climatological fields
fieldsave = fieldsave.subRegion(longitude=(lonbounds[0], lonbounds[1]))
zonalmeanfield = np.ma.average(np.ma.average(fieldsave, axis=0),axis=2)

lats = fieldsave.getLatitude()[:]
p = fieldsave.getLevel()[:]

x, y = np.meshgrid(lats, p)




if fsave == 'u3D':
    ctfieldlevels = np.arange(-30,35,5)

if fsave == 'ftotal' or fsave == 'fhigh' or fsave == 'fmid' or fsave == 'flow':
    fieldmin=0
    fieldmax=30
    fieldstep = 0.5
    cbstep = 5
elif fsave == 'cldfrac3D':
    fieldmin = 0
    fieldmax = 30
    fieldstep = 0.5
    cbstep = 5
elif fsave == 'qv3D':
    fieldmin=0
    fieldmax=5
    fieldstep=0.01
    cbstep=0.1
elif fsave == 'RH':
    fieldmin= 0
    fieldmax= 100
    fieldstep= 5
    cbstep = 20
elif fsave == 't3D':
    fieldmin = 200
    fieldmax = 300
    fieldstep=5
    cbstep = 25
elif fsave == 'omega':
    fieldmin = -10
    fieldmax = 10
    fieldstep = 0.1
    cbstep = 2.5
else:
    fieldmin=-10
    fieldmax=10
    fieldstep =0.2
    cbstep=2.5

fieldlevels = np.arange(fieldmin, fieldmax+fieldstep, fieldstep)

fig = plt.figure(figsize=(12,8))
ax = fig.gca()
plot = ax.contourf(x, y, zonalmeanfield, cmap=cx4.mpl_colormap, levels=fieldlevels, extend='both')
#plot = ax.pcolor(x, y, zonalmeanfield, cmap=cx4, vmin=fieldmin, vmax=fieldmax)
ax.get_yaxis().set_tick_params(direction='out')
ax.get_xaxis().set_tick_params(direction='out')
ax.set_ylabel('Pressure (hPa)')
ax.set_xlabel(r'Latitude ($^{\circ}$)')
ax.set_xlim(-15, 60)
ax.set_ylim(50,1000)
ax.invert_yaxis()
cb = plt.colorbar(plot, label=r'{:s}'.format(units))
cb.set_ticks(np.arange(fieldmin,fieldmax+cbstep,cbstep))
cb.set_ticklabels(np.arange(fieldmin,fieldmax+cbstep,cbstep))
plt.title(r'Climatological {:s}'.format(ftitle))
plt.savefig(fout + 'MERRA2_SST{:2.0f}Nto{:2.0f}N_{:s}_MEAN_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(lats[si], lats[ni], fsave, latbounds[0], latbounds[1], str(detr)[0]))
plt.close()















































