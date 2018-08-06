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

#preliminary analysis
#1. look at maps of SST and THF
#2. look at time series of global annual mean SST and THF
#3. look at maps of SST' and THF' (annual mean anomalies)

#make global maps of grid point correlations between deltaSST'/deltat (and delta(SST'^2))/deltat, and THF'.
#1. compute annual anomalies of SST and THF (i.e. subtract global annual mean SST and THF to get SST' and THF')
#2. time-smooth SST' and THF' (as in O'Reilly et al. 2016) using varied window length (e.g. 10 years)
#3. calculate deltaSST'/deltat for deltat equal to the smoothing-window length? 
#4. correlate deltaSST'/deltat with THF' and plot map 
#5.  correlate SST' with THF' and plot map.. positive correlations indicate regions where THF' contributes to the magnitude of SST' growing in time (i.e. dSST'^2/dt > 0, a positive THF feedback)
#6. question: will the maps produced in #4 and #5 look the same? 
#7. can I see the O'Reilly 2016 result in any of these maps?
#8. are positive correlations found in regions where ocean dynamics are known to be important (e.g. boundary currents)

#LONG TERM: do complete surface energy budget analysis (i.e. including cloud radiative effects). 
#look at global atmospheric CRE to quantify component of THF that results from atmospheric CRE.

#fin = '/Users/cpatrizio/data/ECMWF/'
fin = '/Users/cpatrizio/data/MERRA2/'
fout = '/Volumes/GoogleDrive/My Drive/PhD/figures/AMO/'

#fsst = cdms2.open(fin + 'era_interim_moda_SST_1979to2010.nc')
#fTHF = cdms2.open(fin + 'era_interim_moda_THF_1979to2010.nc')

fsst =  cdms2.open(fin + 'MERRA2_SST_ocnthf_monthly1980to2017.nc')
fSLP = cdms2.open(fin + 'MERRA2_SLP_monthly1980to2017.nc')
#radfile = cdms2.open(fin + 'MERRA2_rad_monthly1980to2017.nc')
#frad = cdms2.open(fin + 'MERRA2_tsps_monthly1980to2017.nc')

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

#sst = fsst('sst')
sst = fsst('TSKINWTR')
sst = sst.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
sst = sst[tskip:,:]
#cdutil.setTimeBoundsMonthly(sst)
#lhf and shf are accumulated over 12 hours (units of J/m^2)
#lhf = fTHF('slhf')/(12*60*60)
lhf = fsst('EFLUXWTR')
lhf = lhf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
lhf = lhf[tskip:,:]
#cdutil.setTimeBoundsMonthly(lhf)
#shf = fTHF('sshf')/(12*60*60)
shf = fsst('HFLUXWTR')
shf = shf.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
shf = shf[tskip:,:]

ps = fSLP('SLP')
ps = ps.subRegion(latitude=(minlat, maxlat), longitude=(minlon, maxlon))
ps = ps[tskip:,:]/1e2

#thf is positive up in MERRA2 (energy input into atmosphere by surface fluxes)
thf = lhf + shf

#True for detrending data, False for raw data
detr=True

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

#bounds for AMO (AMOmid in O'Reilly 2016 is defined between 40 N and 60 N, Gulev. et al. 2013 defined between 30 N and 50 N)
latbounds = [0,60]

#initial/final indices for base period
baseti = 0
basetf = 10

sst_an = an_ave(sst)
thf_an = an_ave(thf)
ps_an = an_ave(ps)

#detrend annual fields instead of monthly? prevents thf from blowing up for some reason...
if detr: 
 sst_an, params = detrend_separate(sst_an)
 ps_an, params = detrend_separate(ps_an)
 thf_an, params = detrend_separate(thf_an)

AMO, sstanom_globe_an, sstanom_na_an = calc_NA_globeanom(sst_an, latbounds, lats, lons, baseti, basetf)
NAthf2, thfanom_globe_an, thfanom_na_an = calc_NA_globeanom(thf_an, latbounds, lats, lons, baseti, basetf)


 #thf blows up after detrending, need to mask values. Why doesn't this work for plotting later?
 #thf = np.ma.array(thf, mask=np.bitwise_or(thf_mask, np.abs(thf) > 1e2))

 #this seems to work for plotting, but have to mask nan values later
 #thf[np.abs(thf) > 1e3] = np.nan
     
 #sstanom_globe_an = signal.detrend(sstanom_globe_an) 
 #sstanom_na_an = signal.detrend(sstanom_na_an)
 #AMO = signal.detrend(AMO)
 
 #NAthf2 = signal.detrend(NAthf2)
 #thfanom_globe_an = signal.detrend(thfanom_globe_an)
 #thfanom_na_an = signal.detrend(thfanom_na_an)

sstcorrs = MV.zeros((nlat,nlon))
sstpvals = MV.zeros((nlat,nlon))
thfcorrs_lt = MV.zeros((nlat,nlon))
thfcorrs_st = MV.zeros((nlat,nlon))
pscorrs_lt = MV.zeros((nlat,nlon))
pscorrs_st = MV.zeros((nlat,nlon))
pscorrs = MV.zeros((nlat, nlon))
thfcorrs = MV.zeros((nlat, nlon))

thf_globe_an = spatial_ave(thf_an, lats)
ps_globe_an = spatial_ave(ps_an, lats)
sst_globe_an = spatial_ave(sst_an, lats)

#subtract global annual mean to isolate processes in NA
thfprime = thf_an.T - thf_globe_an
thfprime = thfprime.T

#thfprime = signal.detrend(thfprime, axis=0)
#thfprime = thf_an

psprime = ps_an.T - ps_globe_an
psprime = psprime.T

#psprime = signal.detrend(psprime, axis=0)

sstprime = sst_an.T - sst_globe_an
sstprime = sstprime.T

#sstprime = signal.detrend(sstprime, axis=0)


#CHANGE THIS TO MODIFY RM WINDOW LENGTH
N_map=5
ci = (N_map-1)/2
ltlag = 5
stlag = 1

lagmax=11
lags = np.arange(-lagmax,lagmax+1)

thflagcorrs = np.zeros((len(lags), nlat, nlon))
thflagcorrs_lt = np.zeros((len(lags), nlat, nlon))
thflagcorrs_st = np.zeros((len(lags), nlat, nlon))
pslagcorrs = np.zeros((len(lags), nlat, nlon))
pslagcorrs_lt = np.zeros((len(lags), nlat, nlon))
pslagcorrs_st = np.zeros((len(lags), nlat, nlon))

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

#thf_lt = running_mean(thf_an, N_map)
#thf_st = thf_an[ci:-ci,:] - thf_lt
thf_lt = running_mean(thfprime, N_map)
thf_st = thfprime[ci:-ci,:] - thf_lt

ps_lt = running_mean(psprime, N_map)
ps_st = psprime[ci:-ci,:] - ps_lt

nt = sst_an.shape[0]
nt_lt = thf_lt.shape[0]



#compute correlation between long-term/short-term AMO and sst, thf, SLP
#also compute lagged correlation between THF and AMO
#todo: compute lagged correlation between SLP and AMO
print 'calculating correlations between AMO and THF, SLP...'
for i in range(nlat):         

     print 'latitude', lats[i]
   

     sstprime_g = sstprime[:,i,:]
     thfprime_g = thfprime[:,i,:]
     psprime_g = psprime[:,i,:]
     thf_lt_g = thf_lt[:,i,:]
     thf_st_g = thf_st[:,i,:]
     ps_lt_g = ps_lt[:,i,:]
     ps_st_g = ps_st[:,i,:]
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd.reshape(-1,1), sstprime_g)
     sstcorrs[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd.reshape(-1,1), thfprime_g)
     thfcorrs[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd.reshape(-1,1), psprime_g)
     pscorrs[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd_lt.reshape(-1,1), thf_lt_g)
     thfcorrs_lt[i,:] = np.squeeze(clf.coef_)
     
     clf = linear_model.LinearRegression()
     clf.fit(AMOstd_st[stlag:].reshape(-1,1), thf_st_g[:-stlag,:])
     thfcorrs_st[i,:] = np.squeeze(clf.coef_)
     
     for lag in lags:
         
         scaler = StandardScaler()
         thfstd = scaler.fit_transform(thfprime_g)
         thfstd_lt = scaler.fit_transform(thf_lt_g)
         thfstd_st = scaler.fit_transform(thf_st_g)
         psstd = scaler.fit_transform(psprime_g)
         psstd_lt = scaler.fit_transform(ps_lt_g)
         psstd_st = scaler.fit_transform(ps_st_g)
         
         thfclf = linear_model.LinearRegression()
         thfclf_lt = linear_model.LinearRegression()
         thfclf_st = linear_model.LinearRegression()
         psclf = linear_model.LinearRegression()
         psclf_lt = linear_model.LinearRegression()
         psclf_st = linear_model.LinearRegression()
         #THF LAGS SST
         if lag > 0:
            thfclf.fit(AMOstd[:-lag], thfstd[lag:,:])
            thfclf_lt.fit(AMOstd_lt[:-lag], thfstd_lt[lag:,:])
            thfclf_st.fit(AMOstd_st[:-lag], thfstd_st[lag:,:])
            psclf.fit(AMOstd[:-lag], psstd[lag:,:])
            psclf_lt.fit(AMOstd_lt[:-lag], psstd_lt[lag:,:])
            psclf_st.fit(AMOstd_st[:-lag], psstd_st[lag:,:])
        #THF LEADS SST
         elif lag < 0: 
            thfclf.fit(AMOstd[-lag:], thfstd[:lag,:])
            thfclf_lt.fit(AMOstd_lt[-lag:], thfstd_lt[:lag,:])
            thfclf_st.fit(AMOstd_st[-lag:], thfstd_st[:lag,:])
            psclf.fit(AMOstd[-lag:], psstd[:lag,:])
            psclf_lt.fit(AMOstd_lt[-lag:], psstd_lt[:lag,:])
            psclf_st.fit(AMOstd_st[-lag:], psstd_st[:lag,:])
         else:
            thfclf.fit(AMOstd, thfstd)
            thfclf_lt.fit(AMOstd_lt, thfstd_lt)
            thfclf_st.fit(AMOstd_st, thfstd_st)
            psclf.fit(AMOstd, psstd)
            psclf_lt.fit(AMOstd_lt, psstd_lt)
            psclf_st.fit(AMOstd_st, psstd_st)
            
            
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

NAthf_laggedcorr_lt = np.zeros((len(windows), len(lags)))
NAthf_laggedcorr_st = np.zeros((len(windows), len(lags)))
NAps_laggedcorr_lt = np.zeros((len(windows), len(lags)))
NAps_laggedcorr_st = np.zeros((len(windows), len(lags)))

print 'calculating lagged correlation between AMO and NA THF, SLP for different RM windows...'
#commpute lagged correlation between smoothed AMO and NA THF for different RM windows
for k, N in enumerate(windows):
    ci = (N-1)/2
    AMO_lt = running_mean(AMO, N)
    AMO_st = AMO[ci:-ci] - AMO_lt
    AMOstd_lt = (AMO_lt - np.mean(AMO_lt)) / (np.std(AMO_lt) * len(AMO_lt))
    AMOstd_st = (AMO_st - np.mean(AMO_st)) / (np.std(AMO_st) * len(AMO_st))
    
    NAthf_lt = running_mean(NAthf, N)
    NAthf_st = NAthf[ci:-ci] - NAthf_lt 
    NAthfstd_lt = (NAthf_lt - np.mean(NAthf_lt)) / (np.std(NAthf_lt))
    NAthfstd_st = (NAthf_st - np.mean(NAthf_st)) / (np.std(NAthf_st))
    NAthflaggedcorr_lt_temp = np.correlate(NAthfstd_lt, AMOstd_lt, 'full')
    NAthflaggedcorr_st_temp = np.correlate(NAthfstd_st, AMOstd_st, 'full')

    NAps_lt = running_mean(NAps, N)
    NAps_st = NAps[ci:-ci] - NAps_lt 
    NApsstd_lt = (NAps_lt - np.mean(NAps_lt)) / (np.std(NAps_lt))
    NApsstd_st = (NAps_st - np.mean(NAps_st)) / (np.std(NAps_st))
    NApslaggedcorr_lt_temp = np.correlate(NApsstd_lt, AMOstd_lt, 'full')
    NApslaggedcorr_st_temp = np.correlate(NApsstd_st, AMOstd_st, 'full')
    
    lagzero = len(NAthflaggedcorr_lt_temp)/2
    NAthf_laggedcorr_lt[k,:] = NAthflaggedcorr_lt_temp[lagzero-lagmax:lagzero+lagmax+1]
    NAthf_laggedcorr_st[k,:] = NAthflaggedcorr_st_temp[lagzero-lagmax:lagzero+lagmax+1]
    NAps_laggedcorr_lt[k,:] = NApslaggedcorr_lt_temp[lagzero-lagmax:lagzero+lagmax+1]
    NAps_laggedcorr_st[k,:] = NApslaggedcorr_st_temp[lagzero-lagmax:lagzero+lagmax+1]
    #AMOmid_autocorr[k,:] = np.correlate(AMOstd_smooth, AMOstd_smooth, 'full')[lagzero-lagmax:lagzero+lagmax]
    
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
ax.plot(tyears[ci:-ci],AMO_smooth,label='{:1.0f}-yr RM'.format(N_map))
#plt.plot(tyears[ci:-ci], AMO_st, label='short-term residual')
#ax.fill_between(tyears[ci:-ci], 0, AMO_smooth, where= AMO_smooth>0, color='red')
#ax.fill_between(tyears[ci:-ci], 0, AMO_smooth, where= AMO_smooth<0, color='blue')
ax.set_title(r'AMO (NA mean SST - global mean SST)'.format(latbounds[0], latbounds[1]))
ax.axhline(0, color='black')
ax.set_ylabel(r'SST ($^{{\circ}}$C)')
ax.legend()
ax.set_xlabel('time (years)')
plt.savefig(fout + 'MERRA2_AMO_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#Plot NA THF
fig=plt.figure(figsize=(16,14))
plt.suptitle('NA averaging region: {:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N, {:2.0f}$^{{\circ}}$W to {:2.0f}$^{{\circ}}$W'.format(latbounds[0], latbounds[1], 360-lonbounds[0], 360-lonbounds[1]))
ax = fig.add_subplot(311)
ax.plot(tyears, thfanom_globe_an)
ax.set_ylabel(r'THF (W m$^{{-2}}$)')
#ax.set_ylim(-1,1)
ax.set_title(r'global mean THF (base period: {:3.0f} to {:3.0f})'.format(tyears[baseti], tyears[basetf]))
ax.axhline(0, color='black')
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
ax.plot(tyears, NAthf2)
ax.plot(tyears[ci:-ci], NAthf_smooth)
ax.set_ylabel(r'THF (W m$^{{-2}}$)')
plt.title(r'NA mean THF - global mean THF')
plt.axhline(0, color='black')
ax.set_xlabel('time (years)')
plt.savefig(fout + 'MERRA2_thfanom_timeseries_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
plt.close()
    
#Plot correlation between smoothed THF and AMO at different lags and different smoothing window lengths
fig=plt.figure(figsize=(18,14))
ax = fig.add_subplot(221)
ax.pcolor(ll, ww, NAthf_laggedcorr_lt, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.set_title('long-term correlation of AMO with NA THF')
ax.set_xlabel('THF lag (years)')
ax.set_ylabel('RM window (years)')
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
ax = fig.add_subplot(222)
h = ax.pcolor(ll, ww, NAthf_laggedcorr_st, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.set_title('short-term correlation (RM residual)')
ax.set_xlabel('THF lag (years)')
#ax.set_ylabel('smoothing (years)') 
fig.colorbar(h, ax=ax, ticks=np.arange(-1,1.5,0.5), orientation="vertical")
#plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_hist_{:2.0f}Nto{:2.0f}N.pdf'.format(latbounds[0], latbounds[1]))
#plt.close()

i = np.where(windows>N_map)[0][0]-1

#fig=plt.figure(figsize=(18,6))
ax = fig.add_subplot(223)
ax.plot(lags, NAthf_laggedcorr_lt[i,:])
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_ylim(-1,1)
ax.set_title('long-term correlation of AMO with NA THF ({:1.0f}-yr RM)'.format(windows[i]))
ax.set_xlabel('THF lag (years)')
ax = fig.add_subplot(224)
ax.plot(lags, NAthf_laggedcorr_st[i,:])
ax.set_ylim(-1,1)
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_title('short-term correlation'.format(windows[i]))
ax.set_xlabel('THF lag (years)')
plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
#plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_timeseries_{:1.0f}year_{:2.0f}Nto{:2.0f}N.pdf'.format(windows[i],latbounds[0], latbounds[1]))
plt.close()

fig=plt.figure(figsize=(18,14))
ax = fig.add_subplot(221)
ax.pcolor(ll, ww, NAps_laggedcorr_lt, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.set_title('long-term correlation of AMO with NA SLP')
ax.set_xlabel('SLP lag (years)')
ax.set_ylabel('RM window (years)')
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
ax = fig.add_subplot(222)
h = ax.pcolor(ll, ww, NAps_laggedcorr_st, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.set_title('short-term correlation (RM residiual)')
ax.set_xlabel('SLP lag (years)')
#ax.set_ylabel('smoothing (years)') 
fig.colorbar(h, ax=ax, orientation="vertical")
#plt.savefig(fout + 'MERRA2_AMO_SLP_lagcorr_smoothhist_{:2.0f}Nto{:2.0f}N.pdf'.format(latbounds[0], latbounds[1]))
#plt.close()

#fig=plt.figure(figsize=(18,6))
ax = fig.add_subplot(223)
ax.plot(lags, NAps_laggedcorr_lt[i,:])
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_ylim(-1,1)
ax.set_title('long-term correlation of AMO with NA SLP ({:1.0f}-yr RM)'.format(windows[i]))
ax.set_xlabel('SLP lag (years)')
ax = fig.add_subplot(224)
ax.plot(lags, NAps_laggedcorr_st[i,:])
ax.set_ylim(-1,1)
ax.axhline(0, color='black')
ax.axvline(0, color='black')
ax.set_title('short-term correlation'.format(windows[i]))
ax.set_xlabel('SLP lag (years)')
plt.savefig(fout + 'MERRA2_AMO_SLP_lagcorr_smoothhist_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
#plt.savefig(fout + 'MERRA2_AMO_SLP_lagcorr_timeseries_{:1.0f}year_{:2.0f}Nto{:2.0f}N.pdf'.format(windows[i],latbounds[0], latbounds[1]))
plt.close()


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



#SHOULDN'T THIS BE EQUIVALENT TO THE CORRELATION BETWEEN SMOOTHED AMO AND NA THF? i.e. NAthf_laggedcorr_lt[i,:]
#thflagcorrs_test = np.ma.average(thflagcorrs_lt_zonalave, axis=1, weights=weights[NAminlati:NAmaxlati])

lagg, latt = np.meshgrid(lags, NAlats)

#Plot zonally-averaged lagged correlation between long-term AMO and THF
fig=plt.figure(figsize=(22,6))
#plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
ax = fig.add_subplot(131)
h = ax.pcolor(lagg, latt, thflagcorrs_zonalave.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
ax.set_title('correlation of AMO with NA THF')
ax.set_xlabel('THF lag (years)')
ax.set_ylabel('latitude (degrees)')
ax.set_ylim(0,60)
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
ax = fig.add_subplot(132)
ax.pcolor(lagg, latt, thflagcorrs_lt_zonalave.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
ax.set_xlabel('THF lag (years)')
ax.set_ylim(0,60)
ax = fig.add_subplot(133)
h = ax.pcolor(lagg, latt, thflagcorrs_st_zonalave.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
ax.set_title('short-term correlation'.format(N_map))
ax.set_xlabel('THF lag (years)')
ax.set_ylim(0,60)
fig.colorbar(h, ax=ax, orientation="vertical")
plt.savefig(fout + 'MERRA2_AMO_thf_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
plt.close()

#Plot zonally-averaged lagged correlation between long-term AMO and THF
fig=plt.figure(figsize=(22,6))
#plt.suptitle('AMO ({:2.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
ax = fig.add_subplot(131)
h = ax.pcolor(lagg, latt, pslagcorrs_zonalave.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
ax.set_title('correlation of AMO with NA SLP')
ax.set_xlabel('SLP lag (years)')
ax.set_ylabel('latitude (degrees)')
ax.set_ylim(0,60)
#cbar = fig.colorbar(cax, ticks=np.arange(-1,1.5,0.5), orientation='vertical')
ax = fig.add_subplot(132)
h = ax.pcolor(lagg, latt, pslagcorrs_lt_zonalave.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
ax.set_title('long-term correlation ({:1.0f}-yr RM)'.format(N_map))
ax.set_xlabel('SLP lag (years)')
ax.set_ylim(0,60)
ax = fig.add_subplot(133)
ax.pcolor(lagg, latt, pslagcorrs_st_zonalave.T, vmin=-1, vmax=1, cmap=plt.cm.RdBu_r)
ax.axvline(0, color='k')
ax.set_title('short-term correlation'.format(N_map))
ax.set_xlabel('SLP lag (years)')
ax.set_ylim(0,60)
fig.colorbar(h, ax=ax, orientation="vertical")
plt.savefig(fout + 'MERRA2_AMO_SLP_lagcorr_zonalave_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
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
thfstep = 0.5
thfstep_lt = 5
thfstep_st =5
SLPlevels = np.arange(-2, 2+pstep, pstep)
sstlevels = np.arange(-0.8, 0.8+sststep, sststep)
thflevels = np.arange(-15, 15+thfstep, thfstep)
thflevels_lt = np.arange(-200, 200+thfstep_lt, thfstep_lt)
thflevels_st = np.arange(-200, 200+thfstep_st, thfstep_st)

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
ct = ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', linewidths=1)
ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
plot = ax.contourf(x, y, sstcorrs, cmap=plt.cm.RdBu_r, levels=sstlevels, extend='both', transform=prj)
cb = plt.colorbar(plot, label=r'$^{{\circ}}$C')
plt.title(r'regression of SST, SLP on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + 'MERRA2_AMO_sstSLP_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
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
#ax.contour(x, y, pscorrs, levels=SLPlevels, colors='k', inline=1, linewidths=1)
plot = ax.contourf(x, y, thfcorrs, cmap=plt.cm.RdBu_r, levels=thflevels, extend='both', transform=prj)
cb = plt.colorbar(plot, label=r'W m$^{-2}$')
plt.title(r'regression of THF on AMO ({:1.0f}$^{{\circ}}$N to {:2.0f}$^{{\circ}}$N)'.format(latbounds[0], latbounds[1]))
plt.savefig(fout + 'MERRA2_AMO_thf_corr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
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
plt.title(r'resgresion of long-term THF on AMO ({:1.0f}-yr RM)'.format(N_map))
plt.savefig(fout + 'MERRA2_AMO_thf_ltcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
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
plt.savefig(fout + 'MERRA2_AMO_thf_stcorr_map_{:2.0f}Nto{:2.0f}N_detr{:s}.pdf'.format(latbounds[0], latbounds[1], str(detr)[0]))
plt.close()















































