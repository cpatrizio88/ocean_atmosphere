#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:11:59 2017

@author: cpatrizio
"""

import numpy as np
import MV2 as MV
import genutil
from sklearn import linear_model
from scipy import signal

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - np.ma.mean(A, axis=1, keepdims=True)
    B_mB = B - np.ma.mean(B, axis=1, keepdims=True)

    # Sum of squares across rows
    ssA = np.ma.sum(A_mA**2, axis=1)
    ssB = np.ma.sum(B_mB**2, axis=1)

    # Finally get corr coeff
    return np.ma.dot(A_mA,B_mB.T)/np.ma.sqrt(np.ma.dot(ssA[:,None],ssB[None]))

def cov2_coeff(A,B):
    A_mA = A - np.ma.mean(A, axis=1, keepdims=True)
    B_mB = B - np.ma.mean(B, axis=1, keepdims=True)
    
    nt = A_mA.shape[1]

    # Sum of squares across rows
    #ssA = np.ma.sum(A_mA**2, axis=1)
    #ssB = np.ma.sum(B_mB**2, axis=1)

    # Finally get corr coeff
    return np.ma.dot(A_mA,B_mB.T)/(nt-1)
    

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def detrend(y):
    
    #assumes y has time dimension 
    
    trend,intercept = genutil.statistics.linearregression(y)

    #time = np.arange(nobs)
    #time = MV.array(time)
    time = MV.array(y.getTime()) # makes an array of time dimension
    time.setAxis(0,y.getTime()) # passes itslef as axis...

    # The following "grows" trend and time so they are 3D
    detrender,time = genutil.grower(time,trend)
    
    detrended = y - detrender*time
    return detrended

def regressout_x(y, x):
    

    
    nt = y.shape[0]
    dims = y.shape[1:]
    
    y = y.reshape(nt, -1)
    
    
    #yxfit = np.diag(np.ma.cov(x, y, rowvar=False)[:N,N:])
    
    clf = linear_model.LinearRegression()
    
    
    clf.fit(x.reshape(-1,1), y)
    
    
    yxfit = np.squeeze(clf.coef_)
    
    #N = y.shape[0]
#    
#    y = y - y.mean(axis=0, keepdims=True)
#    x = (x - x.mean())/(N-1)
#    
#    yxfit = np.dot(y.T, x)
#
#    yxfit = np.repeat(yxfit[np.newaxis,:], nt, axis=0)
#    
#    yxfit = yxfit.reshape(nt, -1)
    
    x = np.repeat(x[:,np.newaxis],np.prod(y.shape[1:]), axis=1)

    y_x =  np.multiply(yxfit, x)

    y = y - y_x
    
    y = y.reshape(nt, *dims)
    
    
    
    return y

#    time = MV.array(y.getTime()) # makes an array of time dimension
#    time.setAxis(0,y.getTime()) # passes itslef as axis...
#
#    # The following "grows" trend and time so they are 3D
#    detrender,time = genutil.grower(time,CTI)
#    
#    detrended = y - detrender*time
#    return detrended
#    
        

def detrend_common(y, order=1):
    '''detrend multivariate series by common trend

    Paramters
    ---------
    y : ndarray
       data, can be 1d or nd. if ndim is greater then 1, then observations
       are along zero axis
    order : int
       degree of polynomial trend, 1 is linear, 0 is constant

    Returns
    -------
    y_detrended : ndarray
       detrended data in same shape as original 

    '''
    nobs = y.shape[0]
    shape = y.shape
    y_ = y.ravel()
    nobs_ = len(y_)
    t = np.repeat(np.arange(nobs), nobs_ /float(nobs))
    exog = np.vander(t, order+1)
    params = np.linalg.lstsq(exog, y_)[0]
    fittedvalues = np.dot(exog, params)
    resid = (y_ - fittedvalues).reshape(*shape)
    return resid, params

#def regress(y, X, order=1):
#    
#    nobs = y.shape[0]
#    shape = y.shape
#    y_ = y.reshape(nobs, -1)
#    kvars_ = len(y_)
#    t = np.arange(nobs)
#    exog = np.vander(t, order+1)
#    params = np.linalg.lstsq(exog, y_)[0]



def detrend_separate(y, order=1):
    '''detrend multivariate series by series specific trends

    Paramters
    ---------
    y : ndarray
       data, can be 1d or nd. if ndim is greater then 1, then observations
       are along zero axis
    order : int
       degree of polynomial trend, 1 is linear, 0 is constant

    Returns
    -------
    y_detrended : ndarray
       detrended data in same shape as original 

    '''
    nobs = y.shape[0]
    shape = y.shape
    y_ = y.reshape(nobs, -1)
    kvars_ = len(y_)
    t = np.arange(nobs)
    exog = np.vander(t, order+1)
    params = np.linalg.lstsq(exog, y_)[0]
    fittedvalues = np.dot(exog, params)
    resid = (y_ - fittedvalues).reshape(*shape)
    return resid, params

def calc_AMO(sst, latbounds, lats, lons, ti, tf):
    
    #assumes fields are annually averaged 
    #assumes lats )-90,90) and lons (0,360) are monotonic increasing
    slat = latbounds[0]
    nlat = latbounds[1]
    slati = np.where(lats > slat)[0][0]
    nlati = np.where(lats > nlat)[0][0]
    slatgi = np.where(lats > -60)[0][0]
    nlatgi = np.where(lats > 60)[0][0]
    wloni = np.where(lons > 280)[0][0]
    eloni =  np.where(lons > 359.5)[0]
    
    if len(eloni) == 0:
        eloni=-1
    else:
        eloni = eloni[0]
    #sst = sst.subRegion(latitude=[-60,60])
    #sst_globe = cdutil.averager(sst, axis='xy', weights='weighted')
    #sst_globe = spatial_ave(sst, sst.getLatitude()[:])
    sst_globe = spatial_ave(sst[:,slatgi:nlatgi,:],lats[slatgi:nlatgi])
    #sst_globe_an = an_ave(sst_globe)
    sst_globe_an = sst_globe
    sstbase_globe = np.ma.average(sst_globe_an[ti:tf], axis=0)
    #global annual mean SST anomaly 
    #sstbase_globe = 0
    sstanom_globe_an = sst_globe_an - sstbase_globe
    
    #nasst = sst.subRegion(latitude=[slat,nlat], longitude=[-80,0])
    nasst = sst[:,slati:nlati,wloni:eloni]
    #sst_na = spatial_ave(nasst, nasst.getLatitude()[:])
    sst_na = spatial_ave(nasst, lats[slati:nlati])
    sst_na_an = sst_na
    #sst_na = cdutil.averager(nasst, axis='xy', weights='weighted')
    #sst_na_an = an_ave(sst_na)
    sstbase_na = np.ma.average(sst_na_an[ti:tf], axis=0)
    #sstbase_na = 0
    #NA annual mean SST anomaly
    sstanom_na_an = sst_na_an - sstbase_na

    AMO = sstanom_na_an - sstanom_globe_an
    return AMO, sstanom_globe_an, sstanom_na_an

def calc_NA_globeanom(field, latbounds, lats, lons, ti, tf):
    
    #assumes fields are annually averaged 
    #assumes lats (-90,90) and lons (0,360) are monotonic increasing
    slat = latbounds[0]
    nlat = latbounds[1]
    slati = np.where(lats > slat)[0][0]
    nlati = np.where(lats > nlat)[0][0]
    slatgi = np.where(lats > -60)[0][0]
    nlatgi = np.where(lats > 60)[0][0]
    wloni = np.where(lons > 280)[0][0]
    eloni =  np.where(lons > 359.5)[0]
    
    if len(eloni) == 0:
        eloni=-1
    else:
        eloni = eloni[0]
    #field = field.subRegion(latitude=[-60,60])
    #sst_globe = cdutil.averager(sst, axis='xy', weights='weighted')
    #field_globe = spatial_ave(field, field.getLatitude()[:])
    field_globe = spatial_ave(field[:,slatgi:nlatgi,:], lats[slatgi:nlatgi])
    #field_globe_an = an_ave(field_globe)
    field_globe_an = field_globe
    fieldbase_globe = np.ma.average(field_globe_an[ti:tf], axis=0)
    #global annual mean SST anomaly 
    #sstbase_globe = 0
    fieldanom_globe_an = field_globe_an - fieldbase_globe
    
    #nafield = field.subRegion(latitude=[slat,nlat], longitude=[-80,0])
    nafield = field[:,slati:nlati:,wloni:eloni]
    #field_na = spatial_ave(nafield, nafield.getLatitude()[:])
    field_na = spatial_ave(nafield, lats[slati:nlati])
    
    #sst_na = cdutil.averager(nasst, axis='xy', weights='weighted')
    #field_na_an = an_ave(field_na)
    field_na_an = field_na
    fieldbase_na = np.ma.average(field_na_an[ti:tf], axis=0)
    #sstbase_na = 0
    #NA annual mean SST anomaly
    fieldanom_na_an = field_na_an - fieldbase_na

    field_anom = fieldanom_na_an - fieldanom_globe_an
    return field_anom, fieldanom_globe_an, fieldanom_na_an

def calc_NA_globeanom3D(field, latbounds, lats, lons, ti, tf):
    #assumes field has shape (t, z, x, y)
    #returns fields with shape (t, z)
    #assumes fields are annually averaged 
    #assumes lats (-90,90) and lons (0,360) are monotonic increasing
    slat = latbounds[0]
    nlat = latbounds[1]
    slati = np.where(lats > slat)[0][0]
    nlati = np.where(lats > nlat)[0][0]
    slatgi = np.where(lats > -60)[0][0]
    nlatgi = np.where(lats > 60)[0][0]
    wloni = np.where(lons > 280)[0][0]
    eloni =  np.where(lons > 359.5)[0]
    
    if len(eloni) == 0:
        eloni=-1
    else:
        eloni = eloni[0]
    #field = field.subRegion(latitude=[-60,60])
    #sst_globe = cdutil.averager(sst, axis='xy', weights='weighted')
    #field_globe = spatial_ave(field, field.getLatitude()[:])
    field_globe = spatial_ave(field[:,:,slatgi:nlatgi,:], lats[slatgi:nlatgi])
    #field_globe_an = an_ave(field_globe)
    field_globe_an = field_globe
    fieldbase_globe = np.ma.average(field_globe_an[ti:tf,:], axis=0)
    #global annual mean SST anomaly 
    #sstbase_globe = 0
    fieldanom_globe_an = field_globe_an - fieldbase_globe
    
    #nafield = field.subRegion(latitude=[slat,nlat], longitude=[-80,0])
    nafield = field[:,:,slati:nlati:,wloni:eloni]
    #field_na = spatial_ave(nafield, nafield.getLatitude()[:])
    field_na = spatial_ave(nafield, lats[slati:nlati])
    
    #sst_na = cdutil.averager(nasst, axis='xy', weights='weighted')
    #field_na_an = an_ave(field_na)
    field_na_an = field_na
    fieldbase_na = np.ma.average(field_na_an[ti:tf,:], axis=0)
    #sstbase_na = 0
    #NA annual mean SST anomaly
    fieldanom_na_an = field_na_an - fieldbase_na

    field_anom = fieldanom_na_an - fieldanom_globe_an
    return field_anom, fieldanom_globe_an, fieldanom_na_an


def running_mean(x, N):
    cumsum = np.ma.cumsum(np.insert(x, 0, 0, axis=0), axis=0) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def spatial_ave(data, lats):
    #assumes data has dimensions (t,z, x, y)
    #returns dimension (t,z)
    #lats = data.getLatitude()[:]
    weights = np.cos(np.deg2rad(lats))
    zonal_ave = np.ma.average(data, axis=-1)
    spatial_ave = np.ma.average(zonal_ave, axis=-1, weights=weights)
    return spatial_ave

def an_ave(data):
    #COMPUTES ANNUAL AVERAGE ASSUMING MONTHLY DATA
    skip = data.shape[0] % 12
    data = data[skip:,...]
    nyears = data.shape[0] / 12
    datanew = data.reshape(nyears, 12, *data.shape[1:])
    data_an = np.ma.average(datanew, axis=1)
    return data_an

#def calcRH(q,t,p):
    
def calcsatspechum(t,p):

## T is temperature in Kelvins, P is pressure in hPa 

  ## Formulae from Buck (1981):
  es = (1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*(t-273.15)/(240.97+(t-273.15)))
  wsl = (.622*es)/(p-es)# saturation mixing ratio wrt liquid water (g/kg)
  
  es = (1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*(t-273.15)/(272.55+(t-273.15)))

  wsi = (.622*es)/(p-es) # saturation mixing ratio wrt ice (g/kg)

  
  ws = wsl
  
  freezing = t < 273.15
  
  
  ws = ws.getValue()
  wsi = wsi.getValue()
  
  ws[freezing]=wsi[freezing]
  
 
  return ws/(1+ws)