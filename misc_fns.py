#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:11:59 2017

@author: cpatrizio
"""

import numpy as np
#import MV2 as MV
#import genutil
#from sklearn import linear_model
#from scipy import signal
#import cdms2
import xarray as xr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pandas import DataFrame 






## define a function to compute a linear trend of a timeseries
#def linear_trend(x):
#    # pf = np.polyfit(x['i1'].values, x, 1)
#    trend,intercept = genutil.statistics.linearregression(x,axis=1)
#    # we need to return a dataarray or else xarray's groupby won't be happy
#    return xr.DataArray(trend)
#
#def linear_detrend(x):
#        
#    # stack lat and lon into a single dimension called allpoints
#    stacked = x.stack(allpoints=['i3','i4'])
#    # apply the function over allpoints to calculate the trend at each point
#    trend = stacked.groupby('allpoints').apply(linear_trend)
#    # unstack back to lat lon coordinates
#    times = x['i1'].values
#    
#    detrended = x - trend*times
#    detrended_unstacked = detrended.unstack('allpoints')
#    
#    return detrended_unstacked

#def multireg(x1, x2, y, time_dim=0, lon_dim=1, lat_dim=2):
#
#    #y = y.flatten()
#    y=y.stack(allpoints = ['lat', 'lon']).squeeze()
#
#    
#    model = {
#    'vec1': x1,
#    'vec2': x2,
#    'compound_vec': y}
#    
#    df = DataFrame(model, columns=['vec1','vec2','compound_vec'])
#    x = df[['vec1','vec2']].astype(object)
#    y = df['compound_vec'].astype(object)
#    
#    #X = np.column_stack((x1,x2))
#    #X = X.T
#    
#        
#    regr = LinearRegression()
#    regr.fit(x, y)
#    coefs = regr.coef_
#    return coefs

def reg(x,y,time_dim=0,lagx=0):
    
    #1. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards.
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time', how = 'all')

#    if lagy!=0:
#        y   = y.shift(time = -lagy).dropna(dim='time', how = 'all')
        
    #3. Compute data length, mean and standard deviation along time dimension for further use:
    
    x,y = xr.align(x,y)
    n     = x.shape[time_dim]
    xmean = np.mean(x,axis=time_dim)
    ymean = np.mean(y,axis=time_dim)
    xstd  = np.std(x,axis=time_dim)
    #ystd  = np.std(y,axis=time_dim)

    #4. Compute covariance along time dimension
    cov   =  np.sum((x - xmean)*(y - ymean), axis=time_dim)/(n)
    
    
       #5. Compute regression slope and intercept:
    slope     = cov/(xstd**2)
    intercept = ymean - xmean*slope

    return slope, intercept

def cov(x,y,time_dim=0,lagx=0,monthly=False):
    
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards.
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        #x   = x.shift(time = -lagx).dropna(dim='time', how = 'all')
        x   = x.shift(time = -lagx).dropna(dim='time', how = 'all')
    
   # x,y = xr.align(x,y)
   
    if monthly:
        y = y.groupby('month')



    #3. Compute data length, mean and standard deviation along time dimension for further use:
    n     = x.shape[time_dim]
    xmean = np.mean(x,axis=time_dim)
    ymean = np.mean(y,axis=time_dim)
    #xstd  = np.std(x,axis=time_dim)
    #ystd  = np.std(y,axis=time_dim)
    
    if monthly:
        yprime = (y-ymean).groupby('month')
    else:
        yprime = y-ymean

    

    #4. Compute covariance along time dimension
    cov   =  np.sum((x - xmean)*(yprime), axis=time_dim)/(n)
    
    return cov

def cor(x,y,time_dim=0,lagx=0):
    
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards.
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        #x   = x.shift(time = -lagx).dropna(dim='time', how = 'all')
        x   = x.shift(time = -lagx).dropna(dim='time', how = 'all')
    
    x,y = xr.align(x,y)


    #3. Compute data length, mean and standard deviation along time dimension for further use:
    n     = x.shape[time_dim]
    xmean = np.mean(x,axis=time_dim)
    ymean = np.mean(y,axis=time_dim)
    xstd  = np.std(x,axis=time_dim)
    ystd  = np.std(y,axis=time_dim)

    #4. Compute covariance along time dimension
    cov   =  np.sum((x - xmean)*(y - ymean), axis=time_dim)/(n)
    
    return cov/(xstd*ystd)

def cov2_coeff(A,B):
    A_mA = A - np.ma.mean(A, axis=1, keepdims=True)
    B_mB = B - np.ma.mean(B, axis=1, keepdims=True)
    
    nt = A_mA.shape[1]

    # Sum of squares across rows
    #ssA = np.ma.sum(A_mA**2, axis=1)
    #ssB = np.ma.sum(B_mB**2, axis=1)

    # Finally get corr coeff
    return np.ma.dot(A_mA,B_mB.T)/(nt)




def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    #A_mA = A - np.ma.mean(A, axis=1, keepdims=True)
    #B_mB = B - np.ma.mean(B, axis=1, keepdims=True)
    
    # Sum of squares across rows
    #ssA = np.ma.sum(A_mA**2, axis=1)
    #ssB = np.ma.sum(B_mB**2, axis=1)
    
    cov = cov2_coeff(A,B)
    std_A = np.std(A,axis=1)
    std_B = np.std(B,axis=1)
    
    return cov/(std_A*std_B)

    # Finally get corr coeff
    #return np.ma.dot(A_mA,B_mB.T)/np.ma.sqrt(np.ma.dot(ssA[:,np.newaxis],ssB[np.newaxis,:]))
    #return np.ma.dot(A_mA,B_mB.T)/np.ma.sqrt(np.ma.dot(ssA,ssB))
    

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
    
    #removes local trend from y
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


def detrend_ECCO(y):
    
    #removes local trend from y
    #assumes y has time dimension 
    
    trend,intercept = genutil.statistics.linearregression(y,axis=1)

    #time = np.arange(nobs)
    #time = MV.array(time)
    time = MV.array(y['i1'].values) # makes an array of time dimension
    ta = cdms2.createAxis(time)
    ta.id = 'time'
    ta.units = 'months since 1992-01-16'
    time.setAxis(0,ta) # passes itslef as axis...

    # The following "grows" trend and time so they are 3D
    detrender,time = genutil.grower(time,trend)
    
    detrended = y - detrender*time
    return detrended


def regressout_x(x, y, time_dim=0, lagx=0):
    
    #assume 
    
    #1. Add lag information if any, and shift the data accordingly
    if lagx!=0:
        #If x lags y by 1, x must be shifted 1 step backwards.
        #But as the 'zero-th' value is nonexistant, xr assigns it as invalid (nan). Hence it needs to be dropped
        x   = x.shift(time = -lagx).dropna(dim='time', how = 'all')
    
    a, b = reg(x,y,time_dim=time_dim)  
    
    yfit = a*x

    return y - yfit
    


# def regressout_x(y, x):
    

    
#     nt = y.shape[0]
#     dims = y.shape[1:]
    
#     y = y.reshape(nt, -1)
    
    
#     #yxfit = np.diag(np.ma.cov(x, y, rowvar=False)[:N,N:])
    
#     clf = linear_model.LinearRegression()
    
    
#     clf.fit(x.reshape(-1,1), y)
    
    
#     yxfit = np.squeeze(clf.coef_)
    
#     #N = y.shape[0]
# #    
# #    y = y - y.mean(axis=0, keepdims=True)
# #    x = (x - x.mean())/(N-1)
# #    
# #    yxfit = np.dot(y.T, x)
# #
# #    yxfit = np.repeat(yxfit[np.newaxis,:], nt, axis=0)
# #    
# #    yxfit = yxfit.reshape(nt, -1)
    
#     x = np.repeat(x[:,np.newaxis],np.prod(y.shape[1:]), axis=1)

#     y_x =  np.multiply(yxfit, x)

#     y = y - y_x
    
#     y = y.reshape(nt, *dims)
    
    
    
#    return y

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

def spatial_ave_xr(data, lats):
    #assumes data has dimensions (t,z, x, y)
    #returns dimension (t,z)
    #lats = data.getLatitude()[:]
    weights = np.cos(np.deg2rad(lats))
    sum_of_weights = np.sum(weights)
    zonal_ave = data.mean(dim='lon')
    spatial_ave = ((zonal_ave*weights)/sum_of_weights).sum(dim='lat')
    #spatial_ave = np.ma.average(zonal_ave, axis=-1, weights=weights)
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