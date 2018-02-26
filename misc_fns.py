#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:11:59 2017

@author: cpatrizio
"""

import numpy as np

def calc_AMO(sst, latbounds, ti, tf):
    slat = latbounds[0]
    nlat = latbounds[1]
    sst = sst.subRegion(latitude=[-60,60])
    #sst_globe = cdutil.averager(sst, axis='xy', weights='weighted')
    sst_globe = spatial_ave(sst, sst.getLatitude()[:])
    sst_globe_an = an_ave(sst_globe)
    sstbase_globe = np.ma.average(sst_globe_an[ti:tf], axis=0)
    #global annual mean SST anomaly 
    #sstbase_globe = 0
    sstanom_globe_an = sst_globe_an - sstbase_globe
    
    nasst = sst.subRegion(latitude=[slat,nlat], longitude=[-80,0])
    sst_na = spatial_ave(nasst, nasst.getLatitude()[:])
    #sst_na = cdutil.averager(nasst, axis='xy', weights='weighted')
    sst_na_an = an_ave(sst_na)
    sstbase_na = np.ma.average(sst_na_an[ti:tf], axis=0)
    #sstbase_na = 0
    #NA annual mean SST anomaly
    sstanom_na_an = sst_na_an - sstbase_na

    AMO = sstanom_na_an - sstanom_globe_an
    return AMO, sstanom_globe_an, sstanom_na_an

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def spatial_ave(data, lats):
    #lats = data.getLatitude()[:]
    weights = np.cos(np.deg2rad(lats))
    zonal_ave = np.ma.average(data, axis=2)
    spatial_ave = np.ma.average(zonal_ave, axis=1, weights=weights)
    return spatial_ave

def an_ave(data):
    #COMPUTES ANNUAL AVERAGE ASSUMING MONTHLY DATA
    skip = data.shape[0] % 12
    data = data[skip:,...]
    nyears = data.shape[0] / 12
    datanew = data.reshape(nyears, 12, *data.shape[1:])
    data_an = np.ma.average(datanew, axis=1)
    return data_an
    
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