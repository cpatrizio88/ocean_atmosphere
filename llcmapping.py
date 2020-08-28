import os
import numpy as np
from matplotlib import pyplot as plt
import cartopy as cart
import pyresample
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#For offline use, we need the land features saved locally
#cart.config['data_dir']= '~/.local/share/cartopy/shapefiles/natural_earth/physical'
#cart.config['pre_existing_data_dir']= '~/.local/share/cartopy/shapefiles/natural_earth/physical'


class LLCMapper:

    def __init__(self, ds, dx=0.25, dy=0.25):

        # Extract LLC 2D coordinates
        lons_1d = ds.XC.values.ravel()
        lats_1d = ds.YC.values.ravel()
        
        # Define original grid
        self.orig_grid = pyresample.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)

        # Longitudes latitudes to which we will we interpolate
        lon_tmp = np.arange(-180, 180, dx) + dx/2
        lat_tmp = np.arange(-90, 90, dy) + dy/2

        # Define the lat lon points of the two parts.
        self.new_grid_lon, self.new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)
        self.new_grid  = pyresample.geometry.GridDefinition(lons=self.new_grid_lon,
                                                            lats=self.new_grid_lat)
        
        #self.lon_0 = lon_0

        
        
    def __call__(self, da, ax=None, bnds=[0,360,-90,90], projection_name='PlateCarree', **plt_kwargs):
        
         # Central longitude must be between -180 and 180 (greenwich meridian is 0) 
        #lon_0 = (bnds[0] + bnds[1])/2 - 360
        lon_0 = (bnds[0] + bnds[1])/2 
        
        #print('lon_0', lon_0)
        
        if projection_name == 'PlateCarree':
            projection = cart.crs.PlateCarree(central_longitude=lon_0)
        elif projection_name == 'Robinson':
            projection = cart.crs.Robinson(central_longitude=lon_0)
        else:
            print('Projection name must be "PlateCarree" or "Robinson".')

        assert set(da.dims) == set(['face', 'j', 'i']), "da must have dimensions ['face', 'j', 'i']"

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        #print(self.orig_grid.shape)
        #print(da.values.shape)
        #print(self.new_grid.shape)

        field = pyresample.kd_tree.resample_nearest(self.orig_grid, da.values,
                                                    self.new_grid,
                                                    radius_of_influence=100000,
                                                    fill_value=None)


        vmax = plt_kwargs.pop('vmax', field.max())
        vmin = plt_kwargs.pop('vmin', field.min())

        m = plt.axes(projection=projection)
        x,y = self.new_grid_lon, self.new_grid_lat
        
        #ax= plt.gca()

                    
        pardiff = 30.
        merdiff = 60.
        
        if np.abs(bnds[1] - bnds[0]) < 180:
            merdiff = 30.
        if np.abs(bnds[1] - bnds[0]) < 90:
            merdiff = 15.
        if np.abs(bnds[3]- bnds[2]) < 90:
            pardiff = 15.
            
        par = np.arange(-90.,90.+pardiff,pardiff)
        mer = np.arange(-180.,180.+merdiff,merdiff)
            
        ax=plt.gca()
        ax.set_xticks(mer, crs=cart.crs.PlateCarree())
        ax.set_yticks(par, crs=cart.crs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
        
        ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=cart.crs.PlateCarree())
    
        # Find index where data is splitted for mapping
        split_lon_idx = round(x.shape[1]/(360/(lon_0 if lon_0>0 else lon_0+360)))


        p = m.pcolormesh(x[:,:split_lon_idx], y[:,:split_lon_idx], field[:,:split_lon_idx],
                         vmax=vmax, vmin=vmin, transform=cart.crs.PlateCarree(), zorder=1, **plt_kwargs)
        p = m.pcolormesh(x[:,split_lon_idx:], y[:,split_lon_idx:], field[:,split_lon_idx:],
                         vmax=vmax, vmin=vmin, transform=cart.crs.PlateCarree(), zorder=2, **plt_kwargs)
        
                
        gl=ax.gridlines(crs=cart.crs.PlateCarree(), linewidth=0.5, color='black', alpha=0.6, linestyle='-.', zorder=10)
        gl.xlocator = mticker.FixedLocator(mer)
        gl.ylocator = mticker.FixedLocator(par)
        
        #ax.set_facecolor('grey')

        #m.add_feature(cart.feature.LAND, facecolor='0.5', zorder=3)
        #m.add_feature(cart.feature.COASTLINE,linewidth=0.5, zorder=15)
        label = ''
        if da.name is not None:
            label = da.name
        if 'units' in da.attrs:
            label += ' [%s]' % da.attrs['units']
        orient = 'vertical'
        if np.abs(bnds[1] - bnds[0]) > (np.abs(bnds[3] - bnds[2]) - 10):
            orient = 'horizontal'
        cb = plt.colorbar(p, fraction=0.07, pad=0.1, label=label, orientation=orient)
        
        return m, ax
