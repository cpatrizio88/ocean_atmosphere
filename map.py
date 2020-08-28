import numpy as np
from matplotlib import pyplot as plt
import cartopy as cart
import matplotlib.ticker as mticker
#from matplotlib import ticker
from matplotlib.colors import LogNorm
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

class Mapper:

#    def __init__(self):
    
    #logscale is a parameter that is used to fix labeling for symmetric log colorbar, which for now is used by default

        
    def __call__(self, field, ct=False, logscale=True, log=False, ax=None, bnds=[0,360,-90,90], title='', units='', cbfrac=0.11,projection_name='PlateCarree', **plt_kwargs):
        
         # Central longitude must be between -180 and 180 (greenwich meridian is 0) 
        #lon_0 = (bnds[0] + bnds[1])/2 - 360
        lon_0 = (bnds[0] + bnds[1])/2 
        
        #print('lon_0', lon_0)
        
        if projection_name == 'PlateCarree':
            proj = cart.crs.PlateCarree(central_longitude=lon_0)
        elif projection_name == 'Robinson':
            proj = cart.crs.Robinson(central_longitude=lon_0)
        elif projection_name == 'AlbersEqualArea':
            proj = cart.crs.AlbersEqualArea(central_longitude=lon_0)
        else:
            print('Projection name is invalid.')
            
        if projection_name == 'PlateCarree':
            proj2 = cart.crs.PlateCarree()
        elif projection_name == 'Robinson':
            proj2 = cart.crs.Robinson()
        elif projection_name == 'AlbersEqualArea':
            proj2 = cart.crs.AlbersEqualArea()
        else:
            print('Projection name is invalid.')




        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        #print(self.orig_grid.shape)
        #print(da.values.shape)
        #print(self.new_grid.shape)



        vmax = plt_kwargs.pop('vmax', field.max())
        vmin = plt_kwargs.pop('vmin', field.min())

        m = plt.axes(projection=proj)
        x,y = field.lon, field.lat
        
        #ax= plt.gca()

                    
        pardiff = 30.
        merdiff = 60.
        
        if np.abs(bnds[1] - bnds[0]) < 180:
            merdiff = 30.
        #if np.abs(bnds[1] - bnds[0]) < 90:
        #    merdiff = 15.
        #if np.abs(bnds[3]- bnds[2]) < 90:
        #    pardiff = 15.
            
        par = np.arange(-90.,90.+pardiff,pardiff)
        mer = np.arange(-180.,180.+merdiff,merdiff)
            
        ax=plt.gca()
        ax.set_xticks(mer, crs=proj2)
        ax.set_yticks(par, crs=proj2)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
        
        ax.set_extent((bnds[0], bnds[1], bnds[2], bnds[3]), crs=proj2)
    
        # Find index where data is splitted for mapping
        #split_lon_idx = round(x.shape[1]/(360/(lon_0 if lon_0>0 else lon_0+360)))
        #norm=LogNorm(vmin=vmin, vmax=vmax)
        
        lognorm=LogNorm(vmin=vmin, vmax=vmax)
        
        if log:
            p = m.pcolormesh(x, y, field,
                         vmax=vmax, vmin=vmin, norm=lognorm, transform=proj2, zorder=1, **plt_kwargs)
        else:
            p = m.pcolormesh(x, y, field,
                         vmax=vmax, vmin=vmin, transform=proj2, zorder=1, **plt_kwargs)
        #p = m.pcolormesh(x[:,split_lon_idx:], y[:,split_lon_idx:], field[:,split_lon_idx:],
        #                 vmax=vmax, vmin=vmin, transform=cart.crs.PlateCarree(), zorder=2, **plt_kwargs)
 
        if ct:
            ctstep = np.abs(vmax-vmin)/20
            ctlevels = np.arange(vmin,vmax+ctstep, ctstep)
            ct=plt.contour(x, y, field, colors='k', levels=ctlevels, linewidths=1,  transform=proj2)
            if np.any(np.round(ct.levels, 5) == 0):
                ct.collections[np.where(np.round(ct.levels, 5) == 0)[0][0]].set_linewidth(0)
                #ax.clabel(ct, fontsize=9, inline=1, fmt='%1.1f')
                for line in ct.collections:
                    if line.get_linestyle() != [(None, None)]:
                        line.set_linestyle([(0, (8.0, 8.0))])
                
        gl=ax.gridlines(crs=proj2, linewidth=0.5, color='black', alpha=0.6, linestyle='-.', zorder=10)
        gl.xlocator = mticker.FixedLocator(mer)
        gl.ylocator = mticker.FixedLocator(par)
        
        #ax.set_facecolor('grey')

        m.add_feature(cart.feature.LAND, edgecolor='k', facecolor='grey', zorder=3)
        #m.add_feature(cart.feature.COASTLINE,linewidth=0.5, zorder=15)
        plt.title(title)
        orient = 'vertical'
        if np.abs(bnds[1] - bnds[0]) > (np.abs(bnds[3] - bnds[2]) - 10):
            orient = 'horizontal'

        cb=plt.colorbar(p, label=units,fraction=cbfrac, pad=0.11, orientation=orient)
        
        #cb.set_ticks([-0.4,1.0,1.6,2.0])
        #cb.ax.set_ticklabels([-0.4,1.0,1.6,2.0])
        if logscale: 
            if vmin >= 0:
                ticklabels = [vmin,0.1,1.0,vmax]
            else:
                #ticklabels = [vmin,-0.01,0,0.01,vmax]
                #ticklabels = [vmin,-0.1,0,0.1,vmax]
                ticklabels = [vmin,-0.1,0,0.1,vmax]
                #ticklabels = [vmin,-1.0,-0.1,0.1,1.0,vmax]
            cb.set_ticks(ticklabels)
            cb.ax.set_xticklabels(ticklabels)
        cb.ax.tick_params(labelsize=24)
        
        return m, ax
