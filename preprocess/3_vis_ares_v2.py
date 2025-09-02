import cartopy.crs as ccrs
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.plot(lon, lat, marker='o', color='red', transform=ccrs.Geodetic())
plt.show()
