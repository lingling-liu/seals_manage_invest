import os
import xarray as xr 
import rioxarray as rio 

path = 'C:/Users/salmamun/Files/base_data/crops/earthstat/22491997/CROPGRIDSv1.08_NC_maps/CROPGRIDSv1.08_NC_maps'
output_path = "C:/Users/salmamun/Files/base_data/crops/earthstat/crop_production"
files = os.listdir(path)
crops = [f.split(".")[1].split("_")[1] for f in files if f.endswith('.nc') and f!='Countries_2018.nc']

for cr in crops:
    crop_nc_data = xr.open_dataset(os.path.join(path, f'CROPGRIDSv1.08_{cr}.nc'))
    crop_band = crop_nc_data['qual']
    crop_band = crop_band.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    crop_band.rio.write_crs("epsg:4326", inplace=True)
    output_path_cr = os.path.join(output_path, cr)
    if not os.path.exists(output_path_cr):
        os.makedirs(output_path_cr)
    crop_band.rio.to_raster(os.path.join(output_path_cr, f"{cr}_Production.tif"))

# print(crop_nc_data.variables.keys())




