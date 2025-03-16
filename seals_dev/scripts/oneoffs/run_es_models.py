import os
import shutil
import yaml
import geopandas as gpd
import pygeoprocessing.geoprocessing as geo
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling

import natcap.invest.sdr.sdr
import time


def slice_inputs(base_raster, source_raster_dict, aoi_vector, target_folder, huc4= 'None'):
    """
    Assumes that `source_raster_dict` will be keyed as new_name: orig_path
    """
    base_lulc_info = geo.get_raster_info(base_raster)
    
    src_paths = []
    dst_paths = []
    dst_dict = {}
    for k, v in source_raster_dict.items():
        if os.path.splitext(v)[1] in [".tif", ".bil", ".img"]:
            src_paths.append(v)
            dst_paths.append(os.path.join(target_folder, f'{k}.tif'))
            dst_dict[k] = os.path.join(target_folder, f'{k}.tif')
    
    try:
        raster_align_index = int([i for i, x in enumerate(src_paths) if x == source_raster_dict[f'dem_7755']][0])
    except:
        raster_align_index = None

    geo.align_and_resize_raster_stack(
        base_raster_path_list=src_paths,
        target_raster_path_list=dst_paths,
        resample_method_list=['near' for _ in src_paths],
        target_pixel_size=base_lulc_info['pixel_size'],
        bounding_box_mode="intersection",
        base_vector_path_list=[aoi_vector],
        raster_align_index=raster_align_index,
        vector_mask_options={'mask_vector_path': aoi_vector},
    )
    return dst_dict

def reproject_7755(raster_path, target_raster_path, dst_crs='EPSG:7755'):
    '''
    This function reprojects the raster to EPSG:7755 or any given crs.
    This helper function is used in data_preparation function.
    '''
    with rio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            "compress": "LZW"
        })
        if src.crs != dst_crs:
            with rio.open(target_raster_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
        else:
            shutil.copy(raster_path, target_raster_path)

def fix_watersheds(watersheds_path, aoi_path, unique_id_field):
    '''
    This function fixes the watersheds shapefile by:
     1) clipping the watershed file to aoi and then
     2) exploding multipart polygons and keeping the largest area polygon for each unique id.
    TODO: The fix might delete relative large polygons if they are not the largest in the multipart polygon. Need to discuss this.
    Do we implement a filter based on area?
    '''
    watersheds_gdf = gpd.read_file(watersheds_path)
    if watersheds_gdf.crs==None:
        watersheds_gdf.crs = "EPSG:4326"
    if watersheds_gdf.crs != "EPSG:7755":
        watersheds_gdf = watersheds_gdf.to_crs(7755)

    aoi_gdf = gpd.read_file(aoi_path)
    clipped_gdf = gpd.clip(watersheds_gdf, aoi_gdf, keep_geom_type=False)
    clipped_wtershed = watersheds_path.replace(".shp", "_IND_clipped.shp")
    clipped_gdf.to_file(clipped_wtershed, driver='ESRI Shapefile')

    watersheds_gdf = clipped_gdf.explode(index_parts=True)
    watersheds_gdf = watersheds_gdf.loc[watersheds_gdf.geometry.geometry.type=='Polygon']       # Only keep Polygons
    watersheds_gdf["area"] = watersheds_gdf['geometry'].area
    multipart_wtershed = watersheds_path.replace(".shp", "_IND_multipart.shp")
    watersheds_gdf.to_file(multipart_wtershed, driver='ESRI Shapefile')
    watersheds_gdf = watersheds_gdf.merge(watersheds_gdf.groupby([unique_id_field])['area'].max().reset_index(), on=[unique_id_field, 'area'], how='right')
    fixed_watersheds = watersheds_path.replace(".shp", "_IND_fixed.shp")
    watersheds_gdf.to_file(fixed_watersheds, driver='ESRI Shapefile')

def data_preparation(seal_lulc_4326):
    '''
    This function prepares the data for the ecosystem services models.
    You need to run this function onece to prepare the data.
    I did not give any argrument to this function because I want to run this function only once.'''

    source_raster_dict = {}
    source_raster_dict[f"dem_con"]= "D:/MANAGE/CommonInputs/DEMHydroSHED3s/as_con_3s/as_con_3s.tif"
    source_raster_dict[f"lulc"]= seal_lulc_4326
    source_raster_dict[f"precipitation_annual"]= "D:/NDR/CommonInputs/MonthlyPrecipitation2021/WorldClim/wc2.1_30s_prec_annual.tif"
    source_raster_dict[f"k_factor"]= "D:/NDR/CommonInputs/ISRIC/global_soil_erodibility.tif"
    source_raster_dict[f"rainfall_erosivity"]= "D:/NDR/CommonInputs/GlobalRFactor/GlobalR/GlobalR_NoPol.tif"

    raster_folder = "D:/MANAGE/CommonInputs/RasterInputs"
    aoi_buffer_4326 = "D:/MANAGE/CommonInputs/aoi/aoi_IND_buffer.shp"

    for k, v in source_raster_dict.items():
        single_dict = {k: v}
        base_raster = v
        slice_inputs(base_raster, single_dict, aoi_buffer_4326, raster_folder)
        unprojected_raster = os.path.join(raster_folder, f'{k}.tif')
        projected_raster = os.path.join(raster_folder, f'{k}_7755.tif')
        reproject_7755(unprojected_raster, projected_raster, dst_crs='EPSG:7755')

    # Get other resolution of dem
    dem_path = os.path.join(raster_folder, "dem_con_7755.tif")
    dem_resolution(dem_path, resolution=300)
    dem_resolution(dem_path, resolution=1000)
    
    # Fix watersheds
    watersheds_path = "D:/MANAGE/CommonInputs/watersheds/hybas_as_lev06_v1c.shp"
    aoi_path = "D:/MANAGE/CommonInputs/aoi/aoi_IND.shp"
    fix_watersheds(watersheds_path, aoi_path=aoi_path, unique_id_field="SORT")

def dem_resolution(dem_path, resolution=300):
    '''
    This function resamples the dem to the given resolution.
    '''
    with rio.open(dem_path) as dataset:
        scale_factor_x = dataset.res[0]/resolution
        scale_factor_y = dataset.res[1]/resolution

        profile = dataset.profile.copy()
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor_y),
                int(dataset.width * scale_factor_x)
            ),
            resampling=Resampling.cubic_spline
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (1 / scale_factor_x),
            (1 / scale_factor_y)
        )
        profile.update({"height": data.shape[-2],
                        "width": data.shape[-1],
                    "transform": transform})

    target_dem_path = dem_path.replace('.tif', f"_{resolution}.tif")
    with rio.open(target_dem_path, "w", **profile) as dataset:
        dataset.write(data)


def sdr_run(config):
    '''
    This function runs the SDR model with given args.
    '''
    with open(config) as yaml_data_file:
        sdr_args = yaml.load(yaml_data_file, Loader=yaml.FullLoader)
    natcap.invest.sdr.sdr.execute(sdr_args)
    print(f"==============================================SDR finished======================================")
    
if __name__ == "__main__":
    star_time = time.time()
    seal_lulc_4326 = "C:/Users/salmamun/Files/seals/projects/manage_ind_boundary/intermediate/fine_processed_inputs/lulc/esa/seals7/lulc_esa_seals7_2017.tif"
    # data_preparation(seal_lulc_4326)          # Run this function to prepare the data. Run it only once.
    config  = "C:/Users/salmamun/Files/seals/seals_dev/scripts/ind_sdr_sm.yaml"
    sdr_run(config)
    print(f"Total time taken: {time.time() - star_time} seconds")