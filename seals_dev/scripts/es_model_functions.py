import os
import itertools
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import pygeoprocessing.geoprocessing as geo
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import time
from osgeo import gdal
gdal.UseExceptions()
from scripts.pollination_shocks import calculate_crop_value_and_shock
import scripts.sdr_local as sdr_local
from scripts.pollination_biophysical_model import pollination_sufficiency
import csv
import hazelbean as hb
import seals_utils
import logging
import shapely
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

def es_options(p):
    p.aoi_buffer_distance = 5000        # buffer distance around the aoi in meters
    p.force_es_data_preparation=False       # If True, it will force the data preparation step. Otherwise, it will skip if the files are already present.
    p.process_common_inputs_first_time = True
    p.dem_resolution = 300      # Set the DEM resolution for the SDR model.
    # Define where additional data is stored
    p.common_input_folder = os.path.join(p.base_data_dir, 'seals', 'es_common_inputs')

def es_models(p):
    ''''
    function to create directories for ecosystem services models.
    '''
    pass

def pollination_results(p):
    pollination_biophysical(p)
    pollination_economic(p)
def sdr_results(p):
    logger.info("Preparing data for SDR model")
    sdr_data_preparation(p)
    logger.info("Running SDR model")
    run_sdr(p)
    logger.info("Calculating LPL shock")
    lpl_shock(p)

def slice_inputs(base_raster, source_raster_dict, aoi_vector, target_folder):
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

def reproject_raster(raster_path, target_raster_path, dst_crs='EPSG:7755'):
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

def fix_watersheds(watersheds_path, aoi_path, aoi_label, unique_id_field, projected_crs, output_folder=None):
    '''
    This function fixes the watersheds shapefile by:
     1) clipping the watershed file to aoi and then
     2) exploding multipart polygons and keeping the largest area polygon for each unique id.
    TODO: The fix might delete relative large polygons if they are not the largest in the multipart polygon. Need to discuss this.
    Do we implement a filter based on area?
    '''
    # watersheds_path = "C:/Users/salmamun/Files/base_data/seals/es_common_inputs/watersheds_level_7_valid.gpkg"
    # aoi_path = "C:/Users/salmamun/Files/seals/projects/Tanzania/intermediate/project_aoi_gtap_r251/aoi_buffer/aoi_TZA_32736.gpkg"
    # aoi_path = "C:/Users/salmamun/Files/seals/projects/Tanzania/intermediate/project_aoi_gtap_r251/aoi_TZA.gpkg"
    # aoi_label = "TZA"
    # unique_id_field = "SORT"
    # projected_crs = 32736
    # output_folder = 'C:\\Users\\salmamun\\Files\\seals\\projects\\Tanzania\\intermediate\\es_models\\sdr_results\\es_clipped_inputs'
    
    if projected_crs==54030:
        crs_string = "ESRI"
    else:
        crs_string = "EPSG"
    watersheds_gdf = gpd.read_file(watersheds_path)
        
    if output_folder==None:
        output_folder = os.path.dirname(watersheds_path)
    clipped_watershed = os.path.join(output_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_clipped.gpkg"))
    if not hb.path_exists(clipped_watershed):
        aoi_gdf = gpd.read_file(aoi_path)

        # # CRS match check
        # if watersheds_gdf.crs != aoi_gdf.crs:
        #     aoi_gdf = aoi_gdf.to_crs(watersheds_gdf.crs)

        # # Unified AOI geometries
        # aoi_union = shapely.union_all(aoi_gdf.geometry)

        # # Perform clipping
        # clipped_gdf = watersheds_gdf.clip(aoi_union, keep_geom_type=False)

        clipped_gdf = watersheds_gdf.clip(aoi_gdf, keep_geom_type=False)
        # Change clipped_gdf crs to projected_crs
        if clipped_gdf.crs==None:
            clipped_gdf.crs = "EPSG:4326"
        if clipped_gdf.crs != f"{crs_string}:{projected_crs}":
            clipped_gdf = clipped_gdf.to_crs(f"{crs_string}:{projected_crs}")
        clipped_gdf.to_file(clipped_watershed, driver='GPKG')
    else:
        clipped_gdf = gpd.read_file(clipped_watershed)

    multipart_watershed = os.path.join(output_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_multipart.gpkg"))
    if not hb.path_exists(multipart_watershed):
        multipart_gdf = clipped_gdf.explode(index_parts=True)
        multipart_gdf = multipart_gdf.loc[multipart_gdf.geometry.geometry.type=='Polygon']       # Only keep Polygons
        multipart_gdf["area"] = multipart_gdf['geometry'].area
        multipart_gdf.to_file(multipart_watershed, driver='GPKG')
    else:
        multipart_gdf = gpd.read_file(multipart_watershed)

    fixed_watersheds = os.path.join(output_folder, os.path.basename(watersheds_path).replace(".gpkg", f"_{aoi_label}_fixed.gpkg"))
    if not hb.path_exists(fixed_watersheds):
        fixed_gdf = multipart_gdf.merge(multipart_gdf.groupby([unique_id_field])['area'].max().reset_index(), on=[unique_id_field, 'area'], how='right')
        fixed_gdf.to_file(fixed_watersheds, driver='GPKG')

def create_aoi_buffer(p):
    '''
    This function creates a buffer around the aoi shapefile.
    The buffer distance is defined in the parameters file.
    '''    
    aoi_gdf = gpd.read_file(p.aoi_path)
    aoi_projected_gdf = aoi_gdf.to_crs(f"{p.crs_string}:{p.projected_crs}")
    # aoi_projected_gdf = aoi_gdf.to_crs(f"ESRI:{p.projected_crs}")

    aoi_projected_gdf.to_file(p.aoi_projected_path, driver='GPKG')
    aoi_projected_gdf['geometry'] = aoi_projected_gdf.buffer(p.aoi_buffer_distance)
    aoi_buffer_gdf = aoi_projected_gdf.to_crs(4326)      # converting back to 4326 for saving the file.
    aoi_buffer_gdf.to_file(p.aoi_buffer_path, driver='GPKG')
    return p.aoi_buffer_path

def sdr_data_preparation(p):
    '''
    This function prepares the data for the ecosystem services models.
    You need to run this function onece to prepare the data.'''
    es_options(p)
    p.es_clipped_input_folder = os.path.join(p.sdr_results_dir, 'es_clipped_inputs')
    clipped_uprojected_raster_folder = os.path.join(p.es_clipped_input_folder, 'unprojected')
    p.clipped_projected_raster_folder = os.path.join(p.es_clipped_input_folder, 'projected')
    hb.create_directories([p.es_clipped_input_folder, clipped_uprojected_raster_folder, p.clipped_projected_raster_folder])
    logger.info(f"Clipping and projecting the rasters for the SDR model.")
    source_raster_dict = {}
    if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                # Build a dict for the lulc labels
                labels_dict = dict(zip(p.all_class_indices, p.all_class_labels))
               
                # this acountcs for the fact that the way the correspondence is loaded is not necessarily in the numerical order
                indices_to_labels_dict = dict(sorted(labels_dict.items()))

                for year in p.years:
                    if p.scenario_type ==  'baseline':
                        seal_lulc_name = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + str(p.key_base_year) + '.tif'
                        seal_lulc_path = os.path.join(p.fine_processed_inputs_dir, "lulc", p.lulc_src_label,  p.lulc_simplification_label, seal_lulc_name)
                    else:
                        seal_lulc_name = 'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label + '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif'
                        seal_lulc_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, seal_lulc_name)
                    source_raster_dict[f"lulc_{p.scenario_type}_{year}"]= seal_lulc_path
     
    
    # Add common input rasters
    if p.process_common_inputs_first_time:
        source_raster_dict[f"dem_con"]= os.path.join(p.common_input_folder, "global_dem_10as.tif")
        source_raster_dict[f"k_factor"]= os.path.join(p.common_input_folder, "global_soil_erodibility.tif")
        source_raster_dict[f"rainfall_erosivity"]= os.path.join(p.common_input_folder, "GlobalR_NoPol.tif")

    aoi_buffer_folder = os.path.join(os.path.dirname(p.aoi_path), "aoi_buffer")
    hb.create_directories([aoi_buffer_folder])
    p.aoi_projected_path = os.path.join(aoi_buffer_folder, os.path.basename(p.aoi_path.replace(".gpkg", f"_{p.projected_crs}.gpkg")))
    p.aoi_buffer_path = os.path.join(aoi_buffer_folder, os.path.basename(p.aoi_path.replace(".gpkg", "_buffer.gpkg")))
    if not hb.path_exists(p.aoi_buffer_path) or not hb.path_exists(p.aoi_projected_path):
        p.aoi_buffer_path = create_aoi_buffer(p)
    
    for k, v in source_raster_dict.items():
        single_dict = {k: v}
        base_raster = v
        unprojected_raster = os.path.join(clipped_uprojected_raster_folder, f'{k}.tif')
        if not hb.path_exists(unprojected_raster) or p.force_es_data_preparation:
            if hb.path_exists(base_raster):
                slice_inputs(base_raster, single_dict, p.aoi_buffer_path, clipped_uprojected_raster_folder)
        
        projected_raster = os.path.join(p.clipped_projected_raster_folder, f'{k}_{p.projected_crs}.tif')
        if not hb.path_exists(projected_raster) or p.force_es_data_preparation:
            if hb.path_exists(unprojected_raster):
                reproject_raster(unprojected_raster, projected_raster, dst_crs=f"{p.crs_string}:{p.projected_crs}")

    if p.process_common_inputs_first_time:
        # Get other resolution of dem
        dem_path = os.path.join(p.clipped_projected_raster_folder, f"dem_con_{p.projected_crs}.tif")
        p.scaled_dem_path = dem_path.replace('.tif', f"_{p.dem_resolution}.tif")
        if not hb.path_exists(p.scaled_dem_path) or p.force_es_data_preparation:
            dem_resolution(dem_path, resolution=p.dem_resolution)
        
        # Fix watersheds
        logger.info(f"Fixing watersheds")
        p.watersheds_path = os.path.join(p.common_input_folder, "watersheds_level_7_valid.gpkg")
        p.fixed_watersheds_path = os.path.join(p.es_clipped_input_folder, os.path.basename(p.watersheds_path).replace(".gpkg", f"_{p.aoi_label}_fixed.gpkg"))
        if not hb.path_exists(p.fixed_watersheds_path):
            if hb.path_exists(p.watersheds_path):
                fix_watersheds(watersheds_path=p.watersheds_path, aoi_path=p.aoi_path, aoi_label=p.aoi_label, unique_id_field="SORT", 
                            projected_crs = p.projected_crs, output_folder=p.es_clipped_input_folder)
        # Copy biophysical table
        logger.info(f"Copying biophysical table")
        sdr_biophysical_table_source = os.path.join(p.common_input_folder, 'sdr_parameter_table_onil.csv')
        p.sdr_biophyscal_table = os.path.join(p.es_clipped_input_folder, 'sdr_biophysical_table.csv')
        if not hb.path_exists(p.sdr_biophyscal_table):
            shutil.copy(sdr_biophysical_table_source, p.sdr_biophyscal_table)

        # reproject area raster to projected crs. Do this once. We will need this for LPL calculations.
        area_4326 = os.path.join(p.project_aoi_gtap_r251_dir, "pyramids", "aoi_ha_per_cell_fine.tif")
        p.area_projected_path = os.path.join(p.clipped_projected_raster_folder, f"area_ha_per_cell_fine_{p.projected_crs}.tif")
        if not hb.path_exists(p.area_projected_path):
            if hb.path_exists(area_4326):
                reproject_raster(area_4326, p.area_projected_path, dst_crs = f'{p.crs_string}:{p.projected_crs}')

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

def prepare_common_sdr_config(p):
    sdr_args = {}
    sdr_args["workspace_dir"] = os.path.join(p.sdr_results_dir, f"output_dem_{p.dem_resolution}")
    # result_sufix will be set based on lulc scenario

    # The following are the raster needed for models
    # lulc path is also set based on lulc scenario
    sdr_args["dem_path"] = os.path.join(p.clipped_projected_raster_folder, f"dem_con_{p.projected_crs}_{p.dem_resolution}.tif")
    sdr_args["precipitation_raster_path"] =  os.path.join(p.clipped_projected_raster_folder, f"precipitation_annual_{p.projected_crs}.tif")
    sdr_args["erodibility_path"] =  os.path.join(p.clipped_projected_raster_folder, f"k_factor_{p.projected_crs}.tif")
    sdr_args["erosivity_path"] =  os.path.join(p.clipped_projected_raster_folder, f"rainfall_erosivity_{p.projected_crs}.tif")

    # The following are the vector needed for models
    sdr_args["watersheds_path"] = p.fixed_watersheds_path    

    # The following are the parameters tables
    sdr_args["biophysical_table_path"] = p.sdr_biophyscal_table

    # The following are specific parameters for the models
    sdr_args["threshold_flow_accumulation"] = 75  # Taken from Onil - 75
    sdr_args["k_param"] = 2
    sdr_args["sdr_max"] = 0.8       # DEFAULT_SDR_SDR_MAX
    sdr_args["ic_0_param"] = 0.5        # DEFAULT_SDR_IC_0_PARAM
    sdr_args["l_max"] = 122         # DEFAULT_SDR_L_MAX
    sdr_args["drainage_path"] = ''  # This parameter is optional

    # Resource management parameters
    sdr_args["n_workers"] = p.num_workers     # Provide the number of cores to use.
    return sdr_args

def run_sdr(p):
    '''
    This function runs the SDR model with given args.
    '''
    
    sdr_args = prepare_common_sdr_config(p)
    p.usle_results_path = {}
    p.lulc_projected_path = {}
    if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)

                # Build a dict for the lulc labels
                for year in p.years:
                    sdr_args["results_suffix"] = f"{p.scenario_type}_{year}"
                    sdr_args["lulc_path"] = os.path.join(p.clipped_projected_raster_folder, f"lulc_{p.scenario_type}_{year}_{p.projected_crs}.tif")
                    p.lulc_projected_path[f"lulc_{p.scenario_type}_{year}"] = sdr_args["lulc_path"]
                    # Run the model. Implemented a shortcut to avoid running the model if the results (usle) already exist.
                    p.usle_results_path[f"usle_{p.scenario_type}_{year}"] = os.path.join(sdr_args["workspace_dir"], f"usle_{p.scenario_type}_{year}.tif")
                    if not hb.path_exists(p.usle_results_path[f"usle_{p.scenario_type}_{year}"]):
                        sdr_local.execute(sdr_args)

def get_errosion_by_ha(aligned_rasters, output_folder, binary_threshold=False, **kwargs) -> dict:
    '''
    This function calculates the sediment export by hectare.
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    se_nodata = geo.get_raster_info(aligned_rasters['usle'])['nodata'][0]
    out_dict = {}
    if binary_threshold:
        key = 'usle_binary'
        source_nodata = 8
        data_type = gdal.GDT_Byte
    else:
        key = 'usle_by_ha'
        source_nodata = se_nodata
        data_type = gdal.GDT_Float32

    resolution = kwargs.get('resolution')
    seal_scenario = kwargs.get('seal_scenario')
    target_file = os.path.join(output_folder, f'{key}_{resolution}_{seal_scenario}.tif')
    out_dict[key] = target_file
    def _usle_by_ha(se, ar):
        result = se.copy()
        valid_mask = (se != se_nodata) & (ar != -9999)
        result[valid_mask] = se[valid_mask]/ar[valid_mask]
        if binary_threshold:
            result[valid_mask] = np.where(result[valid_mask] >= binary_threshold, 1, 0)
            result[~valid_mask] = source_nodata    
        return result
        
    geo.raster_calculator(
        [(aligned_rasters['usle'], 1), (aligned_rasters['area'], 1)],
        _usle_by_ha,
        target_file,
        data_type,
        source_nodata
    )
    return out_dict

def get_lpl_shock(aligned_rasters, output_folder, resolution, seal_scenario, binary_threshold=11, lpl_factor=0.08) -> None:
    '''
    This function calculates the sediment export by hectare.
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    usle_binary = get_errosion_by_ha(aligned_rasters, output_folder, 
                                     resolution=resolution, seal_scenario=seal_scenario, 
                                     binary_threshold=binary_threshold)
    errosion_arr = rio.open(usle_binary['usle_binary']).read(1).ravel()
    area_arr = rio.open(aligned_rasters['area']).read(1).ravel()
    lulc_arr = rio.open(aligned_rasters['lulc']).read(1).ravel()

    area_seb_ag = area_arr[np.all([errosion_arr == 1, lulc_arr==2], axis=0)].sum()       # 2 is cropland
    area_ag = area_arr[np.all([area_arr != -9.99900000e+03, lulc_arr==2], axis=0)].sum()

    lpl_shock = area_seb_ag*lpl_factor/area_ag
    return lpl_shock

def align_raster(base_raster: str,
                 source_raster_dict: dict,
                 aoi_vector: str,
                 target_folder: str) -> dict:
    '''
    This function aligns the raster to the base raster.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    base_lulc_info = geo.get_raster_info(base_raster)
    
    src_paths = []
    dst_paths = []
    dst_dict = {}
    for k, v in source_raster_dict.items():
        if os.path.splitext(v)[1] in [".tif", ".bil", ".img"]:
            src_paths.append(v)
            dst_paths.append(os.path.join(target_folder, f'{k}.tif'))
            dst_dict[k] = os.path.join(target_folder, f'{k}.tif')
    

    geo.align_and_resize_raster_stack(
        base_raster_path_list=src_paths,
        target_raster_path_list=dst_paths,
        resample_method_list=['near' for _ in src_paths],
        target_pixel_size=base_lulc_info['pixel_size'],
        bounding_box_mode="intersection",
        base_vector_path_list=[aoi_vector],
        raster_align_index=0,
        vector_mask_options={'mask_vector_path': aoi_vector}
    )
    return dst_dict

def lpl_shock(p):
    p.lpl_shock_all = pd.DataFrame(columns=["scenario", "scenario_type","resolution", "year", "avoided erosion"])
    p.lpl_output_folder = os.path.join(p.sdr_results_dir, "shocks")
    hb.create_directories([p.lpl_output_folder])
    p.lpl_shock_csv_path = os.path.join(p.sdr_results_dir, "lpl_shock.csv")
    if p.scenario_definitions_path is not None:
            p.scenarios_df = pd.read_csv(p.scenario_definitions_path)
            
            for index, row in p.scenarios_df.iterrows():
                seals_utils.assign_df_row_to_object_attributes(p, row)
                seals_utils.set_derived_attributes(p)
                for year in p.years:
                    align_raster_dict = {
                        "usle": p.usle_results_path[f"usle_{p.scenario_type}_{year}"],
                        "area": p.area_projected_path,
                        "lulc": p.lulc_projected_path[f"lulc_{p.scenario_type}_{year}"]
                    }
                    aligned_raster_folder = os.path.join(p.sdr_results_dir, "aligned_rasters", f"aligned_rasters_{p.scenario_type}_{year}_{p.dem_resolution}")
                    hb.create_directories([aligned_raster_folder])
                    
                    aligned_rasters = {k: os.path.join(aligned_raster_folder, f"{k}.tif") for k, v in align_raster_dict.items()}
                    if not hb.path_all_exist(list(aligned_rasters.values())):
                        aligned_rasters = align_raster(align_raster_dict["usle"],
                                                    align_raster_dict, p.aoi_projected_path, aligned_raster_folder)
            
                    # Calculate LPL shocks
                    get_errosion_by_ha(aligned_rasters, p.lpl_output_folder, resolution=p.dem_resolution, 
                                    seal_scenario=p.scenario_type, binary_threshold=False)        
                    # The above is not neccessary for lpl_shocks. Just created those to see usle by ha.
                    lpl_shock = get_lpl_shock(aligned_rasters, p.lpl_output_folder, resolution=p.dem_resolution, 
                                            seal_scenario=p.scenario_type, binary_threshold=11, lpl_factor=0.08)
                    p.lpl_shock_all.loc[len(p.lpl_shock_all)] = [p.scenario_label, p.scenario_type, p.dem_resolution, year, lpl_shock]
    
    p.lpl_shock_all.to_csv(p.lpl_shock_csv_path, index=False)

def pollination_biophysical(p):
    print("start")
    if p.run_this:  
        p.pollination_sufficiency_baseline = {}
        p.pollination_sufficiency_scenario = {}             
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            hb.log('Calculating pollination sufficiency on ' + str(index) + ' of ' + str(len(p.scenarios_df)))
            if p.scenario_type == 'baseline':
                
                for year in p.base_years:
                    current_lulc_path = os.path.join(p.fine_processed_inputs_dir, 'lulc/esa/seals7','lulc_esa_seals7_' + str(p.base_years[0]) + '.tif')
                    current_lulc_bb = hb.get_bounding_box(current_lulc_path)
                    pollination_sufficiency_output_path = os.path.join(p.cur_dir, 'pollination_sufficiency_' + p.scenario_type + '_' + str(p.base_years[0]) + '.tif')
                    if not hb.path_exists(pollination_sufficiency_output_path):
                        hb.log('Running global_invest_main.make_poll_suff on LULC: ' + str(current_lulc_path) + ' and saving results to ' + str(pollination_sufficiency_output_path))
                        # pollination_sufficiency(current_lulc_path, pollination_sufficiency_output_path)
                        pollination_sufficiency(current_lulc_path, pollination_sufficiency_output_path)
                    p.pollination_sufficiency_baseline[f"{p.scenario_type}_{year}"] = pollination_sufficiency_output_path        # Saleh: Saving the path to the object for later use
            elif p.scenario_type != 'baseline':
    
                for year in p.years:
                    current_lulc_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, 
                                                     'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label +
                                                       '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                    current_lulc_bb = hb.get_bounding_box(current_lulc_path)
                    pollination_sufficiency_output_path = os.path.join(p.cur_dir, 'pollination_sufficiency_' + p.scenario_label + '_' + str(p.years[0]) + '.tif')
                    if not hb.path_exists(pollination_sufficiency_output_path):
                        hb.log('Running global_invest_main.make_poll_suff on LULC: ' + str(current_lulc_path) + ' and saving results to ' + str(pollination_sufficiency_output_path))
                        #pollination_sufficiency(current_lulc_path, pollination_sufficiency_output_path)
                        pollination_sufficiency(current_lulc_path, pollination_sufficiency_output_path)
                    p.pollination_sufficiency_scenario[f"{p.scenario_type}_{year}"] = pollination_sufficiency_output_path        # Saleh: Saving the path to the object for later use

def pollination_economic(p):
    p.crop_data_dir = os.path.join(p.common_input_folder, "pollination/crop_production")
    p.pollination_dependence_spreadsheet_input_path = os.path.join(p.common_input_folder, "pollination/rspb20141799supp3.xls")  # Note had to fix pol.dep for coffee and green broadbean as it was 25 not .25
    output_dir = os.path.join(p.cur_dir)
    p.pollination_shock_csv_path = os.path.join(p.cur_dir, 'pollination_shock.csv')

    # Initialize CSV file for shock values
    if not os.path.exists(p.pollination_shock_csv_path):
        with open(p.pollination_shock_csv_path, 'w', newline='') as shock_file:
            writer = csv.writer(shock_file)
            writer.writerow(['scenario', 'pollination supply', 'year', 'Baseline Path', 'Scenario Path'])

    if p.run_this:
        for index, row in p.scenarios_df.iterrows():
            seals_utils.assign_df_row_to_object_attributes(p, row)
            hb.log('Calculating pollination_economic on ' + str(index) + ' of ' + str(len(p.scenarios_df)))
            if p.scenario_type == 'baseline':
                for year in p.base_years:
                    lulc_baseline_path = os.path.join(p.fine_processed_inputs_dir, 'lulc/esa/seals7','lulc_esa_seals7_' + str(year) + '.tif')
                    pollination_sufficiency_baseline = p.pollination_sufficiency_baseline[f"{p.scenario_type}_{year}"] 
                    # os.path.join(os.path.dirname(p.cur_dir), 'pollination_biophysical','pollination_sufficiency_' + p.scenario_type + '_' + str(year) + '.tif') 
                    p.base_year = year                   
            elif p.scenario_type != 'baseline':
                for year in p.years:
                    lulc_scenario_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, 
                                                     'lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label + '_' + p.exogenous_label +
                                                       '_' + p.climate_label + '_' + p.model_label + '_' + p.counterfactual_label + '_' + str(year) + '.tif')
                    pollination_sufficiency_scenario = p.pollination_sufficiency_scenario[f"{p.scenario_type}_{year}"] 
                    # os.path.join(os.path.dirname(p.cur_dir), 'pollination_biophysical', 'pollination_sufficiency_' + p.scenario_label + '_' + str(year) + '.tif')                   
                    calculate_crop_value_and_shock(
                        lulc_baseline_path, lulc_scenario_path, pollination_sufficiency_baseline, pollination_sufficiency_scenario,
                        p.crop_data_dir, p.pollination_dependence_spreadsheet_input_path, output_dir, p.pollination_shock_csv_path, p.scenario_label,
                        year, p.base_year
                    )

def manage_input(p):
    p.shock_output_dir = os.path.join(p.project_dir, "output")
    hb.create_directories([p.shock_output_dir])
    # pollination_shock_df = pd.read_csv("C:/Users/salmamun/Files/seals/projects/India/intermediate/es_models/pollination_results/pollination_shock.csv")
    pollination_shock_df = pd.read_csv(p.pollination_shock_csv_path)
    # lpl_shock_df = pd.read_csv("C:/Users/salmamun/Files/seals/projects/India/intermediate/es_models/sdr_results/lpl_shock.csv")
    lpl_shock_df = pd.read_csv(p.lpl_shock_csv_path)
    manage_input_df = pd.DataFrame(columns=['reg', 'year', 'pollination supply', 'avoided erosion'])
    for index, row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p, row)
        p.lpl_baseline = lpl_shock_df.loc[(lpl_shock_df['scenario_type'] == 'baseline'), 'avoided erosion'].values[0]
        if p.scenario_type != 'baseline':
            for year in p.years:
                pollination_supply = pollination_shock_df.loc[(pollination_shock_df['scenario'] == p.scenario_label) & (pollination_shock_df['year'] == year), 'pollination supply'].values[0]
                lpl_shock_scenario = lpl_shock_df.loc[(lpl_shock_df['scenario'] == p.scenario_label) & (lpl_shock_df['year'] == year), 'avoided erosion'].values[0]
                avoided_erosion = 1- (lpl_shock_scenario - p.lpl_baseline)
                manage_input_df.loc[len(manage_input_df)] = [p.aoi_label, year, pollination_supply, avoided_erosion]

    shock_output_file = os.path.join(p.shock_output_dir, "shock_output.csv")
    if os.path.exists(shock_output_file):
        manage_shock_df = pd.read_csv(shock_output_file)
        manage_shock_df = (pd.concat([manage_shock_df, manage_input_df], ignore_index=True, sort =False).drop_duplicates(['year'], keep='last'))
    else:
        manage_shock_df = manage_input_df
    manage_shock_df.to_csv(shock_output_file, index=False)
    shutil.rmtree(p.manage_input_dir, ignore_errors=True)
