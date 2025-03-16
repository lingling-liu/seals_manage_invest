import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from hazelbean import rasterize_to_match
import pygeoprocessing.geoprocessing as geo


SEAL7_MANAGE_LANDUSE_DICT = {
    'Shrubland': 5,
    'Pastureland': 3,
    'SavnGrassLnd': 3,
    'cropLand': 2,
    'Builtupland': 1,
    'Forest': 4,
    'unManagedForest': 4,
    'Otherland': 5}

SEAL7_LANDUSE_DICT = {
    'urban': 1,
    'cropLand': 2,
    'grassland': 3,
    'forest': 4,
    'othernat': 5,
    'water': 6,
    'other': 7
    }


def reformat_aez(regional_change_csv):
    regional_change_df = pd.read_csv(regional_change_csv)
    regional_change_df.columns = ["GTAP141", "AEZ", "landuse", "year", "change_percent", "abs_change_mha"]
    for col in ["GTAP141", "AEZ", "landuse"]:
        regional_change_df[col] = regional_change_df[col].str.replace("'", '')
    regional_change_df['AEZ'] = regional_change_df['AEZ'].str.replace("f-AEZ", '')
    return regional_change_df



def get_total_by_aez_lu(regional_change_df, aez_vector_file, area_ha, seal7_lulc):
    aez = gpd.read_file(aez_vector_file)
    aez_iso = aez[aez['GTAP141'] == iso3]
    aez_iso = aez_iso.to_crs("EPSG:4326")
    aez_iso_file = os.path.join(workspace, "intermediate", "project_aoi", "pyramids", f"aoi_{iso3}.gpkg")
    aez_iso.to_file(aez_iso_file, driver='GPKG')

    aez_raster = os.path.join(workspace, "intermediate", "project_aoi", "pyramids", f"aez_{iso3}_raster.tif")
    rasterize_to_match(input_vector_path = aez_iso_file,
                       match_raster_path = seal7_lulc,
                        output_raster_path = aez_raster,
                          burn_column_name='AEZ',
                            burn_values=None,
                              datatype=5,
                                ndv=None,
                                  all_touched=False)
    area = read_raster_as_array(area_ha)
    lulc = read_raster_as_array(seal7_lulc)
    aez = read_raster_as_array(aez_raster)
    for aez_val in pd.unique(regional_change_df.AEZ):
        regional_change_df_aez = regional_change_df[regional_change_df['AEZ'] == aez_val]
        for lu_desc in pd.unique(regional_change_df.landuse):
            lulc_code = SEAL7_MANAGE_LANDUSE_DICT[lu_desc]
            lulc_mask = lulc == lulc_code
            aez_mask = aez == int(aez_val)
            area_lu_aez = area * lulc_mask * aez_mask
            regional_change_df.loc[(regional_change_df['landuse'] == lu_desc) & (regional_change_df["AEZ"] == aez_val),
                                   ["total_ha"]] = np.sum(area_lu_aez)
    
    regional_change_df["change_ha"] = regional_change_df["total_ha"]*regional_change_df["change_percent"]/100
    
    return regional_change_df

def read_raster_as_array(raster_path):
    with rio.open(raster_path) as src:
        return src.read(1)



def input_seal_format(regional_change_df):
    regional_change_df['region_label'] = regional_change_df['GTAP141'] + regional_change_df['AEZ']
    regional_change_df = regional_change_df.pivot(index='region_label', columns='landuse', values='change_ha')
    regional_change_df = regional_change_df.reset_index()
    regional_change_df = regional_change_df.fillna(0)
    return regional_change_df

def get_SEALS_lucodes(regional_change_df, SEAL7_MANAGE_LANDUSE_DICT, SEAL7_LANDUSE_DICT):
    regional_change_df['seals_code'] = regional_change_df['landuse'].map(SEAL7_MANAGE_LANDUSE_DICT) 
    SEAL7_LANDUSE_DICT = dict((v,k) for k,v in SEAL7_LANDUSE_DICT.items())
    regional_change_df['seals_desc'] = regional_change_df['seals_code'].map(SEAL7_LANDUSE_DICT)
    return regional_change_df

def collapse_to_country(regional_change_df):
    regional_change_df = regional_change_df[['abs_change_mha', 'seals_desc']].groupby('seals_desc').sum()*1000000
    regional_change_df = regional_change_df.reset_index()
    return regional_change_df

def country_input_seals_format(regional_change_df):
    regional_change_df = regional_change_df.T
    regional_change_df.columns = regional_change_df.iloc[0]
    regional_change_df = regional_change_df[1:]
    regional_change_df.insert(0, 'region_label', ['tza'])
    return regional_change_df

def get_lulc_area(lulc_raster, area_ha_raster):
    lulc = read_raster_as_array(lulc_raster)
    area = read_raster_as_array(area_ha_raster)
    lulc_area = {}
    for lulc_code in np.unique(lulc):
        if lulc_code == 0 or lulc_code == 255:
            continue
        else:
            lulc_mask = lulc == lulc_code
            lulc_area[lulc_code] = np.sum(area * lulc_mask)
    return lulc_area

def get_aligned_rasters(base_raster, aoi_vector, source_raster_dict):
    base_lulc_info = geo.get_raster_info(base_raster)
    
    src_paths = []
    dst_paths = []
    for k, v in source_raster_dict.items():
        if os.path.splitext(v)[1] in [".tif", ".bil", ".img"]:
            src_paths.append(v)
            target_raster_path = v.replace(".tif", "_aligned.tif")
            dst_paths.append(target_raster_path)
    
    geo.align_and_resize_raster_stack(
        base_raster_path_list=src_paths,
        target_raster_path_list=dst_paths,
        resample_method_list=['near' for _ in src_paths],
        target_pixel_size=base_lulc_info['pixel_size'],
        bounding_box_mode="intersection",
        base_vector_path_list=[aoi_vector],
        mask_options={"mask_vector_path": aoi_vector},
        raster_align_index=0            # Assume that first raster is the base raster
    )


if __name__=="__main__":
    iso3 = "IND"
    workspace = "C:/Users/salmamun/Files/seals/projects/manage_ind_boundary"
    aez_vector_file = "G:/Shared drives/NatCapTEEMs/Projects/WB_MANAGE_Project/GTAPLULCv10adminaez/GTAPLULCv10adminaez.shp"
    project_aoi = os.path.join(workspace, "intermediate", "project_aoi_gtap_r251", f"aoi_{iso3}.gpkg")
    area_ha = os.path.join(workspace, "intermediate", "project_aoi_gtap_r251", "pyramids", "aoi_ha_per_cell_fine.tif")
    seal7_lulc = os.path.join(workspace, "intermediate", "fine_processed_inputs", "lulc", "esa", "seals7", "lulc_esa_seals7_2017.tif")

    regional_change_csv = "G:/Shared drives/NatCapTEEMs/Projects/WB_MANAGE_Project/toInvest_abs.csv"
    
    
    regional_change_df = reformat_aez(regional_change_csv)
    regional_change_df = get_SEALS_lucodes(regional_change_df, SEAL7_MANAGE_LANDUSE_DICT, SEAL7_LANDUSE_DICT)
    regional_change_df = collapse_to_country(regional_change_df)
    regional_change_df = country_input_seals_format(regional_change_df)



    regional_change_df = get_total_by_aez_lu(regional_change_df, aez_vector_file, area_ha, seal7_lulc)
    regional_change_df = input_seal_format(regional_change_df)
    regional_change_df.to_csv(os.path.join(workspace, "intermediate", "project_aoi_gtap_r251", "pyramids", f"regional_change_input_country.csv"), index=False)

    # Check if things work
    source_raster_dict = {
        "bau": os.path.join(workspace, "intermediate", "stitched_lulc_simplified_scenarios", "lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2030.tif"),
        "bau_shift": os.path.join(workspace, "intermediate", "stitched_lulc_simplified_scenarios", "lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2030.tif"),
        "baseline": seal7_lulc,
        "area_ha_raster": os.path.join(workspace, "intermediate", "project_aoi_gtap_r251", "pyramids", "aoi_ha_per_cell_fine.tif")
    }
    # Do the following once
    get_aligned_rasters(base_raster = seal7_lulc, aoi_vector=project_aoi, source_raster_dict=source_raster_dict)
    base_lulc_raster = os.path.join(workspace, "intermediate", "fine_processed_inputs", "lulc", "esa", "seals7", "lulc_esa_seals7_2017_aligned.tif")
    area_ha_raster = os.path.join(workspace, "intermediate", "project_aoi_gtap_r251", "pyramids", "aoi_ha_per_cell_fine_aligned.tif")
    base_lulc_area = get_lulc_area(base_lulc_raster, area_ha_raster)

    bau_lulc_raster = os.path.join(workspace, "intermediate", "stitched_lulc_simplified_scenarios", "lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_2030_aligned.tif")
    bau_lulc_area = get_lulc_area(bau_lulc_raster, area_ha_raster)

    bau_shift_lulc_raster = os.path.join(workspace, "intermediate", "stitched_lulc_simplified_scenarios", "lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2030_aligned.tif")
    bau_shift_lulc_area = get_lulc_area(bau_shift_lulc_raster, area_ha_raster)

    base = pd.DataFrame.from_dict(base_lulc_area, orient='index', columns=['base_ha'])
    bau = pd.DataFrame.from_dict(bau_lulc_area, orient='index', columns=['bau_ha'])
    bau_shift = pd.DataFrame.from_dict(bau_shift_lulc_area, orient='index', columns=['bau_shift_ha'])

    area_data = pd.concat([base, bau, bau_shift], axis=1)
    area_data = area_data.rename_axis('code').reset_index()
    area_data['landuse'] = area_data['code'].map(dict((v,k) for k,v in SEAL7_LANDUSE_DICT.items()))
    area_data.to_csv(os.path.join(workspace, "intermediate", "project_aoi_gtap_r251", "pyramids", "lulc_area_2030.csv"), index=False)