import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from hazelbean import rasterize_to_match
from pygeoprocessing import zonal_statistics

SEAL7_MANAGE_LANDUSE_DICT = {
    'Shrubland': 5,
    'Pastureland': 3,
    'SavnGrassLnd': 3,
    'cropLand': 2,
    'Builtupland': 1,
    'Forest': 4,
    'unManagedForest': 4,
    'Otherland': 5}

iso3 = "TZA"
workspace = "C:/Users/salmamun/Files/seals/projects/manage"
aez_vector_file = "G:/Shared drives/NatCapTEEMs/Projects/WB_MANAGE_Project/GTAPLULCv10adminaez/GTAPLULCv10adminaez.shp"
project_aoi = os.path.join(workspace, "intermediate", "project_aoi", f"aoi_{iso3}.gpkg")
area_ha = os.path.join(workspace, "intermediate", "project_aoi", "pyramids", "aoi_ha_per_cell_fine.tif")
seal7_lulc = os.path.join(workspace, "intermediate", "fine_processed_inputs", "lulc", "esa", "seals7", "lulc_esa_seals7_2017.tif")

regional_change_csv = "G:/Shared drives/NatCapTEEMs/Projects/WB_MANAGE_Project/toInvest.csv"




def reformat_aez(regional_change_csv):
    regional_change_df = pd.read_csv(regional_change_csv)
    regional_change_df.columns = ["GTAP141", "AEZ", "landuse", "year", "change_percent"]
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

regional_change_df = reformat_aez(regional_change_csv)
regional_change_df = get_total_by_aez_lu(regional_change_df, aez_vector_file, area_ha, seal7_lulc)
regional_change_df = input_seal_format(regional_change_df)
regional_change_df.to_csv(os.path.join(workspace, "intermediate", "project_aoi", "pyramids", f"regional_change_input.csv"), index=False)