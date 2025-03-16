import os
import pandas as pd
import geopandas as gpd
from osgeo import gdal
gdal.UseExceptions()


def get_files(tile_dir: str):
    '''
    Get all the files in the directory that end with .tif
    
    Parameters:
    tile_dir (str): The directory where the files are located
    Returns: a list of file paths
    '''
    dem_fps = []
    for files in os.listdir(tile_dir):
            if files.endswith(".tif") or files.endswith(".shp"):
                dem_fps.append(os.path.join(tile_dir, files))
    return dem_fps


def gdal_parallel_mosaic(dem_fps:list, out_fp:str, dest_crs:str="EPSG:4326", n_cores:str="ALL_CPUS"):
    '''
    Mosaic the DEM files. This method is faster as it can use mutiprocessing and directly saving after each tile is read.
    The rasterio version fails if the number of files is too large.
    Parameters:
    dem_fps (list): A list of file paths.
    out_fp (str): The output file path for the mosaic raster.
    dest_crs (str): The destination crs. Default is EPSG:4326
    n_cores (str): The number of cores to use. Default is ALL_CPUS.
    
    Returns: None. It creates a mosaic raster.'''
    gdal.Warp(destNameOrDestDS=out_fp,
              srcDSOrSrcDSTab = dem_fps,
               format="GTiff",
               resampleAlg="bilinear",
               xRes=0.002777777777777777884,
               yRes=0.002777777777777777884,
               options=f"-overwrite -multi -wm 80% -t_srs {dest_crs} -tr 0.002777777777777777884 0.002777777777777777884 -co TILED=YES -co BIGTIFF=YES -co COMPRESS=LZW -co NUM_THREADS={n_cores}")
    
def join_vector(dem_fps:list, out_fp:str):
    '''
    Join the vector files and save as a single geopackage file.
    Parameters:
    dem_fps (list): A list of vector file paths.
    out_fp (str): The output file path for the joined vector.
    Returns: None. It creates a joined vector file.
    TODO: Add a check to see if a file is empty and skip it from concatations.
    '''
    src_files_to_gpd = gpd.GeoDataFrame()
    for fp in dem_fps:
        src = gpd.read_file(fp)
        src_files_to_gpd = pd.concat([src_files_to_gpd, src], ignore_index=True)
        src_files_to_gpd.to_file(out_fp, driver="GPKG")


if __name__ == "__main__":
    dem_dir = "D:/MANAGE/CommonInputs/DEMHydroSHED3s/continetal_inputs"
    watershed_dir = "D:/MANAGE/CommonInputs/GlobalWatersheds/continetal_inputs"
    dem_output = os.path.join(os.path.dirname(dem_dir), "global_dem_10as.tif")
    watershed_output = os.path.join(os.path.dirname(watershed_dir), "watersheds_level_7.gpkg")
    dem_fps = get_files(dem_dir)
    watershed_fps = get_files(watershed_dir)
    # gdal_parallel_mosaic(dem_fps, dem_output)
    join_vector(watershed_fps, watershed_output)