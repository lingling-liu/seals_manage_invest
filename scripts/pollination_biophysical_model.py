''''
This script is part of global_invest>ecosystem_services_function.py. I deleted the carbon part to make it simple.
I also cleaned unnecessary imports.
'''

import os, sys
import hazelbean as hb
import logging
import multiprocessing
import os
import sys
import time
from osgeo import gdal
from osgeo import osr
import numpy
import pygeoprocessing
import scipy.ndimage.morphology
import taskgraph

LANDCOVER_DATA_MAP = {
    'data_suffix': 'landcover raster.tif',
}
_MULT_NODATA = -1
_MASK_NODATA = 2
GLOBIO_AG_CODES = [2, (10, 40), (230, 232)]
GLOBIO_NATURAL_CODES = [6, (50, 180)]
BMP_LULC_CODES = [300]

NODATA = -9999
N_WORKERS = max(1, multiprocessing.cpu_count())

def pollination_sufficiency(lulc_input_path, pollination_sufficiency_output_path):

    
    """
    Pollination sufficiency analysis. This is based off the IPBES-Pollination
    project so that we can run on any new LULC scenarios with ESA classification.
    Used to be called dasgupta_agriculture.py but then we did it for more than just Dasgupta
    """

    global LOGGER, WORKING_DIR, ECOSHARD_DIR, CHURN_DIR, _MULT_NODATA, _MASK_NODATA, GLOBIO_AG_CODES, GLOBIO_NATURAL_CODES, BMP_LULC_CODES, NODATA, N_WORKERS, LANDCOVER_DATA_MAP

    WORKING_DIR = os.path.split(pollination_sufficiency_output_path)[0]
    ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
    CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

    # format of the key pairs is [data suffix]: [landcover raster]
    # these must be ESA landcover map type
    LANDCOVER_DATA_MAP = {
        'data_suffix': 'landcover raster.tif',
    }

    # set a limit for the cache
    gdal.SetCacheMax(2**28)

    logging.basicConfig(
        level=logging.DEBUG,
        format=(
            '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
            ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
        stream=sys.stdout)
    LOGGER = logging.getLogger('pollination')
    logging.getLogger('taskgraph').setLevel(logging.INFO)

    _MULT_NODATA = -1
    _MASK_NODATA = 2
    # the following are the globio landcover codes. A tuple (x, y) indicates the
    # inclusive range from x to y. Pollinator habitat was defined as any natural
    # land covers, as defined (GLOBIO land-cover classes 6, secondary vegetation,
    # and  50-180, various types of primary vegetation). To test sensitivity to
    # this definition we included "semi-natural" habitats (GLOBIO land-cover
    # classes 3, 4, and 5; pasture, rangeland and forestry, respectively) in
    # addition to "natural", and repeated all analyses with semi-natural  plus
    # natural habitats, but this did not substantially alter the results  so we do
    # not include it in our final analysis or code base.

    GLOBIO_AG_CODES = [2, (10, 40), (230, 232)]
    GLOBIO_NATURAL_CODES = [3, 4, 5, (50, 180)]
    BMP_LULC_CODES = [300]

    WORKING_DIR = os.path.split(pollination_sufficiency_output_path)[0]
    ECOSHARD_DIR = os.path.join(WORKING_DIR, 'ecoshard_dir')
    CHURN_DIR = os.path.join(WORKING_DIR, 'churn')

    NODATA = -9999
    N_WORKERS = max(1, multiprocessing.cpu_count())

    landcover_path = lulc_input_path    
    

    print('WORKING_DIR', WORKING_DIR)
    print('ECOSHARD_DIR', ECOSHARD_DIR)
    print('CHURN_DIR', CHURN_DIR)

    landcover_raster_list = []
    landcover_raster_list.append(landcover_path)

    task_graph = taskgraph.TaskGraph(
        WORKING_DIR, N_WORKERS, reporting_interval=5.0)
    for dir_path in [
            ECOSHARD_DIR, CHURN_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass
    time.sleep(1.0)

    for landcover_path in landcover_raster_list:
        LOGGER.info("process landcover map: %s", landcover_path)
        calculate_for_landcover(task_graph, landcover_path, pollination_sufficiency_output_path)

    task_graph.join()
    task_graph.close()

    remove_intermediates = True
    if remove_intermediates:
        hb.remove_dirs(CHURN_DIR, safety_check='delete')
        hb.remove_dirs(ECOSHARD_DIR, safety_check='delete')
        hb.remove_path(os.path.join(os.path.split(ECOSHARD_DIR)[0], 'taskgraph_data.db'))



def calculate_for_landcover(task_graph, landcover_path, output_path):
    """Calculate values for a given landcover.
    Parameters:
        task_graph (taskgraph.TaskGraph): taskgraph object used to schedule
            work.
        landcover_path (str): path to a landcover map with globio style
            landcover codes.

    Returns:
        None.
    """
    landcover_key = os.path.splitext(os.path.basename(landcover_path))[0]
    # landcover_key = landcover_key.replace('lulc_' + p.lulc_src_label + '_' + p.lulc_simplification_label, '_')
    output_dir = os.path.join(WORKING_DIR)
    # output_dir = os.path.join(WORKING_DIR, landcover_key)
    for dir_path in [output_dir, ECOSHARD_DIR, CHURN_DIR]:
        try:
            os.makedirs(dir_path)
        except OSError:
            pass


        #NEEDED
    # The proportional area of natural within 2 km was calculated for every
    #  pixel of agricultural land (GLOBIO land-cover classes 2, 230, 231, and
    #  232) at 10 arc seconds (~300 m) resolution. This 2 km scale represents
    #  the distance most commonly found to be predictive of pollination
    #  services (Kennedy et al. 2013).
    kernel_raster_path = os.path.join(CHURN_DIR, 'radial_kernel.tif')
    kernel_task = task_graph.add_task(
        func=create_radial_convolution_mask,
        args=(0.00277778, 2000., kernel_raster_path),
        target_path_list=[kernel_raster_path],
        task_name='make convolution kernel')

    # This loop is so we don't duplicate code for each mask type with the
    # only difference being the lulc codes and prefix
    mask_task_path_map = {}
    for mask_prefix, globio_codes in [
            ('ag', GLOBIO_AG_CODES), ('hab', GLOBIO_NATURAL_CODES),
            ('bmp', BMP_LULC_CODES)]:
        mask_key = f'{landcover_key}_{mask_prefix}_mask'
        mask_target_path = os.path.join(
            CHURN_DIR, f'{mask_prefix}_mask',
            f'{mask_key}.tif')
        mask_task = task_graph.add_task(
            func=mask_raster,
            args=(landcover_path, globio_codes, mask_target_path),
            target_path_list=[mask_target_path],
            task_name=f'mask {mask_key}',)

        mask_task_path_map[mask_prefix] = (mask_task, mask_target_path)


    pollhab_2km_prop_path = os.path.join(
        CHURN_DIR, 'pollhab_2km_prop',
        f'pollhab_2km_prop_{landcover_key}.tif')
    pollhab_2km_prop_task = task_graph.add_task(
        func=pygeoprocessing.convolve_2d,
        args=[
            (mask_task_path_map['hab'][1], 1), (kernel_raster_path, 1),
            pollhab_2km_prop_path],
        kwargs={
            'working_dir': CHURN_DIR,
            'ignore_nodata_and_edges': True},
        dependent_task_list=[mask_task_path_map['hab'][0], kernel_task],
        target_path_list=[pollhab_2km_prop_path],
        task_name=(
            'calculate proportional'
            f' {os.path.basename(pollhab_2km_prop_path)}'))

    # calculate pollhab_2km_prop_on_ag_10s by multiplying pollhab_2km_prop
    # by the ag mask

    pollhab_2km_prop_on_ag_path = output_path
    pollhab_2km_prop_on_ag_task = task_graph.add_task(
        func=mult_rasters,
        args=(
            mask_task_path_map['ag'][1], pollhab_2km_prop_path,
            pollhab_2km_prop_on_ag_path),
        target_path_list=[pollhab_2km_prop_on_ag_path],
        dependent_task_list=[
            pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
        task_name=(
            f'''pollhab 2km prop on ag {
                os.path.basename(pollhab_2km_prop_on_ag_path)}'''))

    #  1.1.4.  Sufficiency threshold A threshold of 0.3 was set to
    #  evaluate whether there was sufficient pollinator habitat in the 2
    #  km around farmland to provide pollination services, based on
    #  Kremen et al.'s (2005)  estimate of the area requirements for
    #  achieving full pollination. This produced a map of wild
    #  pollination sufficiency where every agricultural pixel was
    #  designated in a binary fashion: 0 if proportional area of habitat
    #  was less than 0.3; 1 if greater than 0.3. Maps of pollination
    #  sufficiency can be found at (permanent link to output), outputs
    #  "poll_suff_..." below.

    do_threhold = False
    if do_threhold:
        threshold_val = 0.3
        pollinator_suff_hab_path = os.path.join(
            CHURN_DIR, 'poll_suff_hab_ag_coverage_rasters',
            f'poll_suff_ag_coverage_prop_10s_{landcover_key}.tif')
        poll_suff_task = task_graph.add_task(
            func=threshold_select_raster,
            args=(
                pollhab_2km_prop_path,
                mask_task_path_map['ag'][1], threshold_val,
                pollinator_suff_hab_path),
            target_path_list=[pollinator_suff_hab_path],
            dependent_task_list=[
                pollhab_2km_prop_task, mask_task_path_map['ag'][0]],
            task_name=f"""poll_suff_ag_coverage_prop {
                os.path.basename(pollinator_suff_hab_path)}""")

def create_radial_convolution_mask(
        pixel_size_degree, radius_meters, kernel_filepath):
    """Create a radial mask to sample pixels in convolution filter.
    Parameters:
        pixel_size_degree (float): size of pixel in degrees.
        radius_meters (float): desired size of radial mask in meters.
    Returns:
        A 2D numpy array that can be used in a convolution to aggregate a
        raster while accounting for partial coverage of the circle on the
        edges of the pixel.
    """
    degree_len_0 = 110574  # length at 0 degrees
    degree_len_60 = 111412  # length at 60 degrees
    pixel_size_m = pixel_size_degree * (degree_len_0 + degree_len_60) / 2.0
    pixel_radius = numpy.ceil(radius_meters / pixel_size_m)
    n_pixels = (int(pixel_radius) * 2 + 1)
    sample_pixels = 200
    mask = numpy.ones((sample_pixels * n_pixels, sample_pixels * n_pixels))
    mask[mask.shape[0]//2, mask.shape[0]//2] = 0
    distance_transform = scipy.ndimage.morphology.distance_transform_edt(mask)
    mask = None
    stratified_distance = distance_transform * pixel_size_m / sample_pixels
    distance_transform = None
    in_circle = numpy.where(stratified_distance <= 2000.0, 1.0, 0.0)
    stratified_distance = None
    reshaped = in_circle.reshape(
        in_circle.shape[0] // sample_pixels, sample_pixels,
        in_circle.shape[1] // sample_pixels, sample_pixels)
    kernel_array = numpy.sum(reshaped, axis=(1, 3)) / sample_pixels**2
    normalized_kernel_array = kernel_array / numpy.sum(kernel_array)
    reshaped = None

    driver = gdal.GetDriverByName('GTiff')
    kernel_raster = driver.Create(
        kernel_filepath.encode('utf-8'), n_pixels, n_pixels, 1,
        gdal.GDT_Float32, options=[
            'BIGTIFF=IF_SAFER', 'TILED=YES', 'BLOCKXSIZE=256',
            'BLOCKYSIZE=256'])

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_raster.SetGeoTransform([-180, 1, 0, 90, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    kernel_raster.SetProjection(srs.ExportToWkt())
    kernel_band = kernel_raster.GetRasterBand(1)
    kernel_band.SetNoDataValue(NODATA)
    kernel_band.WriteArray(normalized_kernel_array)

def mask_raster(base_path, codes, target_path):
    LOGGER = logging.getLogger('pollination')
    logging.getLogger('taskgraph').setLevel(logging.INFO)
    """Mask `base_path` to 1 where values are in codes. 0 otherwise.
    Parameters:
        base_path (string): path to single band integer raster.
        codes (list): list of integer or tuple integer pairs. Membership in
            `codes` or within the inclusive range of a tuple in `codes`
            is sufficient to mask the corresponding raster integer value
            in `base_path` to 1 for `target_path`.
        target_path (string): path to desired mask raster. Any corresponding
            pixels in `base_path` that also match a value or range in
            `codes` will be masked to 1 in `target_path`. All other values
            are 0.
    Returns:
        None.
    """
    code_list = numpy.array([
        item for sublist in [
            range(x[0], x[1]+1) if isinstance(x, tuple) else [x]
            for x in codes] for item in sublist])
    LOGGER.debug(f'expanded code array {code_list}')

    base_nodata = pygeoprocessing.get_raster_info(base_path)['nodata'][0]

    def mask_codes_op(base_array, codes_array):
        """Return a bool raster if value in base_array is in codes_array."""
        result = numpy.empty(base_array.shape, dtype=numpy.int8)
        result[:] = _MASK_NODATA
        valid_mask = base_array != base_nodata
        result[valid_mask] = numpy.isin(
            base_array[valid_mask], codes_array)
        return result

    pygeoprocessing.raster_calculator(
        [(base_path, 1), (code_list, 'raw')], mask_codes_op, target_path,
        gdal.GDT_Byte, 2)

def mult_rasters(raster_a_path, raster_b_path, target_path):
    """Multiply a by b and skip nodata."""
    raster_info_a = pygeoprocessing.get_raster_info(raster_a_path)
    raster_info_b = pygeoprocessing.get_raster_info(raster_b_path)

    nodata_a = raster_info_a['nodata'][0]
    nodata_b = raster_info_b['nodata'][0]

    if raster_info_a['raster_size'] != raster_info_b['raster_size']:
        aligned_raster_a_path = (
            target_path + os.path.basename(raster_a_path) + '_aligned.tif')
        aligned_raster_b_path = (
            target_path + os.path.basename(raster_b_path) + '_aligned.tif')
        pygeoprocessing.align_and_resize_raster_stack(
            [raster_a_path, raster_b_path],
            [aligned_raster_a_path, aligned_raster_b_path],
            ['near'] * 2, raster_info_a['pixel_size'], 'intersection')
        raster_a_path = aligned_raster_a_path
        raster_b_path = aligned_raster_b_path

    def _mult_raster_op(array_a, array_b, nodata_a, nodata_b, target_nodata):
        """Multiply a by b and skip nodata."""
        result = numpy.empty(array_a.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (array_a != nodata_a) & (array_b != nodata_b)
        result[valid_mask] = array_a[valid_mask] * array_b[valid_mask]
        return result

    pygeoprocessing.raster_calculator(
        [(raster_a_path, 1), (raster_b_path, 1), (nodata_a, 'raw'),
         (nodata_b, 'raw'), (_MULT_NODATA, 'raw')], _mult_raster_op,
        target_path, gdal.GDT_Float32, _MULT_NODATA)

def threshold_select_raster(
        base_raster_path, select_raster_path, threshold_val, target_path):
    """Select `select` if `base` >= `threshold_val`.
    Parameters:
        base_raster_path (string): path to single band raster that will be
            used to determine the threshold mask to select from
            `select_raster_path`.
        select_raster_path (string): path to single band raster to pass
            through to target if aligned `base` pixel is >= `threshold_val`
            0 otherwise, or nodata if base == nodata. Must be the same
            shape as `base_raster_path`.
        threshold_val (numeric): value to use as threshold cutoff
        target_path (string): path to desired output raster, raster is a
            byte type with same dimensions and projection as
            `base_raster_path`. A pixel in this raster will be `select` if
            the corresponding pixel in `base_raster_path` is >=
            `threshold_val`, 0 otherwise or nodata if `base` == nodata.
    Returns:
        None.
    """
    base_nodata = pygeoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    target_nodata = -9999.

    def threshold_select_op(
            base_array, select_array, threshold_val, base_nodata,
            target_nodata):
        result = numpy.empty(select_array.shape, dtype=numpy.float32)
        result[:] = target_nodata
        valid_mask = (base_array != base_nodata) & (
            select_array >= 0) & (select_array <= 1)
        result[valid_mask] = select_array[valid_mask] * numpy.interp(
            base_array[valid_mask], [0, threshold_val], [0.0, 1.0], 0, 1)
        return result

    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1), (select_raster_path, 1),
         (threshold_val, 'raw'), (base_nodata, 'raw'),
         (target_nodata, 'raw')], threshold_select_op,
        target_path, gdal.GDT_Float32, target_nodata)
    
