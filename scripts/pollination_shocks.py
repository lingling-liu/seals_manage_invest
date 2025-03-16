import os
import hazelbean as hb
import numpy as np
import pandas as pd
import logging
import pygeoprocessing
import csv

def resample_raster(src, dest, pixel_size, projection_wkt, bounding_box):
    pygeoprocessing.align_and_resize_raster_stack(
        [src], [dest], ['near'], pixel_size,
        target_projection_wkt=projection_wkt,
        bounding_box_mode=bounding_box
    )

def calculate_pollinator_adjusted_value(lulc, poll_suff, crop_value_max_lost, crop_value_baseline, year, sufficient_pollination_threshold, logger):
    logger.info(f'Calculating crop value adjusted for pollination sufficiency in {year}')
    return np.where(
        (crop_value_max_lost > 0) & (poll_suff < sufficient_pollination_threshold) & (lulc == 2),
        crop_value_baseline - crop_value_max_lost * (1 - (1 / sufficient_pollination_threshold) * poll_suff),
        np.where(
            (crop_value_max_lost > 0) & (poll_suff >= sufficient_pollination_threshold) & (lulc == 2),
            crop_value_baseline,
            -9999.
        )
    )

def calculate_crop_value_and_shock(lulc_baseline_path, lulc_scenario_path, poll_suff_baseline_path, poll_suff_scenario_path, 
                                   crop_data_dir, pollination_dependence_spreadsheet_input_path, output_dir, shock_value_path, 
                                   scenario_label, year, base_year):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()

    # Create output paths with unique filenames for each scenario
    crop_value_difference_path = os.path.join(output_dir, f'crop_value_difference_from_baseline_to_{scenario_label}_{year}.tif')
    crop_value_pollinator_adjusted_output_path = os.path.join(output_dir, f'crop_value_pollinator_adjusted_{scenario_label}_{year}.tif')

    # Threshold for sufficient pollination
    sufficient_pollination_threshold = 0.3

    # Step 1: Load crop dependence data
    df_dependence = pd.read_excel(pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')
    crop_names = list(df_dependence['Crop map file name'])[:-3]  # Removing the last 3 crops that don't have matching production data
    pollination_dependence = list(df_dependence['poll.dep'])

    # Step 2: Initialize arrays for calculations
    ref_raster = os.path.join(crop_data_dir, 'apple', 'apple_Production.tif')
    ha_shape = hb.get_shape_from_dataset_path(ref_raster)
    crop_value_baseline = np.zeros(ha_shape)
    crop_value_no_pollination = np.zeros(ha_shape)

    # Step 3: Calculate baseline crop value and no-pollination value
    for c, crop_name in enumerate(crop_names):
        logger.info(f'Calculating crop value for {crop_name} with pollination dependence {pollination_dependence[c]}')
        
        # Load crop production data
        crop_yield_path = os.path.join(crop_data_dir, f'{crop_name}', f'{crop_name}_Production.tif')
        crop_yield = hb.as_array(crop_yield_path)
        crop_yield = np.where(crop_yield > 0, crop_yield, 0.0)

        # Calculate value (only based on production)
        crop_value_baseline += crop_yield
        crop_value_no_pollination += crop_yield * (1 - float(pollination_dependence[c]))

    # Step 4: Calculate maximum loss due to pollination
    crop_value_max_lost = crop_value_baseline - crop_value_no_pollination

    crop_value_baseline_path = os.path.join(output_dir, f'crop_value_baseline_{scenario_label}_{base_year}.tif')
    crop_value_max_lost_path = os.path.join(output_dir, f'crop_value_max_lost_{scenario_label}_{year}.tif')

    # Step 5: Save the baseline crop values
    hb.save_array_as_geotiff(crop_value_baseline, crop_value_baseline_path, ref_raster, ndv=-9999, data_type=6)
    hb.save_array_as_geotiff(crop_value_max_lost, crop_value_max_lost_path, ref_raster, ndv=-9999, data_type=6)

    # Step 6: Resample Rasters
    raster_info = pygeoprocessing.get_raster_info(lulc_scenario_path)
    target_pixel_size = raster_info['pixel_size']
    target_projection_wkt = raster_info['projection_wkt']
    target_bb = raster_info['bounding_box']

    resampled_crop_value_baseline_path = os.path.join(output_dir, f'resampled_crop_value_baseline_{scenario_label}_{base_year}.tif')
    resampled_crop_value_max_lost_path = os.path.join(output_dir, f'resampled_crop_value_max_lost_{scenario_label}_{year}.tif')
    resampled_poll_suff_baseline_path = os.path.join(output_dir, f'resampled_poll_suff_{scenario_label}_{base_year}.tif')
    resampled_poll_suff_scenario_path = os.path.join(output_dir, f'resampled_poll_suff_{scenario_label}_{year}.tif')
    resampled_lulc_baseline_path = os.path.join(output_dir, f'resampled_lulc_{scenario_label}_{base_year}.tif')

    resample_raster(crop_value_baseline_path, resampled_crop_value_baseline_path, target_pixel_size, target_projection_wkt, target_bb)
    resample_raster(crop_value_max_lost_path, resampled_crop_value_max_lost_path, target_pixel_size, target_projection_wkt, target_bb)
    resample_raster(poll_suff_baseline_path, resampled_poll_suff_baseline_path, target_pixel_size, target_projection_wkt, target_bb)
    resample_raster(poll_suff_scenario_path, resampled_poll_suff_scenario_path, target_pixel_size, target_projection_wkt, target_bb)
    resample_raster(lulc_baseline_path, resampled_lulc_baseline_path, target_pixel_size, target_projection_wkt, target_bb)

    # Step 7: Calculate Crop Value Adjusted for Pollination Sufficiency in 2017 and 2022
    lulc_baseline = hb.as_array(resampled_lulc_baseline_path)
    poll_suff_baseline = hb.as_array(resampled_poll_suff_baseline_path)
    crop_value_max_lost_aoi = hb.as_array(resampled_crop_value_max_lost_path)
    crop_value_baseline_aoi = hb.as_array(resampled_crop_value_baseline_path)

    crop_value_pollinator_adjusted_baseline = calculate_pollinator_adjusted_value(
        lulc_baseline, poll_suff_baseline, crop_value_max_lost_aoi, crop_value_baseline_aoi, base_year, sufficient_pollination_threshold, logger)

    hb.save_array_as_geotiff(crop_value_pollinator_adjusted_baseline, 
                             os.path.join(output_dir, f'crop_value_pollinator_adjusted_{scenario_label}_{base_year}.tif'),
                              lulc_baseline_path, ndv=-9999, data_type=6)

    lulc_scenario = hb.as_array(lulc_scenario_path)
    poll_suff_scenario = hb.as_array(resampled_poll_suff_scenario_path)

    crop_value_pollinator_adjusted_scenario = calculate_pollinator_adjusted_value(
        lulc_scenario, poll_suff_scenario, crop_value_max_lost_aoi, crop_value_baseline_aoi, year, sufficient_pollination_threshold, logger)

    hb.save_array_as_geotiff(crop_value_pollinator_adjusted_scenario, crop_value_pollinator_adjusted_output_path, lulc_scenario_path, ndv=-9999, data_type=6)

    # Step 8: Calculate Shock Value
    logger.info('Calculating shock value')
    shock = 1 - ((np.sum(crop_value_pollinator_adjusted_scenario) - np.sum(crop_value_pollinator_adjusted_baseline)) / np.sum(crop_value_pollinator_adjusted_baseline))
    logger.info(f'Shock value for scenario {scenario_label} (baseline: {lulc_baseline_path}, scenario: {lulc_scenario_path}): {shock}')

    # Step 9: Append Shock Value to CSV File
    with open(shock_value_path, 'a', newline='') as shock_file:
        writer = csv.writer(shock_file)
        writer.writerow([scenario_label, shock, year, lulc_baseline_path, lulc_scenario_path])

    return shock



# import os
# import hazelbean as hb
# import numpy as np
# import pandas as pd
# import logging
# import pygeoprocessing

# def resample_raster(src, dest, pixel_size, projection_wkt, bounding_box):
#     pygeoprocessing.align_and_resize_raster_stack(
#         [src], [dest], ['near'], pixel_size,
#         target_projection_wkt=projection_wkt,
#         bounding_box_mode=bounding_box
#     )

# def calculate_pollinator_adjusted_value(lulc, poll_suff, crop_value_max_lost, crop_value_baseline, year, sufficient_pollination_threshold, logger):
#     logger.info(f'Calculating crop value adjusted for pollination sufficiency in {year}')
#     return np.where(
#         (crop_value_max_lost > 0) & (poll_suff < sufficient_pollination_threshold) & (lulc == 2),
#         crop_value_baseline - crop_value_max_lost * (1 - (1 / sufficient_pollination_threshold) * poll_suff),
#         np.where(
#             (crop_value_max_lost > 0) & (poll_suff >= sufficient_pollination_threshold) & (lulc == 2),
#             crop_value_baseline,
#             -9999.
#         )
#     )

# def calculate_crop_value_and_shock(lulc_2017_path, lulc_2022_path, poll_suff_2017_path, poll_suff_2022_path, crop_data_dir, pollination_dependence_spreadsheet_input_path, output_dir,shock_value_path):
#     # Setup logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
#     logger = logging.getLogger()

#     # Create output paths
#     crop_value_difference_path = os.path.join(output_dir, 'crop_value_difference_from_baseline_to_2022.tif')
#     crop_value_pollinator_adjusted_output_path = os.path.join(output_dir, 'crop_value_pollinator_adjusted_2022.tif')

#     # Threshold for sufficient pollination
#     sufficient_pollination_threshold = 0.3

#     # Step 1: Load crop dependence data
#     df_dependence = pd.read_excel(pollination_dependence_spreadsheet_input_path, sheet_name='Crop nutrient content')
#     crop_names = list(df_dependence['Crop map file name'])[:-3]  # Removing the last 3 crops that don't have matching production data
#     pollination_dependence = list(df_dependence['poll.dep'])

#     # Step 2: Initialize arrays for calculations
#     ref_raster = os.path.join(crop_data_dir, 'apple', 'apple_Production.tif')
#     ha_shape = hb.get_shape_from_dataset_path(ref_raster)
#     crop_value_baseline = np.zeros(ha_shape)
#     crop_value_no_pollination = np.zeros(ha_shape)

#     # Step 3: Calculate baseline crop value and no-pollination value
#     for c, crop_name in enumerate(crop_names):
#         logger.info(f'Calculating crop value for {crop_name} with pollination dependence {pollination_dependence[c]}')
        
#         # Load crop production data
#         crop_yield_path = os.path.join(crop_data_dir, f'{crop_name}', f'{crop_name}_Production.tif')
#         crop_yield = hb.as_array(crop_yield_path)
#         crop_yield = np.where(crop_yield > 0, crop_yield, 0.0)

#         # Calculate value (only based on production)
#         crop_value_baseline += crop_yield
#         crop_value_no_pollination += crop_yield * (1 - float(pollination_dependence[c]))

#     # Step 4: Calculate maximum loss due to pollination
#     crop_value_max_lost = crop_value_baseline - crop_value_no_pollination

#     crop_value_baseline_path = os.path.join(output_dir, 'crop_value_baseline.tif')
#     crop_value_max_lost_path = os.path.join(output_dir, 'crop_value_max_lost.tif')

#     # Step 5: Save the baseline crop values
#     hb.save_array_as_geotiff(crop_value_baseline, crop_value_baseline_path, ref_raster, ndv=-9999, data_type=6)
#     hb.save_array_as_geotiff(crop_value_max_lost, crop_value_max_lost_path, ref_raster, ndv=-9999, data_type=6)

#     # Step 6: Resample Rasters
#     raster_info = pygeoprocessing.get_raster_info(lulc_2022_path)
#     target_pixel_size = raster_info['pixel_size']
#     target_projection_wkt = raster_info['projection_wkt']
#     target_bb = raster_info['bounding_box']

#     resampled_crop_value_baseline_path = os.path.join(output_dir, 'resampled_crop_value_baseline.tif')
#     resampled_crop_value_max_lost_path = os.path.join(output_dir, 'resampled_crop_value_max_lost.tif')
#     resampled_poll_suff_2017_path = os.path.join(output_dir, 'resampled_poll_suff_2017.tif')
#     resampled_poll_suff_2022_path = os.path.join(output_dir, 'resampled_poll_suff_2022.tif')
#     resampled_lulc_2017_path = os.path.join(output_dir, 'resampled_lulc_2017.tif')

#     resample_raster(crop_value_baseline_path, resampled_crop_value_baseline_path, target_pixel_size, target_projection_wkt, target_bb)
#     resample_raster(crop_value_max_lost_path, resampled_crop_value_max_lost_path, target_pixel_size, target_projection_wkt, target_bb)
#     resample_raster(poll_suff_2017_path, resampled_poll_suff_2017_path, target_pixel_size, target_projection_wkt, target_bb)
#     resample_raster(poll_suff_2022_path, resampled_poll_suff_2022_path, target_pixel_size, target_projection_wkt, target_bb)
#     resample_raster(lulc_2017_path, resampled_lulc_2017_path, target_pixel_size, target_projection_wkt, target_bb)

#     # Step 7: Calculate Crop Value Adjusted for Pollination Sufficiency in 2017 and 2022
#     lulc_2017 = hb.as_array(resampled_lulc_2017_path)
#     poll_suff_2017 = hb.as_array(resampled_poll_suff_2017_path)
#     crop_value_max_lost_aoi = hb.as_array(resampled_crop_value_max_lost_path)
#     crop_value_baseline_aoi = hb.as_array(resampled_crop_value_baseline_path)

#     crop_value_pollinator_adjusted_2017 = calculate_pollinator_adjusted_value(
#         lulc_2017, poll_suff_2017, crop_value_max_lost_aoi, crop_value_baseline_aoi, 2017, sufficient_pollination_threshold, logger)

#     hb.save_array_as_geotiff(crop_value_pollinator_adjusted_2017, os.path.join(output_dir, 'crop_value_pollinator_adjusted_2017.tif'), lulc_2017_path, ndv=-9999, data_type=6)

#     lulc_2022 = hb.as_array(lulc_2022_path)
#     poll_suff_2022 = hb.as_array(resampled_poll_suff_2022_path)

#     crop_value_pollinator_adjusted_2022 = calculate_pollinator_adjusted_value(
#         lulc_2022, poll_suff_2022, crop_value_max_lost_aoi, crop_value_baseline_aoi, 2022, sufficient_pollination_threshold, logger)

#     hb.save_array_as_geotiff(crop_value_pollinator_adjusted_2022, crop_value_pollinator_adjusted_output_path, lulc_2022_path, ndv=-9999, data_type=6)

#     # Step 8: Calculate Shock Value
#     logger.info('Calculating shock value')
#     shock = 1- ((np.sum(crop_value_pollinator_adjusted_2022) - np.sum(crop_value_pollinator_adjusted_2017)) / np.sum(crop_value_pollinator_adjusted_2017))
#     logger.info(f'Shock value: {shock}')

#     return shock

#     # Save shock value to a text file
#     # shock_value_path = os.path.join(output_dir, 'shock_value.txt')
#     with open(shock_value_path, 'w') as shock_file:
#         shock_file.write(f'Shock value: {shock}')

