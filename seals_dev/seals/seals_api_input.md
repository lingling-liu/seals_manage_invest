# Setup parameters

-   Description: This file defines the input to a standard SEALS run. The markdown notation here is parsed to a csv file saved at scenario_definitions.csv. This file is then used to generate the SEALS model run. If you have just installed SEALS, be sure to run the "run_test_standard.py" file to generate an example of the CSV for you to inspect, edit, or use as a template.

## aoi

-   Description: Sets the area of interest. If set as a region_label (e.g., a country-ISO3 codes), all data will be generated based that regions boundaries (as defined in the regional_boundaries_input_path). Other options include setting it to "global" or a specific shapefile, or iso3 code. Good small examples include RWA, BTN. Can be either 3-letter ISO code, keyword "global", or is a path.
-   Type: string, path
-   Default: "RWA"
-   Required: True
-   Examples: "RWA", "BDI", "global", "cartographic/ee/seals/default_inputs/rwa_bdi.shp"

## regional_boundaries_input_path

-   Description: Path to the vector file of the regions present. This should be a geopackage (gpkg) file with a set of polygons paired with a unique region_label. This label should be defined in some regional correspondence file, ideally in the ee_r264_correspondence or the eemarine_r566_correspondence in the base_data. If regional projections are used, the region_labels must correspond to those defined in regional_projections_input_path.
-   Type: path
-   Default: "cartographic/ee/ee_r264_correspondence.gpkg"
-   Required: True
-   Examples: "cartographic/ee/ee_r264_correspondence.gpkg", "cartographic/ee/eemarine_r566_correspondence.gpkg"

## calibration_parameters_source

-   Description: Path to a csv which contains all of the pretrained regressor variables. Can also be "from_calibration" indicating that this run will actually create the calibration\`\` or it can be from a tile-designated file of location-specific regressor variables.
-   Type: path
-   Default: "seals/default_inputs/default_global_coefficients.csv"
-   Required: True
-   Examples: "seals/default_inputs/default_global_coefficients.csv", "from_calibration", "tile_designated"

## base_year_lulc_path

-   Description: Path to the base year LULC data. This is the data that will be used to determine the resolution, extent, and projection of the fine resolution LULC data.
-   Type: path
-   Default: "seals/default_inputs/esa_2017.tif"
-   Required: True
-   Examples: "seals/default_inputs/esa_2017.tif", "seals/default_inputs/esa_1992.tif", "seals/default_inputs/esa_2020\_.tif"

# Projection inputs

-   Description: SEALS supports 2 types of projection inputs: regional and coarse. Regional projections, defined via a vector file (gpkg) and some distribution algorithm (e.g. proportional allocation).Coarse are coarse-resolution gridded data, such as the 30km resolution outputs of IAMs.

## regional_projections_input_path

-   Description: Path to the regional land-use change data. This should be a table (csv) file with a region_label corresponding to a polygon in regional_boundaries_input_path. In this table, changes are reported in hectarage values for each region_label for each of the changing_lulc_classes.
    -   This is not required, but if it is not provided, the coarse projections must be a raster path or raster-producing task. This is because the other coarse input option is an integer resolution for the proportional allocation of the regional projections on the coarse grid of the integer's resolution (See below).
-   Type: path
-   Default: "cartographic/ee/seals/default_inputs/rwa_bdi_regional_changes.csv"
-   Required: False
-   Examples: "cartographic/ee/seals/default_inputs/rwa_bdi_regional_changes.csv"

## coarse_projections_input_path

-   Description: Designates how to handle the coarse land-use change projections. The coarse projections are always gridded, but are much coarser than the output LULC's fine resolution. Can be a path to a coarse-gridded set of rasters for each land-use change and time, can be an integer indicating the resolution of the coarse projection (but assuming a simple smooth proportional allocation of the regional projections to the coarse projections), or can be a string indicating the name of a task in the ProjectFlow object that will generate the coarse projections.
    -   If pointing to a raster, it should be (referred) a directory of geotiffs where directories follow the scenario_structure devined in exogeonous_label, climate_label, model_label, counterfactual_label, and years. The other type is a single netcdf file that contains all of the data. If it is a netcdf, you may optionally use the time_dim_adjustment parameter to adjust the time dimension to match the desired format.
        -   If it is a path, it should ideally be a ref_path (meaning that it is included in the base_data and is relative to the base_data_dir and can be found via p.get_path())
    -   If it is an integer, it should express a pyramid-compatible integer of arcsecond resolution, specifically 10, 30, 150, 300, 900, 1800, 3600, 7200, 14400, 36000 (though anything less than 300 might be weirdly small)
    -   If is a task, it must be an algorithm that produces a directory of geotiffs, as discussed above
-   Type: path, Integer, task_name
-   Default: "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc"
-   Required: True
-   Examples: "luh2/raw_data/rcp45_ssp2/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MESSAGE-ssp245-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp19_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp119-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp26_ssp1/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-IMAGE-ssp126-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp70_ssp3/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-AIM-ssp370-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp34_ssp4/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp434-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp60_ssp4/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-GCAM-ssp460-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp34_ssp5/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp534-2-1-f_gn_2015-2100.nc", "luh2/raw_data/rcp85_ssp5/multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-MAGPIE-ssp585-2-1-f_gn_2015-2100.nc", "900", "1800", "coarse_simplified_projected_ha_difference_from_previous_year",

# Scenario structure

-   Description: Scenarios will follow the relatively strict structure defined below. Currently seals doesn't let you customize the number of different nested categories, but this may come in a future release. For now, make sure that the scenarios follow this exactly and have all parts filled (even with with a placeholder string like "all" or "no_counterfactual").

## scenario_label

-   Description: String that uniquely identifies the scenario. Will be referenced by other scenarios for comparison.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "ssp2_rcp45_luh2-message_bau",

## scenario_type

-   Description: Scenario type determines if it is historical (baseline) or future (anything else) as well as what the scenario should be compared against. I.e., Policy minus BAU.
-   Type: string
    -   One of \["historical", "base_year", "bau", "policy"\]
-   Default: None
-   Required: True
-   Examples: "bau"

## exogenous_label

-   Description: Exogenous label references some set of exogenous drivers like population, TFP growth, LUH2 pattern, SSP database etc
-   Type: string
-   Default: None
-   Required: True
-   Examples: "ssp2"

## climate_label

-   Description: One of the climate RCPs
-   Type: string
-   Default: None
-   Required: True
-   Examples: "rcp45"

## model_label

-   Description: Indicator of which model led to the coarse projection
-   Type: string
-   Default: None
-   Required: True
-   Examples: "luh2-image", "luh2-remind", "luh2-aim", "luh2-gcam", "luh2-magi", "luh2-message", "luh2-merlin", "luh2-orchidee", "luh2-plum", "luh2-sleuth", "luh2-terra", "luh2-wasp"

## counterfactual_label

-   Description: AKA policy scenario, or a label of something that you have tweaked to assess it"s efficacy
-   Type: string
-   Default: None
-   Required: True
-   Examples: "bau", "no_policy", "policy1", "policy2", "policy3"

## years

-   Description: If is not a baseline scenario, these are years into the future. Duplicate the base_year variable below if it is a base year
-   Type: space delimited list of integers
-   Default: 2030 2050
-   Required: True
-   Examples: 2030 2050, 2025 2030 2035 2040 2045 2050

## baseline_reference_label

-   Description: For calculating difference from base year, this references which baseline (must be consistent due to different models having different depictions of the base year)
-   Type: label in scenario_label
-   Default: None
-   Required: True

## base_years

-   Description: Which year in the observed data constitutes the base year. There may be multiple if, for instance, you want to use seals to downscale model outputs that update a base year to a more recent year base year which would now be based on model results but is for an existing year. Paper Idea: Do this for validation.
-   Type: integer, space delimited list of integers
-   Default: 2017
-   Required: True
-   Examples: 2017, 2000 2005 2010 2015

## key_base_year

-   Description: Even with multiple years, we will designate one as the key base year, which will be used e.g. for determining the resolution, extent, projection and other attributes of the fine resolution lulc data. It must refer to a year that has observed LULC data to load.
-   Type: integer
-   Default: 2017
-   Required: True
-   Examples: 2017

## comparison_counterfactual_labels

-   Description: If set to one of the other policies, like BAU, this will indicate which scenario to compare the performance of this scenario to. The tag "no_policy" indicates that it should not be compared to anything.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "no_policy"

# Correspondneces

-   Description: There are two different correspondences that are typically used to map inputs to that required by seals. First is an LULC correspondence that maps the input, e.g., ESA CCI 37 classes, to some simpler classification for which there exists a SEALS calibration file, e.g., seals7. The second is the coarse-gridded land-use projection that maps the input, e.g., LUH2-14, to some simpler classification for which there exists a SEALS calibration file, e.g., seals7.

## lulc_src_label

-   Description: Label of the LULC data being reclassified into the simplified form.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "esa"

## lulc_simplification_label

-   Description: Label of the new LULC simplified classification
-   Type: string
-   Default: None
-   Required: True
-   Examples: "seals7"

## lulc_correspondence_path

-   Description: Path to a csv that will map the a many-to-one reclassification of the src LULC map to a simplified version
-   Type: path
-   Default: None
-   Required: True
-   Examples: "seals/default_inputs/esa_seals7_correspondence.csv"
-   Note: Becasue this is a correspondence file, it should either have columns for at least src_id, dst_id, src_label, dst_label. Or, it should follow the multicorrespondence format, such as in ee_r264_correspondence.csv that specifies domain, attribute and size (but still follows multiple nested many-to-one relationships)

## nonchanging_class_indices

-   Description: To speed up processing, select which classes you know won"t change. For default seals7, this is the urban classes, the water classes, and the bare land class.
-   Type: list of integers
-   Default: \[0, 6, 7\]
-   Required: True
-   Examples: \[0, 6, 7\]

## coarse_src_label

-   Description: Label of the coarse LUC data that will be reclassified to the required coarse gridded projection data used as an input to the coarse gridded allocation.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "luh2-14"

## coarse_simplification_label

-   Description: Label of the simplified coarse LUC data that matches the simplified lulc classification and has a corresponding calibration file.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "seals7"

## coarse_correspondence_path

-   Description: Path to a csv that includes at least src_id, dst_id, src_label, dst_label
-   Type: path
-   Default: None
-   Required: True
-   Examples: "seals/default_inputs/luh2-14_seals7_correspondence.csv"
-   Note: Becasue this is a correspondence file, it should either have columns for at least src_id, dst_id, src_label, dst_label. Or, it should follow the multicorrespondence format, such as in ee_r264_correspondence.csv that specifies domain, attribute and size (but still follows multiple nested many-to-one relationships)

# Coarse projections preproccessing

-   Description: These are mostly deprecated, but these keywords allow for automatically extracting the correct parts of potentially malformed netcdfs.

## time_dim_adjustment

-   Description: Often NetCDF files can have the time dimension in something other than just the year. This string allows for doing operations on the time dimension to match what is desired. e.g., multiply5 add2015
-   Type: string
-   Default: None
-   Required: True
-   Examples: "add2015"

## lc_class_varname

-   Description: Because different NetCDF files have different arrangements (e.g. time is in the dimension versus LU_class is in the dimension), this option allows you to specify where in the input NC the information is. If "all_variables", assumes the LU classes will be the different variables named otherwise it can be a subset of variables, otherwise, if it is a named variable, e.g. LC_area_share then assume that the lc_class variable is
-   Type: string
-   Default: "all_variables"
-   Required: True
-   Examples: "all_variables"

## dimensions

-   Description: Lists which dimensions are stored in the netcdf in addition to lat and lon. Ideally this is just time but sometimes there are more. \# From the csv, this is a space-separated list.
-   Type: string
-   Default: None
-   Required: True
-   Examples: "time"