import os, sys
import seals_utils
import seals_initialize_project
import hazelbean as hb
import pandas as pd
import shutil
import time
main = ''
if __name__ == '__main__':
    start_time = time.time()
    ### ------- ENVIRONMENT SETTINGS -------------------------------
    # Users should only need to edit lines in this section
    
    # Create a ProjectFlow Object to organize directories and enable parallel processing.
    p = hb.ProjectFlow()

    # Assign project-level attributes to the p object (such as in p.base_data_dir = ... below)
    # including where the project_dir and base_data are located.
    # The project_name is used to name the project directory below. If the directory exists, each task will not recreate
    # files that already exist. 
    p.user_dir = os.path.expanduser('~')
   #p.user_dir = r"G:\Shared drives\NatCapTEEMs\Files\base_data\submissions\manage_invest\test_run"
    print(p.user_dir) # This is the user directory where the project will be created. It is set to the user's home directory by default. You can change it to any directory you want to use for your project.
    p.extra_dirs = ['Files','seals', 'projects']
    #p.extra_dirs = ['seals', 'projects']
    #p.project_name = sys.argv[1]
    p.project_name = "Bangladesh"
    # copy input files to specific project folder and base_data folder
    project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    # hb.create_directories(os.path.join(project_dir, 'inputs'))
    # package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # D:/MANAGE/Package/src/seals_dev/seals/run_manage_wb.py
    # # package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("D:/MANAGE/Package/src/seals_dev/seals/run_manage_wb.py")))))
    # shutil.copy2(os.path.join(package_dir, 'projects', p.project_name, 'inputs', 'regional_changes.csv'), os.path.join(p.user_dir, 'Files', 'base_data', 'seals', 'default_inputs', 'regional_changes.csv'))
    # shutil.copy2(os.path.join(package_dir, 'projects', p.project_name, 'inputs', 'scenario_file.csv'), os.path.join(project_dir, 'inputs', 'scenario_file.csv'))  
    # # Get project CRS from parameter file
    par_table = pd.read_csv(os.path.join(project_dir, 'inputs', 'project_parameters.csv'))
    p.projected_crs = int(par_table[par_table['parameters'] == "project_crs"]["value"].values[0])        # crs for the project. It should be changed based on location of the country. It is required for InVEST models. We ar using 7755 for IND, 32736 for TZA, 32645 for BGD
    # p.projected_crs = 54030
    p.crs_string = "ESRI" if p.projected_crs == 54030 else "EPSG"

    # p.project_name = p.project_name 32645+ '_' + hb.pretty_time() # If don't you want to recreate everything each time, comment out this line.
    
    # Based on the paths above, set the project_dir. All files will be created in this directory.
    p.project_dir = os.path.join(p.user_dir, os.sep.join(p.extra_dirs), p.project_name)
    p.set_project_dir(p.project_dir) 
    
    p.run_in_parallel = 1 # Must be set before building the task tree if the task tree has parralel iterator tasks.

    # Build the task tree via a building function and assign it to p. IF YOU WANT TO LOOK AT THE MODEL LOGIC, INSPECT THIS FUNCTION
    # seals_initialize_project.build_standard_run_task_tree(p)
    seals_initialize_project.build_manage_run_task_tree(p)

    # Set the base data dir. The model will check here to see if it has everything it needs to run.
    # If anything is missing, it will download it. You can use the same base_data dir across multiple projects.
    # Additionally, if you're clever, you can move files generated in your tasks to the right base_data_dir
    # directory so that they are available for future projects and avoids redundant processing.
    # The final directory has to be named base_data to match the naming convention on the google cloud bucket.
    # p.base_data_dir = os.path.join(p.user_dir, 'Files/base_data')
    p.user_dir_teems = r"G:\Shared drives\NatCapTEEMs"
    p.base_data_dir = os.path.join(p.user_dir_teems, 'Files/base_data/submissions/manage_invest/raw')

    # ProjectFlow downloads all files automatically via the p.get_path() function. If you want it to download from a different 
    # bucket than default, provide the name and credentials here. Otherwise uses default public data 'gtap_invest_seals_2023_04_21'.
    p.data_credentials_path = None
    p.input_bucket_name = None
    
    ## Set defaults and generate the scenario_definitions.csv if it doesn't exist.
    # SEALS will run based on the scenarios defined in a scenario_definitions.csv
    # If you have not run SEALS before, SEALS will generate it in your project's input_dir.
    # A useful way to get started is to to run SEALS on the test data without modification
    # and then edit the scenario_definitions.csv to your project needs.   
    p.scenario_definitions_filename = 'scenario_file.csv' 
    p.scenario_definitions_path = os.path.join(p.input_dir, p.scenario_definitions_filename)
    seals_initialize_project.initialize_scenario_definitions(p)
        
    # SEALS is based on an extremely comprehensive region classification system defined in the following geopackage.
    global_regions_vector_ref_path = os.path.join('cartographic', 'ee', 'ee_r264_correspondence.gpkg')
    p.global_regions_vector_path = p.get_path(global_regions_vector_ref_path)

    # Set processing resolution: determines how large of a chunk should be processed at a time. 4 deg is about max for 64gb memory systems
    p.processing_resolution = 1.0 # In degrees. Must be in pyramid_compatible_resolutions

    seals_initialize_project.set_advanced_options(p)
    
    p.L = hb.get_logger('test_run_seals')
    hb.log('Created ProjectFlow object at ' + p.project_dir + '/n    from script ' + p.calling_script + '/n    with base_data set at ' + p.base_data_dir)
    
    p.execute()

    result = 'Done!'
    
    print(f"It took {time.time() - start_time} seconds to run the script for {p.aoi_label}.")


