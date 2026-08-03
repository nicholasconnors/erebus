try:
    import eureka.S1_detector_processing.s1_process as s1
    import eureka.S2_calibrations.s2_calibrate as s2
    eureka_installed = True
except ImportError:
    eureka_installed = False

import os
import shutil
from pathlib import Path

CALIBRATED_SUFFIX = "_Eureka"

def process_uncal(root_uncal_folder : str) -> str:
    '''
    Will process uncal data in the given folder up to stage 2 using Eureka! (calints)
    Will place the calints data in a sibling folder which is returned as a string
    
    If the output folder already has content in it this will be skipped
    '''

    file_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Processing uncalibrated data from ", root_uncal_folder)
    
    stage1 = root_uncal_folder + "/S1_eureka.ecf"
    stage2 = root_uncal_folder + "/S2_eureka.ecf"
    shutil.copyfile(file_dir + "/S1_eureka.ecf", stage1)
    shutil.copyfile(file_dir + "/S2_eureka.ecf", stage2)
    
    # Replace names
    root_folder = str(Path(root_uncal_folder).parent)
    input_dir = Path(root_uncal_folder).name
    output_dir = input_dir + CALIBRATED_SUFFIX
    output_path = Path(root_uncal_folder).parent / (input_dir + CALIBRATED_SUFFIX)
    print("Root folder, input dir, output dir: ", root_folder, input_dir, output_dir)
    
    __replace(stage1, root_folder, input_dir, output_dir)
    __replace(stage2, root_folder, input_dir, output_dir)

    has_calints_data = os.path.exists(str(output_path)) and any(output_path.iterdir())
    
    if has_calints_data:
        print(f"Output folder exists: Likely already run! (If untrue, first clear this directory {str(output_path)}).")
        return str(output_path)
    elif not eureka_installed:
        raise("Eureka! is not installed, Erebus cannot process uncal data! Please follow the installation instructions on the Eureka! Github repo.")    

    eventlabel = 'eureka'
    ecf_path = root_uncal_folder
    
    print(f"Outputting calints data to {str(output_path)}")

    s1.rampfitJWST(eventlabel, ecf_path = ecf_path)
    s2.calibrateJWST(eventlabel, ecf_path = ecf_path)
    
    return str(output_path)

def __replace(file_path : str, root_folder : str, input_dir : str, output_dir : str):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    contents = contents.replace("{ROOT_FOLDER}", root_folder)
    contents = contents.replace("{INPUT_DIR}", input_dir)
    contents = contents.replace("{OUTPUT_DIR}", output_dir)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(contents)