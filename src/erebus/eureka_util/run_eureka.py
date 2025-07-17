import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import os
import shutil
from pathlib import Path

def process_uncal(root_uncal_folder : str):
    '''
    Will process uncal data in the given folder up to stage 2 using Eureka! (calints)
    Will place the calints data in a sibling folder
    '''
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(file_dir)
    
    stage1 = root_uncal_folder + "/S1_eureka.ecf"
    stage2 = root_uncal_folder + "/S2_eureka.ecf"
    shutil.copyfile(file_dir + "/S1_eureka.ecf", stage1)
    shutil.copyfile(file_dir + "/S2_eureka.ecf", stage2)
    
    # Replace names
    root_folder = "./"
    input_dir = Path(root_uncal_folder).name
    output_dir = input_dir + "_Eureka"
    print("Root folder, input dir, output dir: ", root_folder, input_dir, output_dir)
    
    __replace(stage1, root_folder, input_dir, output_dir)
    __replace(stage2, root_folder, input_dir, output_dir)

    eventlabel = 'eureka'
    ecf_path = '.' + os.sep + input_dir

    s1.rampfitJWST(eventlabel, ecf_path = ecf_path)
    s2.calibrateJWST(eventlabel, ecf_path = ecf_path)

def __replace(file_path : str, root_folder : str, input_dir : str, output_dir : str):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    contents = contents.replace("{ROOT_FOLDER}", root_folder)
    contents = contents.replace("{INPUT_DIR}", input_dir)
    contents = contents.replace("{OUTPUT_DIR}", output_dir)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(contents)