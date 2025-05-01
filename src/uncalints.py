import os

# TODO: In public release this should be changed somehow
os.environ['CRDS_PATH'] = '/home/nconnors/crds_cache'
os.environ['CRDS_SERVER_URL']= "https://jwst-crds.stsci.edu"

import glob
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Image2Pipeline
from eureka.S1_detector_processing.ramp_fitting import Eureka_RampFitStep

import os

class Custom1Pipeline(Detector1Pipeline):
    '''
    Internal class for running stage 1 of the JWST pipeline
    '''
    def run(self, input_file : str, output_dir : str):
        self.firstframe.skip = True
        self.lastframe.skip = True
        # This is the part that actually makes the data usable
        # Apparently this method is just an easy place to inject custom code in
        self.ramp_fit = Eureka_RampFitStep()
        
        self.save_results = True
        self.output_dir = output_dir
        # Calls "__call__"
        self(input_file)

def __run_stage_1(folder : str):
    '''
    Expects input mirimage_uncal.fits files to be in the given folder (JWST from MAST)
    Each fits file is in its own folder
    Puts calibrated results in a sibling folder called "Calibrated"
    
    Weird image breaking stuff happens in this stage
    '''

    input_folder = f"{folder}"
    output_folder = f"{folder}/../Stage1Output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in glob.glob(input_folder + "/*/*_mirimage_uncal.fits"):
        print(f"Running stage 1 on {file}")
        Custom1Pipeline().run(file, output_folder)
    
class Custom2Pipeline(Image2Pipeline):
    '''
    Internal class for running stage 2 of the JWST pipeline
    '''
    def run(self, input_file : str, output_dir : str):
        self.skip_photom = True
        self.skip_extract_1d = True
        self.skip_flat_field = True
        self.skip_msaflagopen = True
        self.skip_nsclean = True
        self.skip_imprint = True
        self.skip_bkg_subtract = True
        self.skip_master_background = True
        self.skip_wavecorr = True
        self.skip_straylight = True
        self.skip_fringe = True
        self.skip_pathloss = True
        self.skip_barshadow = True
        self.skip_wfss_contam = True
        self.skip_residual_fringe = True
        self.skip_pixel_replace = True
        self.skip_resample = True
        self.skip_cube_build = True
        self.save_results = True
        self.output_dir = output_dir
        # Calls "__call__"
        self(input_file)
        

def __run_stage_2(folder : str):
    input_folder = f"{folder}/../Stage1Output"
    output_folder = f"{folder}/../Calibrated"

    stage_2_files = glob.glob(input_folder + "/*_rateints.fits")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in stage_2_files:
        print(f"Running stage 2 on {file}")
        Custom2Pipeline().run(file, output_folder)

def process_uncalints(folder : str) -> str:
    '''
    Processes the uncalibrated data and returns the folder containing calints files
    If the data was already processes it skips to the end
    '''
    __run_stage_1(folder)
    __run_stage_2(folder)
    return folder + "/../Calibrated"