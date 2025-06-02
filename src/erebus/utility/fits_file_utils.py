import numpy as np
import astropy
import os
import glob
from jwst.datamodels import dqflags
import re
from tqdm import tqdm

def get_fits_files_visits_in_folder(folder : str):
    '''
    Checks all the fits files in the folder and returns all the unique visit IDs
    '''
    all_files = np.array(glob.glob(folder + '/**/*.fits', recursive=True))
    file_names = [os.path.basename(path) for path in all_files] 
    # First 13 characters are the visit ID
    visits = np.unique([file_name[:13] for file_name in file_names])
    if len(visits) == 0:
        print(f"No visits were found in [{folder}]. Is the directory mounted?")
    return visits

def get_fits_files_in_folder(folder : str, visit_name : str, calibrated : bool):
    '''
    Gets every fits name file matching the visit_name
    Either uncal or calints
    Searches folders in the directory as well
    '''
    all_files = np.array(glob.glob(folder + '/**/*.fits', recursive=True))
    return np.array(all_files[[True if visit_name in file and ("calints" if calibrated else "uncal") 
                               in file else False for file in all_files]])

def load_all_calints_for_visit(folder : str, visit_name : str):
    '''
    Combines all pixel data into a single 3d array from all calints files in the directory
    Pixels marked DO_NOT_USE are set to -1
    '''

    print(f"Loading data from calints in {folder}")
    calints_file_names = get_fits_files_in_folder(folder, visit_name, True)
    
    combined_data = []
    combined_times = []
    
    print(f"Loading {len(calints_file_names)} segments")
    
    for file_name in tqdm(calints_file_names):
        # Segment number is in the file name
        segment_number = int(re.search("seg[0-9][0-9][0-9]", file_name).group().replace("seg", ""))
        
        file = astropy.io.fits.open(file_name)
        data = file['SCI'].data
        time = file['INT_TIMES'].data
        dq_array = file['DQ'].data
        file.close()
                
        # We take the start of the integration as our time
        int_starts = [t[1] for t in time]
        
        # Rejecting any data points that are flagged as DO NOT USE
        do_not_use = dqflags.interpret_bit_flags('DO_NOT_USE', mnemonic_map = dqflags.pixel)
        indx=np.where((dq_array & do_not_use) != 0)
        data[indx] = -1
        
        for d in data:
            combined_data.append(d)
        for t in int_starts:
            combined_times.append(t)
                
    # Sort the data by time
    print("Sorting data")
    
    combined_data = np.array(combined_data)
    combined_times = np.array(combined_times)

    sorted_indices = np.argsort(combined_times)
    sorted_t = combined_times[sorted_indices]
    sorted_data = combined_data[sorted_indices]
    
    return sorted_data, sorted_t