from lib.PreprocessData import PreprocessData
import logging
import sys

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration variables
hdf5_file_path = '/home/ms5267@drexel.edu/moberg-precicecap/data/Patient_2021-12-21_04_16.h5'
annotation_path = '/home/ms5267@drexel.edu/moberg-precicecap/data/20240207-annotations-export-workspace=precicecap-patient=7-annotation_group=90.csv'
annotation_metadata = {
    'modality': 'ART',
    'location': 'na',
    'scale_wrt_hd5': 1e3,
    'data_group_name': 'Waveforms/ART_na',
    'timestamp_group_name': 'Waveforms/ART_na_Timestamps'
}

target_destination = '/home/ms5267@drexel.edu/moberg-precicecap/SCAE/data/abp/'

try:
    creator = PreprocessData(hdf5_file_path, target_destination, annotation_path, annotation_metadata, signal_type="ABP")
    creator.cleanup_folders()
    
    artifacts_raw = creator.get_artifacts_raw()
    non_artifacts_raw = creator.get_non_artifacts_raw()
    
    creator.create_images_from_signal(artifacts_raw)
    creator.create_images_from_signal(non_artifacts_raw, signal_type='non-artifact', num_images=100000)
    
    creator.split_train_test_val()

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    sys.exit(1)
