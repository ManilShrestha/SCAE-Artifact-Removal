import sys
import os
import shutil
from lib.Utilities import *
import h5py
import pandas as pd

import scipy
from scipy.ndimage import gaussian_filter1d

from PIL import Image

from tqdm import tqdm


class PreprocessData:
    def __init__(self, source_hd5_path, out_path, annotation_file_path, annotation_metadata, signal_type="ABP") -> None:
        """This class is responsible to generate the image data as required by SCAE paper

        Args:
            source_hd5_path (String): Source hd5 file path
            out_path (String): Path where preprocessed data is stored.
            annotation_file_path (String): File path to the annotation csv file
            annotation_metadata (Dict): Dictionary with information such as modality, location and scale
            signal_type (String): ABP by default. Can be ECG too._summary_

        """

        if signal_type not in ['ABP', 'ECG']:
            raise NotImplementedError('Only ABP and ECG supported currently.')
        
        self.signal_type = signal_type
        self.annotation_file_path = annotation_file_path
        self.annotation_metadata = annotation_metadata
        self.source_hd5_path = source_hd5_path
        
        # Training folders
        self.train_artifact_folder = out_path + 'train/artifact/'
        self.train_non_artifact_folder = out_path + 'train/non-artifact/'

        # Validation folders
        self.val_artifact_folder = out_path + 'val/artifact/'
        self.val_non_artifact_folder = out_path + 'val/non-artifact/'

        # Testing folders
        self.test_artifact_folder = out_path + 'test/artifact/'
        self.test_non_artifact_folder = out_path + 'test/non-artifact/'

        # Load the annotation information in pandas dataframe
        df_annotation = pd.read_csv(self.annotation_file_path)
        df_annotation_filtered = df_annotation[(df_annotation['modality']==annotation_metadata['modality']) & (df_annotation['location']==annotation_metadata['location'])]

        self.artifacts = df_annotation_filtered[["start_time","end_time"]].to_numpy() * int(annotation_metadata['scale_wrt_hd5'])

    
    def cleanup_folders(self):
        """Cleans up the target directories for fresh load.
        """
        folders = [self.train_artifact_folder, self.train_non_artifact_folder, self.val_artifact_folder, self.val_non_artifact_folder, self.test_artifact_folder, self.test_non_artifact_folder]

        for folder in folders:
            print(f'Cleaning folder {folder}')

            if os.path.exists(folder):
                shutil.rmtree(folder)
            
            os.makedirs(folder)

    
    def get_artifacts_raw(self):
        """ Gets the artifacts in 1D signal data from the annotation file.
        """
        # Load the hdf5 data into memory
        with h5py.File(self.source_hd5_path, 'r') as file:
            dataset = file[self.annotation_metadata['data_group_name']]
            timestamp = file[self.annotation_metadata['timestamp_group_name']]

            data, timestamp = dataset[:], timestamp[:]


        artifact_raw = []
        for art in self.artifacts:
            start_idx = np.searchsorted(timestamp, art[0], side='left')
            end_idx = np.searchsorted(timestamp, art[1], side='left')
            # For 125Hz, the sample_len is 1250 for 10sec sample
            ##################################################

            interval_data = data[start_idx:end_idx]
            artifact_raw.append(interval_data)

    
        # Now filter the artifacts whose more than 50% data is less than zero
        # Calculate the percentage < 0 in the array
        artifact_raw_clean = []
        for arr in artifact_raw:
            if arr.size==0:
                continue
            percentage = (np.sum(arr < 0) / arr.size) * 100

            # Check if the percentage is greater than 50%
            if percentage >= 50:
                count_less_than_zero = np.sum(arr < 0)
                print(f"Out of {arr.shape}, {count_less_than_zero} are negative values")
            else:
                artifact_raw_clean.append(arr)

        return artifact_raw_clean
    

    def get_non_artifacts_raw(self):
        """Get raw signals with no overlap to artifacts
        """
        
        def has_artifact(candidate_interval):
            for artifact in self.artifacts:
                # Calculate the maximum start time and minimum end time between candidate_interval and artifact
                start_max = max(candidate_interval[0], artifact[0])
                end_min = min(candidate_interval[1], artifact[1])
                
                # Check for overlap
                if start_max < end_min:
                    # If there is an overlap, return True
                    return True
            
            # If no overlap is found with any artifact, return False
            return False
        
        # Load the hdf5 data into memory
        with h5py.File(self.source_hd5_path, 'r') as file:
            dataset = file[self.annotation_metadata['data_group_name']]
            timestamp = file[self.annotation_metadata['timestamp_group_name']]

            data, timestamp = dataset[:], timestamp[:]

        # Generate 5000 unique random values from 0 to 58360000 without replacement
        random_values = np.random.choice(range(len(timestamp)), 10000, replace=False)

        na_signals = []
        for r in random_values:
            start_idx_na, end_idx_na = r, r+1250
            candidate_timestamp = timestamp[start_idx_na:end_idx_na]
            
            if not has_artifact(candidate_timestamp):
                na_signals.append(data[start_idx_na:end_idx_na])
        

        return na_signals


    
    def create_images_from_signal(self, raw_signal, signal_type = 'artifact', num_images=None):
        """Creates images from the signal passed 

        Args:
            raw_signal (List of list): This is a 2D array with signals to be converted into images
            signal_type (str, optional): Primarily to identify which folder to use. 'artifact' or 'non-artifact'
            num_images (int): Number of images to create. Optional for artifact compulsory for non-artifact.
        """

        if signal_type == 'non-artifact' and num_images is None:
            raise ValueError('num_images is required for non-artifact signal types')

        count_pulses=1

        for signal in tqdm(raw_signal):
            pulses = self.get_pulses(signal, sigma=3)

            for p in pulses:
                d = self.interpolate_and_normalize(p)
                image = self.convert_1d_into_image(d)
                
                image_to_save = Image.fromarray(image.astype('uint8')*255, 'L')

                # If the signal is non-artifact, and if num of pulses exceeds the num_images, stop
                if signal_type=='non-artifact':
                    image_to_save.save(f'{self.train_non_artifact_folder}{signal_type}_{count_pulses}.jpg')
                    if count_pulses >= num_images:
                        print(f'{count_pulses+1} number of pulse images (non-artifact ridden) have been created.')		
                        return {count_pulses+1}
                else:
                    image_to_save.save(f'{self.train_artifact_folder}{signal_type}_{count_pulses}.jpg')
                
                count_pulses+=1

        print(f'{count_pulses+1} number of pulse images have been created for {signal_type}.')


    def get_pulses(self, signal, sigma=2):
        filtered_signal = gaussian_filter1d(signal, sigma=sigma)

        troughs, _ = scipy.signal.find_peaks(-filtered_signal)
        pulses = []
        for i in range(len(troughs)-1):
            s=signal[troughs[i]:troughs[i+1]]
            
            if len(s)>0:
                pulses.append(s)
        
        return pulses 
    

    def interpolate_and_normalize(self, signal):
        # Original indices
        x_original = np.arange(len(signal))

        # New indices for the desired length of 64
        x_new = np.linspace(0, len(signal) - 1, 64)

        # Perform cubic spline interpolation
        cs = scipy.interpolate.CubicSpline(x_original, signal)
        interpolated_array = cs(x_new)

        # Normalize the interpolated array to have values between 0 and 1
        normalized_array = (interpolated_array - interpolated_array.min()) / (interpolated_array.max() - interpolated_array.min())
        
        # Convert the nan elements to zero
        normalized_array[np.isnan(normalized_array)] = 0
        
        return normalized_array
    

    def convert_1d_into_image(self, signal):
        image=np.zeros((64,64))
        for x,y in enumerate(signal):
            image[x][int(y*63)]=1
        
        image = np.rot90(image, k=1)

        return image
    

    def split_train_test_val(self, ratio = (0.6,0.2,0.2)):
        """Check out the train_artifact_folder and then split them based on ratio as train/val/test.
        Train non-artifact should be same number as train artifact.
        Val non-artifact should be same number as val artifact.
        Remaining should should go as non-artifact in test.

        Args:
            ratio (tuple, optional): train/val/test. Defaults to (0.6,0.2,0.2).
        """
        from sklearn.model_selection import train_test_split
        
        # Ensure the ratio sums to 1
        if sum(ratio) != 1.0:
            raise ValueError("Ratios must sum to 1")

        train_ratio, val_ratio, test_ratio = ratio

        # Handle artifact data splitting
        artifact_files = os.listdir(self.train_artifact_folder)
        num_artifacts = len(artifact_files)
        num_val_artifacts = int(num_artifacts * val_ratio)
        num_test_artifacts = int(num_artifacts * test_ratio)

        train_artifact_files, temp_artifact_files = train_test_split(artifact_files, test_size=num_val_artifacts + num_test_artifacts)
        val_artifact_files, test_artifact_files = train_test_split(temp_artifact_files, test_size=num_test_artifacts)

        # Move artifact files
        # self._move_files(train_artifact_files, self.train_artifact_folder)
        self._move_files(val_artifact_files, self.train_artifact_folder, self.val_artifact_folder)
        self._move_files(test_artifact_files, self.train_artifact_folder, self.test_artifact_folder)

        # Handle non-artifact data splitting
        non_artifact_files = os.listdir(self.train_non_artifact_folder)
        num_train_non_artifacts = int(num_artifacts * train_ratio)
        num_val_non_artifacts = int(num_artifacts * val_ratio)


        temp_non_artifact_files, train_non_artifact_files  = train_test_split(non_artifact_files, test_size=num_train_non_artifacts)
        test_non_artifact_files, val_non_artifact_files  = train_test_split(temp_non_artifact_files, test_size=num_val_non_artifacts)

        # Move non-artifact files
        # self._move_files(train_non_artifact_files, self.train_non_artifact_folder)
        self._move_files(val_non_artifact_files, self.train_non_artifact_folder, self.val_non_artifact_folder)
        self._move_files(test_non_artifact_files, self.train_non_artifact_folder, self.test_non_artifact_folder)

        folders = [
            self.train_artifact_folder, self.train_non_artifact_folder,
            self.val_artifact_folder, self.val_non_artifact_folder,
            self.test_artifact_folder, self.test_non_artifact_folder
        ]
        for folder in folders:
            num_files = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
            print(f"{folder}: {num_files} files")



    def _move_files(self, files, source_dir, target_dir):
        """Helper function to move files to a designated directory."""
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            shutil.move(os.path.join(source_dir, file), target_dir)

