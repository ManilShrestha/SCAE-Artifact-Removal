from datetime import datetime
import numpy as np

def log_info(log_message):
	print( datetime.now().strftime("%H:%M:%S"),":\t ", log_message , "\n")


def has_artifact(candidate_interval, artifacts):
		for artifact in artifacts:
			# Calculate the maximum start time and minimum end time between candidate_interval and artifact
			start_max = max(candidate_interval[0], artifact[0])
			end_min = min(candidate_interval[1], artifact[1])
			
			# Check for overlap
			if start_max < end_min:
				# If there is an overlap, return True
				return True
		
		# If no overlap is found with any artifact, return False
		return False

def has_overlap(candidate_interval, test_instances):
    
    test_set = np.array(test_instances)[:,0:2]
    
    for sample in test_set:
        # Calculate the maximum start time and minimum end time between candidate_interval and artifact
        start_max = max(candidate_interval[0], sample[0])
        end_min = min(candidate_interval[1], sample[1])
        
        # Check for overlap
        if start_max < end_min:
            # If there is an overlap, return True
            return True
    
    # If no overlap is found with any artifact, return False
    return False

	
