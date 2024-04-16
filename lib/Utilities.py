from datetime import datetime
import numpy as np
import torch 

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

	
def evaluate_model(SCAEModel, CNNmodel, loader, device):
    # Ensure the model is in evaluation mode
    actual_labels = []
    predicted_labels = []
    
    CNNmodel.eval()

    # Disable gradient calculation for efficiency and to prevent changes to the model
    with torch.no_grad():
        correct = 0
        total = 0
        
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device)

            inputs_recon = SCAEModel(inputs)
            
            # Forward pass
            outputs = CNNmodel(inputs_recon)
            
            # Convert outputs probabilities to predicted class
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            actual_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
    
    return actual_labels, predicted_labels
