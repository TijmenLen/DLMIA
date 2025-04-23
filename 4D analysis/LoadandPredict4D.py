#%% Imports
import os
import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, ScaleIntensity, Resize
from scipy.ndimage import zoom
import scipy.stats
import pandas as pd
import math

#%% Define the EnsembleModel class
class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        # Collect and average the predictions (soft voting)
        outputs_list = [torch.softmax(model(x), dim=1) for model in self.models]
        avg_outputs = torch.mean(torch.stack(outputs_list), dim=0)
        return outputs_list, avg_outputs

#%% Paths
input_folder = "4D_image"  # Folder containing input 4D NIfTI files
output_folder = "4D_segmentation"  # Folder to save the output segmentations
ensemble_model_path = "ensembleHeartUNet (2).pt"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

#%% Helper Functions
def get_voxel_size(img_path):
    nimg = nib.load(img_path)
    voxel_sizes = nimg.header.get_zooms()
    return voxel_sizes

def resize_or_pad_slice(pred_np, original_shape):
    pred_shape = pred_np.shape[:2]  # Get the height and width of the prediction
    if pred_shape != original_shape[:2]:
        # Calculate scaling factors if resizing is needed
        scaling_factors = (
            original_shape[0] / pred_shape[0],  # Scale height
            original_shape[1] / pred_shape[1]   # Scale width
        )
        # Resize the slice using nearest-neighbor interpolation
        pred_np_resized = zoom(pred_np, zoom=scaling_factors, order=0)
    else:
        pred_np_resized = pred_np

    # Ensure the resized slice matches the original shape
    return pred_np_resized

#%% Load the saved ensemble model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensemble_model = torch.load(ensemble_model_path, map_location=device)
ensemble_model.to(device)
ensemble_model.eval()  # Ensure the model is in evaluation mode

#%% Define the test_transform
test_transform = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),  # Normalize intensity to [0, 1]
    Resize(spatial_size=(256, 256))  # Resize to a fixed spatial size
])

#%% Initialize a DataFrame to store uncertainties for all files
all_uncertainties = []

#%% Process all 4D NIfTI files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".nii.gz"):  # Process only NIfTI files
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, f"segmentation_{filename}")

        print(f"Processing file: {filename}")

        # Load the 4D NIfTI file
        img = nib.load(input_filepath)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        # Get the number of time points
        num_timepoints = data.shape[3]

        # Dictionary to hold 3D segmentations and uncertainties for each time step
        segmentations_3d = []
        uncertainties_3d = []

        for t in range(num_timepoints):
            print(f"Processing time step {t + 1}/{num_timepoints}...")

            # Extract the 3D volume for the current time point
            img_3d = data[:, :, :, t]

            # Process each 2D slice in the 3D volume
            slices_segmented = []
            slices_uncertainty = []
            for z in range(img_3d.shape[2]):
                # Extract the 2D slice
                img_2d = img_3d[:, :, z]

                # Add a channel dimension manually
                img_2d_with_channel = img_2d[np.newaxis, :, :]

                # Apply the test_transform to the 2D slice
                transformed_img_2d = test_transform(img_2d_with_channel)

                # Convert the MetaTensor to a NumPy array
                if isinstance(transformed_img_2d, torch.Tensor):  # Check if it's a MetaTensor or Tensor
                    transformed_img_2d = transformed_img_2d.numpy()

                # Convert to tensor and move to device
                img_2d_tensor = torch.from_numpy(transformed_img_2d).float().unsqueeze(0).to(device)

                # Run the model on the 2D slice
                with torch.no_grad():
                    outputs_list, avg_output = ensemble_model(img_2d_tensor)  # Extract avg_outputs from the tuple
                    output_np = torch.argmax(avg_output, dim=1).squeeze().cpu().numpy()  # Use avg_output

                    # Calculate pixel-wise entropy (uncertainty)
                    avg_output_np = avg_output.squeeze().cpu().numpy()  # Convert softmax probabilities to NumPy
                    entropy = scipy.stats.entropy(avg_output_np, axis=0)  # Calculate entropy along the class axis

                    # Calculate area-weighted uncertainty for each class
                    class_uncertainties = []
                    for class_id in range(avg_output_np.shape[0]):  # Iterate over each class
                        class_mask = (output_np == class_id)  # Mask for the current class
                        class_area = np.sum(class_mask)  # Number of pixels in the class
                        if class_area > 0:
                            class_entropy = np.mean(entropy[class_mask])  # Average entropy for the class
                            class_uncertainty = class_entropy * (class_area / entropy.size)  # Area-weighted uncertainty
                        else:
                            class_uncertainty = 0  # No pixels for this class
                        class_uncertainties.append(class_uncertainty)

                    # Average uncertainty across all classes
                    slice_uncertainty = np.sum(class_uncertainties)

                # Resize or pad the output to match the original slice shape
                output_resized = resize_or_pad_slice(output_np, img_2d.shape)

                # Append the segmented slice and its uncertainty
                slices_segmented.append(output_resized)
                slices_uncertainty.append(slice_uncertainty)

            # Stack the segmented slices to form a 3D volume
            segmentation_3d = np.stack(slices_segmented, axis=-1)
            segmentations_3d.append(segmentation_3d)

            # Calculate the uncertainty for the time frame
            time_frame_uncertainty = np.mean(slices_uncertainty)  # Average uncertainty across slices
            uncertainties_3d.append(time_frame_uncertainty)

        # Combine all 3D segmentations into a 4D volume
        segmentation_4d = np.stack(segmentations_3d, axis=-1)

        # Save the 4D segmentation as a NIfTI file
        segmentation_4d_nifti = nib.Nifti1Image(segmentation_4d, affine, header)
        nib.save(segmentation_4d_nifti, output_filepath)

        # Append uncertainties for this file to the global DataFrame
        for time_step, uncertainty in enumerate(uncertainties_3d, start=1):
            all_uncertainties.append({"Filename": filename, "Time Step": time_step, "Uncertainty": uncertainty})

        print(f"Saved segmentation to {output_filepath}")

#%% Save all uncertainties to a single Excel file with each file in a separate column
excel_filepath = os.path.join(output_folder, "all_uncertainties.xlsx")

# Create a dictionary where keys are filenames and values are lists of uncertainties
uncertainties_dict = {}
for entry in all_uncertainties:
    filename = entry["Filename"]
    uncertainty = entry["Uncertainty"]
    if filename not in uncertainties_dict:
        uncertainties_dict[filename] = []
    uncertainties_dict[filename].append(uncertainty)

# Find the maximum length of the uncertainty lists
max_length = max(len(uncertainties) for uncertainties in uncertainties_dict.values())

# Pad shorter lists with NaN to make all lists the same length
for filename, uncertainties in uncertainties_dict.items():
    uncertainties_dict[filename] += [math.nan] * (max_length - len(uncertainties))

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(uncertainties_dict, orient="columns")

# Save the DataFrame to an Excel file
df.to_excel(excel_filepath, index=False)

print(f"Saved all uncertainties to {excel_filepath}")
