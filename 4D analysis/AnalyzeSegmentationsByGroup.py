import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec

# Define the folder containing the segmentations and the patient info file
segmentation_folder = r"c:\Users\User\OneDrive - University of Twente\00. MSc Robotics\M2A Deep Learning for Medical Image analysis\Project\4D and 3D representation END MODEL\4D_segmentation"
patient_info_path = r"c:\Users\User\OneDrive - University of Twente\00. MSc Robotics\M2A Deep Learning for Medical Image analysis\Project\4D and 3D representation END MODEL\Info per patient\PatientInfo.xlsx"

# Load patient group information
patient_info = pd.read_excel(patient_info_path)
patient_groups = patient_info.set_index("PatientID")["Group"].to_dict()

# Initialize a dictionary to store mask volumes for each group
mask_volumes_by_group = {}

# Find the maximum number of time frames across all segmentations
max_timepoints = 0
for filename in os.listdir(segmentation_folder):
    if filename.endswith(".nii.gz"):
        segmentation_path = os.path.join(segmentation_folder, filename)
        seg_img = nib.load(segmentation_path)
        seg_data = seg_img.get_fdata()
        max_timepoints = max(max_timepoints, seg_data.shape[3])

# Loop through all segmentation files in the folder
for filename in os.listdir(segmentation_folder):
    if filename.startswith("segmentation_") and filename.endswith("_4d.nii.gz"):  # Match your file naming convention
        patient_id = filename.split("_")[1]  # Extract patient ID (e.g., "patient101")
        if patient_id not in patient_groups:
            print(f"Group information not found for {patient_id}. Skipping...")
            continue

        group = patient_groups[patient_id]
        segmentation_path = os.path.join(segmentation_folder, filename)
        print(f"Processing {filename} for group {group}...")

        # Load the segmentation file
        seg_img = nib.load(segmentation_path)
        seg_data = seg_img.get_fdata()

        # Get the voxel size from the segmentation image
        voxel_sizes = seg_img.header.get_zooms()[:3]  # Voxel dimensions in mm
        voxel_volume = np.prod(voxel_sizes)  # Volume of a single voxel in mm³
        print(f"Voxel size for {filename}: {voxel_sizes}, Voxel volume: {voxel_volume} mm³")

        # Get the number of time frames and unique mask labels
        num_timepoints = seg_data.shape[3]
        unique_labels = np.unique(seg_data)

        # Initialize a dictionary to store volumes for this patient
        patient_volumes = {label: [] for label in unique_labels if label != 0}

        # Calculate the volume of each mask over time for this patient
        for t in range(num_timepoints):
            frame = seg_data[:, :, :, t]
            for label in unique_labels:
                if label == 0:  # Skip background
                    continue
                mask_voxels = np.sum(frame == label)
                mask_volume_mm3 = mask_voxels * voxel_volume  # Convert to mm³
                mask_volume_cm3 = mask_volume_mm3 / 1000  # Convert to cm³
                patient_volumes[label].append(mask_volume_cm3)

        # Normalize the time frames to the maximum length
        for label, volumes in patient_volumes.items():
            original_time = np.linspace(0, 1, num_timepoints)
            target_time = np.linspace(0, 1, max_timepoints)
            interpolator = interp1d(original_time, volumes, kind="linear", fill_value="extrapolate")
            normalized_volumes = interpolator(target_time)
            patient_volumes[label] = normalized_volumes

        # Add this patient's volumes to the group's dictionary
        if group not in mask_volumes_by_group:
            mask_volumes_by_group[group] = {}
        for label, volumes in patient_volumes.items():
            if label not in mask_volumes_by_group[group]:
                mask_volumes_by_group[group][label] = []
            mask_volumes_by_group[group][label].append(volumes)

# Create subfigures
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 2, figure=fig)  # Define a 3x2 grid

# Plot the "NOR" group over two columns
row, col = 0, 0
for group, mask_volumes in mask_volumes_by_group.items():
    mean_volumes = {}
    std_volumes = {}
    for label, patient_volumes in mask_volumes.items():
        # Convert the list of patient volumes into a NumPy array (patients x time frames)
        patient_volumes_array = np.array(patient_volumes)
        mean_volumes[label] = np.mean(patient_volumes_array, axis=0)
        std_volumes[label] = np.std(patient_volumes_array, axis=0)

    # Define mask labels and colors
    mask_labels = {
        1: ("Right Ventricle", "red"),
        2: ("Myocardium", "green"),
        3: ("Left Ventricle", "blue")
    }

    # Create a subplot for the group
    if group == "NOR":
        ax = fig.add_subplot(gs[0, :])  # "NOR" spans two columns
    else:
        ax = fig.add_subplot(gs[row + 1, col])  # Other groups in 2x2 grid
        col += 1
        if col > 1:  # Move to the next row after two columns
            col = 0
            row += 1

    for label in mean_volumes.keys():
        mean = mean_volumes[label]
        std = std_volumes[label]
        timepoints = range(len(mean))
        mask_name, color = mask_labels.get(label, (f"Mask {int(label)}", "gray"))  # Default to "gray" for unknown masks
        ax.plot(timepoints, mean, label=mask_name, color=color)
        ax.fill_between(timepoints, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title(f"Mask Volume Over Time (Group: {group})")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Volume (cm³)")
    ax.legend()
    ax.grid()

# Adjust layout and save the figure
plt.tight_layout()
output_plot_path = os.path.join(segmentation_folder, "mask_volumes_over_time_subfigures.png")
plt.savefig(output_plot_path)
print(f"Combined plot saved to {output_plot_path}")

# Show the plot
plt.show()