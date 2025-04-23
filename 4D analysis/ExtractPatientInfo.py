import os
import pandas as pd

# Define the folder containing the patient folders
info_folder = r"c:\Users\User\OneDrive - University of Twente\00. MSc Robotics\M2A Deep Learning for Medical Image analysis\Project\4D and 3D representation END MODEL\Info per patient"

# Initialize a list to store patient data
patient_data = []

# Loop through all patient folders
for patient_folder in os.listdir(info_folder):
    patient_path = os.path.join(info_folder, patient_folder)
    if os.path.isdir(patient_path):  # Ensure it's a folder
        info_file = os.path.join(patient_path, "Info.cfg")
        if os.path.exists(info_file):  # Check if Info.cfg exists
            # Read the Info.cfg file
            with open(info_file, "r") as file:
                patient_info = {}
                for line in file:
                    if ":" in line:  # Parse key-value pairs
                        key, value = line.strip().split(":")
                        patient_info[key.strip()] = value.strip()
                # Add the patient ID to the data
                patient_info["PatientID"] = patient_folder
                patient_data.append(patient_info)

# Convert the list of patient data to a DataFrame
df = pd.DataFrame(patient_data)

# Define the output Excel file path
output_excel_path = os.path.join(info_folder, "PatientInfo.xlsx")

# Save the DataFrame to an Excel file
df.to_excel(output_excel_path, index=False)

print(f"Patient information has been saved to {output_excel_path}")