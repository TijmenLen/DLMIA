# DLMIA
This is the final project of group 4, for the course Deep Learning for 3d Medical Image Analysis. The group has taken on the ACDC segmentation challenge, using a 2D-UNet model.

# Jupyter notebooks
2DUnet.ipynb
This is the main script, defining and training the model using 5-fold cross-validation and ensembling the best results of each fold into one final model. It outputs the best models for each 5 folds and the ensembled model itself.

Segmentations.ipynb
Loads the ensembled model and runs inference on the test dataset. This is set up for two different test datasets, namely the original ACDC data and a secret challenge dataset, which is part of the M&M's dataset. It outputs a folder of all segmentations made by the model.

Metrics.ipynb
Adapted from the original ACDC metrics code, provides Dice scores for each class except for the background and also the volumes of each class and the error in these volumes by comparing the predicted segmentations to the ground truth segmentations. It outputs a csv file containing all metrics for each scan.

Visualisation.ipynb
Visualises the scans or the segmentations, depending on which part of the notebook is run. This allows for visual inspection of the segmentations.

# Uncertainty


# 4D analysis
For 4D analysis, the trained 2D model is used. 

LoadandPredict4D.py
Loads the 4D .nii images from the input_folder=“4D_image” . And predicts the segmentation as a 4D .nii image saved in output_folder= “4D_segmentation”. It will also provide an Excel file with the uncertainties of every timestep from every patient. 

Visualize4D.py
Visualizes a selected 4D image with segementation and saves them as a video. One video of a 3D view and one video with a sagital, axial and coronal view. 

ExtractPatientInfo.py
Extracts patient information and saves that to an excel file to be used in AnalyzeSegmentationsByGroup.py

AnalyzeSegmentationsByGroup.py
Takes as input the 4D segemenations and information excel of the patients. From this the volume of the three segmentation masks are determined (left ventricle, right ventricle and myocardium). This information is then grouped into the classification of the patients. 
The outcome is a graph of the volume changing over time. 

# 4D results
Video files of 4d results of running inference on the test set with the trained ensemble can be found using the following link: https://drive.google.com/drive/folders/1c2XZVIcurkBmshi5sCM7aBD9Vc1uS81v?usp=sharing
