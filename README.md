# DLMIA
This is the final project of group 4, for the course Deep Learning for 3d Medical Image Analysis. 

# Research abstract
Segmentation of the heart in cardiac cine MR is clinically used to quantify cardiac function. We propose the use of a 2D U-Net architecture. The study utilises data from the Automatic Cardiac Diagnosis Challenge (ACDC), containing labels for the left ventricle, right ventricle and the myocardium. The model contains dropout for regularisation, an Adam optimiser, and consists of an averaging ensemble of a 5-fold cross-validation. The segmentation model achieved Dice scores of 0.8873, 0.8873 and 0.7939 on the ACDC test set for the left ventricle, right ventricle and myocardium, respectively. Uncertainty mapping was implemented to assess model reliability, revealing higher uncertainty at the boundaries of the segmented regions and in slices without relevant structures. Performance showed improved segmentation accuracy in the end-diastolic phase compared to the end-systolic phase. While the model achieved strong results, particularly for the ventricular regions, further improvement could be achieved by incorporating 3D spatial context via a 2.5D or 3D network and by using additional data augmentation techniques. 

# Jupyter notebooks
2DUnet.ipynb
This is the main script, defining and training the model using 5-fold cross-validation and ensembling the best results of each fold into one final model. It outputs the best models for each 5 folds and the ensembled model itself.

Segmentations.ipynb
Loads the ensembled model and runs inference on the test dataset. This is set up for two different test datasets, namely the original ACDC data and a secret challenge dataset, which is part of the M&M's dataset. It outputs a folder of all segmentations made by the model.

Metrics.ipynb
Adapted from the original ACDC metrics code, provides Dice scores for each class except for the background and also the volumes of each class and the error in these volumes by comparing the predicted segmentations to the ground truth segmentations. It outputs a CSV file containing all metrics for each scan.

Visualisation.ipynb
Visualises the scans or the segmentations, depending on which part of the notebook is run. This allows for visual inspection of the segmentations.

# Uncertainty
To get an insight into the uncertainty of the model, an uncertainty map is made and compared to the total error of the segmentation. For the uncertainty quantification, Monte-Carlo dropout is used, with 10 dropouts per model in the ensemble. Then the standard deviation of each pixel is taken and used to plot the uncertainty. The total error per segmentation is determined by combining the error per mask, RV, LV, Myocardium and the background.

# Uncertainty results
The results of the uncertainty script can be found at: https://drive.google.com/drive/folders/1FQDyexkSG3M87z-AsO-h-nrHWxS9_JtM?usp=drive_link

# 4D analysis
For 4D analysis, the trained 2D model is used. 

LoadandPredict4D.py
Loads the 4D .nii images from the input_folder “4D_image” and predicts the segmentation as a 4D .nii image saved in the output_folder “4D_segmentation”. It will also provide an Excel file with the uncertainties of every timestep from every patient. 

Visualize4D.py
Visualises a selected 4D image with segmentation and saves it as a video. One video of a 3D view and one video with a sagittal, axial and coronal view. 

ExtractPatientInfo.py
Extracts patient information and saves that to an Excel file to be used in AnalyzeSegmentationsByGroup.py

AnalyzeSegmentationsByGroup.py
Takes as input the 4D segmentations and Excel information of the patients. From this, the volumes of the three segmentation masks are determined (left ventricle, right ventricle and myocardium). This information is then grouped into the classification of the patients. 
The outcome is a graph of the volume changing over time. 

# 4D results
Video files of 4d results of running inference on the test set with the trained ensemble can be found at: https://drive.google.com/drive/folders/1CDDUxJaIDD1Fff8dCYquEnJ1J88g6VMj?usp=drive_link
