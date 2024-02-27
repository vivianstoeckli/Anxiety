# Anxiety
This repository consists of a algorithm used to calculate ML-features to classify between anxiety and non-anxiety in ECG and RSP data. 
Additionally it consists of 2 functions used in said algorithm.

# Abstract:
This study innovatively assesses anxiety disorders using wearable devices, specifically exploring Electrocardiography (ECG) and Respiration Signal (RSP). Novel features derived from Hilbert and Wavelet transforms are introduced to enhance anxiety detection. Data preprocessing involves signal segmentation, categorization, and the application of a sliding window technique. Heart Rate Variability (HRV) and Respiration Rate Variability (RRV) are extracted alongside features from the Hilbert and Wavelet transforms, never previously employed for anxiety classification. The dataset is divided into training (80\%) and test (20\%) subsets, and feature selection is conducted through ANOVA. Machine learning models are trained, yielding remarkable results. The ensemble model (Subspace k-nearest neighbors (KNN)) achieves 99.8\% accuracy, support vector machine at 99.6\%, KNN at 99.3\% and the Neural Network at 99.0\% using the complete feature set. Additionally, a random forest ranking method is applied to reduce feature dimensionality and computing time, resulting in an accuracy of 0.999 and a Matthews correlation coefficient of 0.997 with a selected set of 8 new features. These outcomes highlight the effectiveness of the introduced features from Hilbert and Wavelet transforms in enhancing anxiety detection, providing both accuracy and computational efficiency.
<img width="512" alt="New Model_step1_2" src="https://github.com/vivianstoeckli/Anxiety/assets/117519298/889f5ece-5eb0-4b38-afdc-7ac51ee3fc2c">

# Code Description
The repository hosts the source code for the research presented in the master thesis titled "Towards instantaneous detection of anxiety with ECG and RSP data." It comprises four primary codes, 'Data Preprocessing with Feature extraction', 'Feature Ranking Method', a code to prepare the data for classification called 'Classification Preparation' and a 'Classification Analysis' in addition to various examples demonstrating their application. The 'Data Preprocessing with Feature extraction' specifically developed for the purposes of this study, offers a possibility for the user to filter the data with a suggested filter or by building an own, furthermore it segments the data into windows and extracts 58 features from the ECG or Respiratory data. 

# Data Preprocessing and Feature Extraction

This MATLAB script performs preprocessing and feature calculation tasks on ECG (Electrocardiogram) and RSP (Respiration) data. The script is designed to accomplish the following tasks:

1. **Data Loading**:
   - Loads ECG and RSP data from multiple MAT files in the 'Data' folder.
   - Organizes the loaded data into a cell array for further processing.

2. **Filtering**:
   - Applies Butterworth bandpass filters to the ECG and RSP data to remove noise and isolate relevant frequency components.
   - Utilizes zero-phase filtering (using filtfilt) to preserve signal integrity.

3. **Rearranging Data**:
   - Rearranges the filtered data into separate cell arrays for ECG and RSP signals.
   - Prepares the data for sliding window analysis.

4. **Segmentation**:
   - Performs segmentation to extract windows of the provided data.

5. **Feature Calculation**:
   - Calculates various features including Heart Rate Variability-, Respiratory Rate Variability-,Frequency Domain-, Hilbert Transform- and Wavelet Transform features.
   - Stores the calculated features in separate cell arrays for each participant.

This script provides a comprehensive framework for preprocessing physiological data and extracting meaningful features for further analysis. It leverages MATLAB's signal processing capabilities to ensure accurate and reliable results.

# Setup

## Hardware Requirements
This code requires only a standard computer with enough RAM to support the in-memory operations. 

## Software Requirements

### Matlab Dependencies
We recommend using Matlab R2023a+ to run the scripts. 

**MATLAB Toolboxes**: The following MATLAB Toolboxes need to be installed:
  - Signal Processing Toolbox
  - Wavelet Toolbox (for Wavelet analysis)
  - Statistics and Machine Learning Toolbox

# Files

  - Classification_Preparation.m
  - Feature_Ranking_Method.m
  - Preprocessing_and_Feature_Extraction.m




