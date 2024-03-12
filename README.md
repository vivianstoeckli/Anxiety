# Anxiety
This repository consists of a algorithm used to calculate ML-features to classify between anxiety and non-anxiety in ECG and RSP data. 
Additionally it consists of 2 functions used in said algorithm.

# Abstract:

This thesis innovatively assesses anxiety disorders using wearable devices, specifically exploring
Electrocardiography (ECG) and Respiration Signal (RSP). Novel features derived from hilbert and
wavelet transforms are introduced to enhance anxiety detection. Data preprocessing involves
signal segmentation, categorization, and the application of a sliding window technique. Heart Rate
Variability (HRV) and Respiration Rate Variability (RRV) are extracted alongside features from the
hilbert and wavelet transforms, never previously employed for anxiety classification. The dataset
is divided into training (80%) and test (20%) subsets, and feature selection is conducted through
ANOVA. Machine learning models are trained, yielding remarkable results. The ensemble model
(Subspace k-nearest neighbors (KNN)) achieves scores of 0.998 accuracy, support vector machine
at 0.996, KNN at 0.993 and the Neural Network at 0.990 using the complete feature set. Additionally,
a random forest ranking method is applied to reduce feature dimensionality and computing time,
resulting in an accuracy of 0.999 and a Matthews correlation coefficient of 0.995 with a selected set
of 8 new features extracted all from the hilbert transformed ECG or RSP signal. These outcomes
highlight the effectiveness of the introduced features from the hilbert transform in enhancing
anxiety detection, providing both accuracy and computational efficiency
<img width="512" alt="New Model_step1_2" src="https://github.com/vivianstoeckli/Anxiety/assets/117519298/889f5ece-5eb0-4b38-afdc-7ac51ee3fc2c">

# Code Description
The repository hosts the source code for the research presented in the master thesis titled "Towards instantaneous detection of anxiety with ECG and RSP data." It comprises four primary codes, 'Data Preprocessing with Feature extraction', 'Feature Ranking Method', a code to prepare the data for classification called 'Classification Preparation' and a 'Classification Analysis' in addition to various examples demonstrating their application. The 'Data Preprocessing with Feature extraction' specifically developed for the purposes of this study, offers a possibility for the user to filter the data with a suggested filter or by building an own, furthermore it segments the data into windows and extracts 58 features from the ECG or Respiratory data. 



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

  - Preprocessing_and_Feature_Extraction.m: This MATLAB script performs preprocessing and feature calculation tasks on ECG (Electrocardiogram) and RSP (Respiration) data. The script is designed to accomplish the following tasks:
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

  - Classification_Preparation.m: This file builds a matrix to start the classification process. It is optimized for using the classification learner app in MATLAB.
1. **Ground Truth**:
   - Loads ground truth vector based on the input data.
2. **Loading Features**:
   - Loads features and sorts entries by anxiety/non anxiety with the help of the segmentation data.
3. **Building matrix**:
   - Builds matrix with all features to feed into the classification learner.

  - Feature_Ranking_Method.m: This file uses a random forest algorithm to rank features based on the input matrix. The number of trees can be chosen by the user
1. **Normalization**:
   - Normalizes the data.
2. **Random Forest Model**:
   - Creates a random forest model with out-of-bag (OOB) predictor importance with number of trees as an user-input.
3. **Out-of-Bag (OOB) Error Estimation**:
   - During the training process, each tree in the Random Forest is grown using a bootstrap sample of the data, which means that some samples are left out of each bootstrap sample. These left-out samples constitute the out-of-bag (OOB) samples.
   - For each observation in the dataset, the OOB error is estimated by aggregating predictions from the trees that did not use that observation in their bootstrap sample.
   - The OOB error provides an estimate of the model's predictive performance on unseen data.
3. **Feature Ranking**:
   - The OOB predictor importance is calculated by evaluating how much the prediction error increases when the values of a particular feature are permuted (randomly shuffled) across the OOB samples while keeping other features unchanged.
   - This calculation is performed for each feature individually, and the increase in OOB error caused by permuting each feature is recorded.
   - Features that cause a larger increase in OOB error when permuted are considered more important, as they contribute more to the predictive performance of the model.

# Dataset
In the thesis we used the dataset provided by Elgendi et al. The Dataset can be downloaded here:  https://figshare.com/articles/dataset/Anxiety_Dataset_2022/19875217

# Contact
If you have any questions, please feel free to contact us though email: Vivian Stöckli (vivian.stoeckli@gmail.com) or Mohamed Elgendi (moe.elgendi@hest.ethz.ch)

# Additional Material
: File with Classification Analysis of each Participant.




