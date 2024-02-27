%% ECG and RSP Data Preprocessing and HRV/RRV/Hilbert/Wavelet Features Calculation
% This script performs the following tasks:

% 1. Load ECG and RSP data from multiple MAT files in the 'Data' folder.
% 2. Apply Butterworth bandpass filters to the ECG and RSP data to remove noise and isolate relevant frequency components.
% 3. Rearrange the filtered data into separate cell arrays for ECG and RSP signals.
% 4. Perform sliding window analysis and calculate HRV (Heart Rate Variability) and RRV (Respiratory Rate Variability) metrics as well as the Hilbert- and Wavelet Trnsform for each window.
% 5. Store the calculated Features in separate cell arrays for each participant.



folder_path = 'Data'; 
file_pattern = fullfile(folder_path, '*.mat'); 

mat_files = dir(file_pattern); % List all MAT files in the folder
num_files = numel(mat_files); % Number of MAT files

data = cell(num_files, 1); % Cell array to store the loaded data from each MAT file

for i = 1:num_files
    file_path = fullfile(folder_path, mat_files(i).name); % Full path to the current MAT file
    % Load the MAT file
    loaded_data = load(file_path);
    % If you want to load the entire MAT file into the cell array, use:
    data{i} = loaded_data;
end
data_cell = data;

%% filtering

% Define the filter parameters for ECG
ecg_sampling_rate = 500; 
ecg_filter_order = 2;
ecg_low_cutoff_freq = 8;
ecg_high_cutoff_freq = 20;

% Define the filter parameters for RSP
rsp_sampling_rate = 500; 
rsp_filter_order = 2;
rsp_low_cutoff_freq = 0.04;
rsp_high_cutoff_freq = 0.3;

% Compute the normalized cutoff frequencies for ECG
ecg_nyquist_freq = 0.5 * ecg_sampling_rate;
ecg_normalized_low_cutoff_freq = ecg_low_cutoff_freq / ecg_nyquist_freq;
ecg_normalized_high_cutoff_freq = ecg_high_cutoff_freq / ecg_nyquist_freq;

% Compute the normalized cutoff frequencies for RSP
rsp_nyquist_freq = 0.5 * rsp_sampling_rate;
rsp_normalized_low_cutoff_freq = rsp_low_cutoff_freq / rsp_nyquist_freq;
rsp_normalized_high_cutoff_freq = rsp_high_cutoff_freq / rsp_nyquist_freq;

% Design the Butterworth filters for ECG and RSP
[ecg_b, ecg_a] = butter(ecg_filter_order, [ecg_normalized_low_cutoff_freq, ecg_normalized_high_cutoff_freq], 'bandpass');
[rsp_b, rsp_a] = butter(rsp_filter_order, [rsp_normalized_low_cutoff_freq, rsp_normalized_high_cutoff_freq], 'bandpass');

% Apply the filters to all ECG and RSP data using filtfilt (zero-phase filtering)
filtered_data_cell = cell(size(data_cell));
for i = 1:numel(data_cell)
    data_struct = data_cell{i};
    ecg_data = data_struct.data(:, 1); % the ECG data is stored in the first column of the 'data' field
    rsp_data = data_struct.data(:, 2); % the RSP data is stored in the second column of the 'data' field

    % Apply the filters to the ECG and RSP data using filtfilt (zero-phase filtering)
    filtered_ecg = filtfilt(ecg_b, ecg_a, ecg_data);
    squared_signal = filtered_ecg .^ 2; % Square the filtered signal
    filtered_rsp = filtfilt(rsp_b, rsp_a, rsp_data);

    % Update the struct with the filtered ECG and RSP data
    data_struct.data(:, 1) = squared_signal;
    data_struct.data(:, 2) = filtered_rsp;
    data_struct.data(:, 3) = filtered_ecg;

    % Store the updated struct in the filtered data cell array
    filtered_data_cell{i} = data_struct;
end

% Save the filtered data cell array (optional)
save('filtered_data_cell.mat', 'filtered_data_cell');

%%

% Initialize the new cell array
rearranged_data_cell = cell(1, 19);

% Loop through each struct in the original cell array
for i = [1:11, 13:19]  % prt. 12 not included in analysis
    % Extract the 'data' field from the current struct
    data_in_struct = filtered_data_cell{i}.data;
    
    % Extract the first dimension of the 'data' field and store it in the new cell array
    rearranged_data_cell_ECG_squared{i} = data_in_struct(:, 3);
    rearranged_data_cell_ECG{i} = data_in_struct(:, 1);
    rearranged_data_cell_RSP{i} = data_in_struct(:, 2);
end

%%  calculation of RSP- and ECG-cell:


% Parameters
samplingFrequency = 500; % Hz
windowSize = 60; % seconds
stepSize = 1; % seconds
wavelet_name = 'db4'; % Choose a wavelet 
level = 5; % Decomposition level
% Define frequency ranges 
vlf_freq_range = [0.0033, 0.04];
ulf_freq_range = [0,0.003];
lf_freq_range = [0.04, 0.15];
hf_freq_range = [0.15, 0.4];


% Calculate window and step sizes in samples
windowSizeSamples = windowSize * samplingFrequency;
stepSizeSamples = stepSize * samplingFrequency;



% Initialize the HRV cell array
ECG_features_cell = cell(1, 19);
RSP_features_cell = cell(1, 19);

% Iterate over each field in the rearranged_data_cell
for p = 1:19
    data_in_field_RSP = rearranged_data_cell_RSP{p}; % Extract the data from the current field
    numSamples = numel(data_in_field_RSP);
    data_in_field_ECG_squared = rearranged_data_cell_ECG_squared{p}; % Extract the data from the current field
    data_in_field_ECG = rearranged_data_cell_ECG{p}; % Extract the data from the current field
    numSamples = numel(data_in_field_ECG_squared);
    % Calculate the number of windows for this field
    numWindows = floor((numSamples - windowSizeSamples) / stepSizeSamples) + 1;

    % Initialize cell arrays to store RSP metrics for this field
    fields = {'rmssdRSP', 'sdnnRSP', 'breathingRate', 'breathingRateSD', 'mean_hr', 'sdnn', 'heart_rate_sd', 'rmssd', 'peak_heights', 'vlf', 'ulf', 'lf', 'lf_values_bb', 'hf', 'hf_values_bb', 'lf_to_hf_ratio', 'lf_to_hf_ratio_values_bb', 'ln_hf_power', 'sd1', 'sd2', 'sd1_sd2_ratio', 'sd1_values_bb', 'sd2_values_bb', 'sd1_sd2_ratio_values_bb', 'pnn50', 'mean_nn', 'mean_bb', 'mad_nn', 'iqr_nn', 'median_nn', 'median_bb', 'envelope_mean', 'envelope_std', 'envelope_skewness', 'envelope_kurtosis', 'envelope_mean_RSP', 'envelope_std_RSP', 'envelope_skewness_RSP', 'envelope_kurtosis_RSP', 'mean_frequency', 'mean_frequency_RSP', 'std_frequency', 'std_frequency_RSP', 'wavelet_approx_coeff_mean', 'wavelet_approx_coeff_std', 'waveletEnergy', 'waveletEntropy','wavelet_approx_coeff_mean_RSP', 'wavelet_approx_coeff_std_RSP', 'waveletEnergy_RSP', 'waveletEntropy_RSP','mean_phase', 'std_phase','mean_phase_RSP','std_phase_RSP','skewness_phase','kurtosis_phase','skewness_phase_RSP','kurtosis_phase_RSP'};
    for i = 1:numel(fields)
        eval([fields{i} '_values = cell(1, numWindows);']);
    end


    % Slide the window over the data and calculate ECG and RSP metrics for each window
    totalIterations = numWindows;
    h = waitbar(0, 'Processing Data...');
    tic; % Start the timer

    for j = 1:numWindows
        startIndex = (j - 1) * stepSizeSamples + 1;
        endIndex = startIndex + windowSizeSamples - 1;
        windowData_RSP = data_in_field_RSP(startIndex:endIndex);
        windowData_ECG_squared = data_in_field_ECG_squared(startIndex:endIndex);
        windowData_ECG = data_in_field_ECG(startIndex:endIndex);


        % calculate RRV metrics for this window
        [r_peaks, breathingRateSD, rmssdRSP, sdnnRSP, breathingRate] = findpeaks_RespData(windowData_RSP, samplingFrequency);
        % calculate HRV metrics for this window
        [peak_heights,R_peaks, mean_heart_rate, rmssdECG, sdnnECG, heart_rate_sd] = detectRPeaksAndCalculateMetrics(windowData_ECG_squared, samplingFrequency);

        % Calculate the RR intervals from R peaks (assuming R_peaks contains the detected R peak locations)
        rr_intervals = diff(R_peaks) / samplingFrequency * 1000;  % Convert to milliseconds
        bb_intervals = diff(r_peaks) / samplingFrequency * 1000;  % Convert to milliseconds
        % Calculate the successive differences between RR intervals
        successive_diffs = abs(diff(rr_intervals));
        successive_diffs_bb = abs(diff(bb_intervals));
        rr_diff = diff(rr_intervals);
        bb_diff = diff(bb_intervals);
      
        % Store HRV metrics for this window in the respective cell arrays
        mean_hr_values{j} = mean_heart_rate;
        sdnn_values{j} = sdnnECG;
        heart_rate_sd_values{j} = heart_rate_sd;
        rmssd_values{j} = rmssdECG;
        %peak_heights_values{j,:,p} = peak_heights;
        % Store RRV metrics for this window in the respective cell arrays
        rmssdRSP_values{j} = rmssdRSP;
        sdnnRSP_values{j} = sdnnRSP;
        breathingRate_values{j} = breathingRate;
        breathingRateSD_values{j} = breathingRateSD;
        

        % Calculate SD1 (Poincaré plot standard deviation perpendicular to the line of identity)
        sd1 = sqrt(0.5 * std(rr_diff).^2);
        sd1_bb = sqrt(0.5 * std(bb_diff).^2);
        % Calculate SD2 (Poincaré plot standard deviation along the line of identity)
        sd2 = sqrt(2 * std(rr_intervals).^2 - 0.5 * std(rr_diff).^2);
        sd2_bb = sqrt(2 * std(bb_intervals).^2 - 0.5 * std(bb_diff).^2);
        % Calculate SD1/SD2
        sd1_sd2_ratio = sd1/sd2;
        sd1_sd2_ratio_bb = sd1_bb/sd2_bb;
        % Store SD1 for this window
        sd1_values{j} = sd1;
        sd1_values_bb_values{j} = sd1_bb;
        % Store SD2 for this window
        sd2_values{j} = sd2;
        sd2_values_bb_values{j} = sd2_bb;
        sd1_sd2_ratio_values{j} = sd1_sd2_ratio;
        sd1_sd2_ratio_values_bb_values{j} = sd1_sd2_ratio_bb;

     
        % Calculate the MedianNN for this window
        median_nn = median(successive_diffs);
        median_bb = median(successive_diffs_bb);
        % Store the MedianNN value for this window
        median_nn_values{j} = median_nn;
        median_bb_values{j} = median_bb;


        % Calculate pNN50 for this window
        pnn50 = sum(abs(diff(rr_intervals)) > 50) / (length(rr_intervals) - 1) * 100;  % Calculate the percentage
        % Store the pNN50 value for this window
        pnn50_values{j} = pnn50;


        % Calculate MeanNN/BB for this window
        mean_nn = mean(rr_intervals);
        mean_bb = mean(bb_intervals);
        % Store MeanNN/BB for this window
        mean_nn_values{j} = mean_nn;
        mean_bb_values{j} = mean_bb;
        
        
        % Calculate MadNN for this window
        mad_nn = median(abs(rr_intervals - median(rr_intervals)));
        % Store MadNN for this window
        mad_nn_values{j} = mad_nn;


        % Calculate IQRNN for this window
        iqr_nn = iqr(rr_intervals);
        % Store IQRNN values for this window
        iqr_nn_values{j} = iqr_nn;
        

        % calculation of the frequency features:
        vlf_power = bandpower(windowData_ECG,samplingFrequency,vlf_freq_range);
        ulf_power = bandpower(windowData_ECG,samplingFrequency,ulf_freq_range);
        lf_power = bandpower(windowData_ECG,samplingFrequency,lf_freq_range);
        lf_power_bb = bandpower(windowData_RSP,samplingFrequency,lf_freq_range);
        hf_power = bandpower(windowData_ECG,samplingFrequency,hf_freq_range);
        hf_power_bb = bandpower(windowData_RSP,samplingFrequency,hf_freq_range);
        lf_to_hf_ratio = lf_power / hf_power;
        lf_to_hf_ratio_bb = lf_power_bb / hf_power_bb;
        ln_hf_power = log(hf_power);
        % Store the frequency features for this window
        vlf_values{j} = vlf_power;
        ulf_values{j} = ulf_power;
        lf_values{j} = lf_power;
        lf_values_bb_values{j} = lf_power_bb;
        hf_values{j} = hf_power;
        hf_values_bb_values{j} = hf_power_bb;
        lf_to_hf_ratio_values{j} = lf_to_hf_ratio;
        lf_to_hf_ratio_values_bb_values{j} = lf_to_hf_ratio_bb;
        ln_hf_power_values{j} = ln_hf_power;



% Hilbert:
        % Calculate envelope, inst. phase and inst. frequency using Hilbert transform
        analytic_signal_ECG = hilbert(windowData_ECG);
        analytic_signal_RSP = hilbert(windowData_RSP);
        envelope = abs(analytic_signal_ECG);
        envelope_RSP = abs(analytic_signal_RSP);
        phase = angle(analytic_signal_ECG); % wrapped or unwrapped
        phase_RSP = angle(analytic_signal_RSP); % wrapped or unwrapped
        instantaneous_frequency = (1 / (2 * pi)) * diff(phase) * samplingFrequency;
        instantaneous_frequency_RSP = (1 / (2 * pi)) * diff(phase_RSP) * samplingFrequency;

        % calculate envelope features:
        envelope_mean = mean(envelope);
        envelope_std = std(envelope);
        envelope_skewness = skewness(envelope);
        envelope_kurtosis = kurtosis(envelope);
        envelope_mean_RSP = mean(envelope_RSP);
        envelope_std_RSP = std(envelope_RSP);
        envelope_skewness_RSP = skewness(envelope_RSP);
        envelope_kurtosis_RSP = kurtosis(envelope_RSP);
        % store envelope fetures:
        envelope_mean_values{j} = envelope_mean;
        envelope_std_values{j} = envelope_std;
        envelope_skewness_values{j} = envelope_skewness;
        envelope_kurtosis_values{j} = envelope_kurtosis;
        envelope_mean_RSP_values{j} = envelope_mean_RSP;
        envelope_std_RSP_values{j} = envelope_std_RSP;
        envelope_skewness_RSP_values{j} = envelope_skewness_RSP;
        envelope_kurtosis_RSP_values{j} = envelope_kurtosis_RSP;


        % Calculate the inst. phase features
        mean_phase = mean(phase);
        std_phase = std(phase);
        mean_phase_RSP = mean(phase_RSP);
        std_phase_RSP = std(phase_RSP);
        skewness_phase = skewness(phase);
        kurtosis_phase = kurtosis(phase);
        skewness_phase_RSP = skewness(phase_RSP);
        kurtosis_phase_RSP = kurtosis(phase_RSP);
        % Store the inst. phase features
        mean_phase_values{j} = mean_phase;
        std_phase_values{j} = std_phase;
        mean_phase_RSP_values{j} = mean_phase_RSP;
        std_phase_RSP_values{j} = std_phase_RSP;
        skewness_phase_values{j} = skewness_phase;
        kurtosis_phase_values{j} = kurtosis_phase;
        skewness_phase_RSP_values{j} = skewness_phase_RSP;
        kurtosis_phase_RSP_values{j} = kurtosis_phase_RSP;

        % Calculate instantaneous frequency features
        mean_frequency = mean(instantaneous_frequency);
        std_frequency = std(instantaneous_frequency);
        mean_frequency_RSP = mean(instantaneous_frequency_RSP);
        std_frequency_RSP = std(instantaneous_frequency_RSP);
        % Store inst. frequency features
        mean_frequency_values{j} = mean_frequency;
        mean_frequency_RSP_values{j} = mean_frequency_RSP;
        std_frequency_values{j} = std_frequency;
        std_frequency_RSP_values{j} = std_frequency_RSP;


% Wavelet:
        % Perform wavelet decomposition_ECG
        [c, l] = wavedec(windowData_ECG, level, wavelet_name);
        approx_coeff = appcoef(c, l, wavelet_name, level);
        detail_coeffs = detcoef(c, l, 1:level);

        % Perform wavelet decomposition_RSP
        [c_RSP, l_RSP] = wavedec(windowData_RSP, level, wavelet_name);
        approx_coeff_RSP = appcoef(c_RSP, l_RSP, wavelet_name, level);
        detail_coeffs_RSP = detcoef(c_RSP, l_RSP, 1:level);

        % Calculate wavelet transform features from coefficients_ECG
        wavelet_approx_coeff_mean = mean(approx_coeff);
        wavelet_approx_coeff_std = std(approx_coeff);

        % Calculate wavelet transform features from coefficients_RSP
        wavelet_approx_coeff_mean_RSP = mean(approx_coeff_RSP);
        wavelet_approx_coeff_std_RSP = std(approx_coeff_RSP);
        % Calculate Wavelet Energy/Entropy_ECG
        waveletEnergy = sum(abs(approx_coeff).^2);
        waveletEntropy = wentropy(c, 'shannon');
        % Calculate Wavelet Energy/Entropy_RSP
        waveletEnergy_RSP = sum(abs(approx_coeff_RSP).^2);
        waveletEntropy_RSP = wentropy(c_RSP, 'shannon');

        % Store wavelet transform features for this window in the
        % respective cell arrays_ECG
        wavelet_approx_coeff_mean_values{j} = wavelet_approx_coeff_mean;
        wavelet_approx_coeff_std_values{j} = wavelet_approx_coeff_std;
        waveletEnergy_values{j} = waveletEnergy;
        waveletEntropy_values{j} = waveletEntropy;

        % Store wavelet transform features for this window in the
        % respective cell arrays_RSP
        wavelet_approx_coeff_mean_RSP_values{j} = wavelet_approx_coeff_mean_RSP;
        wavelet_approx_coeff_std_RSP_values{j} = wavelet_approx_coeff_std_RSP;
        waveletEnergy_RSP_values{j} = waveletEnergy_RSP;
        waveletEntropy_RSP_values{j} = waveletEntropy_RSP;

        % Update the waitbar and display progress update
        progress = j / totalIterations;
        elapsed = toc;
        estimatedTimeRemaining = (elapsed / progress) - elapsed;
        waitbar(progress, h, sprintf('Processing Data... %.2f%% - Time Remaining: %.2f seconds', progress * 100, estimatedTimeRemaining));
    end

    % Close the waitbar for this field
    close(h);


   
    % Store ECG metrics for this field in the respective cell array
    RSP_features_cell{p} = struct(...
        'rmssdRSP', rmssdRSP_values, ...
        'sdnnRSP', sdnnRSP_values, ...
        'breathingRate', breathingRate_values, ...
        'breathingRateSD', breathingRateSD_values, ...
        'mean_bb', mean_bb_values, ...
        'median_bb', median_bb_values, ...
        'lf_bb', lf_values_bb_values, ...
        'hf_bb', hf_values_bb_values, ...
        'lf_to_hf_ratio_bb', lf_to_hf_ratio_values_bb_values, ...
        'sd1_values_bb', sd1_values_bb_values, ...
        'sd2_values_bb', sd2_values_bb_values, ...
        'sd1_ratio_sd2_bb', sd1_sd2_ratio_values_bb_values, ...
        'envelope_mean_RSP', envelope_mean_RSP_values, ...
        'envelope_std_RSP', envelope_std_RSP_values, ...
        'envelope_skewness_RSP', envelope_skewness_RSP_values, ...
        'envelope_kurtosis_RSP', envelope_kurtosis_RSP_values, ...
        'mean_frequency_RSP', mean_frequency_RSP_values, ...
        'std_frequency_RSP', std_frequency_RSP_values, ...
        'wavelet_approx_coeff_mean_RSP', wavelet_approx_coeff_mean_RSP_values, ...
        'wavelet_approx_coeff_std_RSP', wavelet_approx_coeff_std_RSP_values, ...hrv_cell
        'waveletEntropy_RSP',waveletEntropy_RSP_values,...
        'waveletEnergy_RSP',waveletEnergy_RSP_values,...
        'mean_phase_RSP', mean_phase_RSP_values, ...
        'std_phase_RSP',std_phase_RSP_values, ...
        'skewness_phase_RSP',skewness_phase_RSP_values, ...
        'kurtosis_phase_RSP',kurtosis_phase_RSP_values);
    


    ECG_features_cell{p} = struct(...
        'rmssd', rmssd_values, ...
        'mean_hr', mean_hr_values, ...
        'sdnn', sdnn_values, ...
        'heart_rate_sd', heart_rate_sd_values, ...
        'pnn50', pnn50_values, ...
        'mean_nn', mean_nn_values, ...
        'mad_nn', mad_nn_values, ...
        'iqr_nn', iqr_nn_values, ...
        'median_nn', median_nn_values, ...
        'vlf', vlf_values, ...
        'ulf', ulf_values, ...
        'lf', lf_values, ...
        'hf', hf_values, ...
        'envelope_mean', envelope_mean_values, ...
        'envelope_std', envelope_std_values, ...
        'envelope_skewness', envelope_skewness_values, ...
        'envelope_kurtosis', envelope_kurtosis_values, ...
        'wavelet_approx_coeff_mean', wavelet_approx_coeff_mean_values, ...
        'wavelet_approx_coeff_std', wavelet_approx_coeff_std_values, ...
        'waveletEntropy',waveletEntropy_values,...
        'waveletEnergy',waveletEnergy_values,...
        'lf_to_hf_ratio', lf_to_hf_ratio_values, ...
        'ln_hf_power', ln_hf_power_values, ...
        'sd1', sd1_values, ...
        'sd2', sd2_values, ...
        'sd1_ratio_sd2', sd1_sd2_ratio_values, ...
        'mean_frequency', mean_frequency_values,...
        'std_frequency', std_frequency_values,...
        'mean_phase', mean_phase_values, ...
        'std_phase',std_phase_values, ...
        'skewness_phase',skewness_phase_values, ...
        'kurtosis_phase',kurtosis_phase_values);


end
