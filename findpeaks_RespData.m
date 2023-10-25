function [R_peaks, breathingRateSD, rmssdRSP, sdnnRSP, breathingRate] = findpeaks_RespData(RSP_signal, sampling_rate)
    % RSP_signal: input RSP signal
    % sampling_rate: sampling rate of the ECG signal
    
    % Step 1: Pre-processing
    cutoff_frequency = 0.2; % Adjust this value to set the cutoff frequency in Hz
    order = 4; % Filter order
    [b, a] = butter(order, cutoff_frequency / (sampling_rate/2), 'high');

    % Apply the filter to isolate the breathing signal
    breathing_signal = filtfilt(b, a, RSP_signal);

    % Step 2: Use findpeaks to detect QRS complex
    [~, R_peaks] = findpeaks(breathing_signal, 'MinPeakHeight', 0.002 * max(breathing_signal), 'MinPeakDistance', 0.6 * sampling_rate);
    
    % Calculate the mean heart rate
    RR_intervals = diff(R_peaks) / sampling_rate;
    breathingRate = 60 / mean(RR_intervals);
    
    % Calculate the RMSSD
    successive_diff = diff(RR_intervals);
    rmssdRSP = sqrt(mean(successive_diff .^ 2));
    
    % Calculate SDNN
    sdnnRSP = std(RR_intervals);
    
    % Calculate the standard deviation of the heart rate
    breathingRateSD = std(60 ./ RR_intervals);
   
end