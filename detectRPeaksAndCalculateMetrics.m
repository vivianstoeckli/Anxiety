function [peak_heights,R_peaks, mean_heart_rate, rmssd, sdnn, heart_rate_sd] = detectRPeaksAndCalculateMetrics(ECG_signal, sampling_rate)
    % ECG_signal: input ECG signal
    % sampling_rate: sampling rate of the ECG signal
    
    % Define the bandpass filter parameters
    lowcut = 5;     % Lower cutoff frequency in Hz
    highcut = 15;   % Higher cutoff frequency in Hz
    order = 4;      % Filter order (adjust as needed)

    % Create a bandpass filter using the butter function
    [b, a] = butter(order, [lowcut, highcut] / (sampling_rate / 2), 'bandpass');
    
    % Apply the bandpass filter to the ECG signal using filtfilt
    filtered_signal = filtfilt(b, a, ECG_signal);
    
    % Square the filtered signal element-wise
    squared_signal = filtered_signal .^ 2;
    
    % Step 2: Use findpeaks to detect QRS complex
    [~, R_peaks] = findpeaks(squared_signal, 'MinPeakHeight', 0.002 * max(squared_signal), 'MinPeakDistance', 0.6 * sampling_rate);


    for i = 1:length(R_peaks)
    % peak heights:
        peak_heights(i) = sqrt(ECG_signal(R_peaks(i),1));
    end
    
    
    % Calculate the mean heart rate
    RR_intervals = diff(R_peaks) / sampling_rate;
    mean_heart_rate = 60 / mean(RR_intervals);
    
    % Calculate the RMSSD
    successive_diff = diff(RR_intervals);
    rmssd = sqrt(mean(successive_diff .^ 2));
    
    % Calculate SDNN
    sdnn = std(RR_intervals);
    
    % Calculate the standard deviation of the heart rate
    heart_rate_sd = std(60 ./ RR_intervals);
   
end