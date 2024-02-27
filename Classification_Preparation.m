%% build the matrix:
matrix = zeros(35838,1); % number of lines dependant on the data set used
matrix(1:24840, end) = 1; % label for anxiety

% segment lengths dependant on the data set
segment_lengths = [3*60+17, 1*60+55, 3*60+55, 1*60+33, 5*60+14, 3*60+57, 14*60+34, 7*60+4];


% Initialize an empty cell array to store the segments
segments = cell(19, numel(segment_lengths)); 

for k = [1:11, 13:19]
    data = [RSP_cell{1,k}.skewness_phase_RSP]; % choose the feature to add to the matrix (either from ECG_cell or RSP_cell)


% Initialize a variable to keep track of the current position in the data array
    currentPosition = 1;

% Iterate through each segment length
    for i = 1:numel(segment_lengths)
    % Calculate the end position of the current segment
        endPosition = currentPosition + segment_lengths(i) - 61;
        if endPosition > 2411
            endPosition = 2411;
        end
    
    % Extract the segment from the data array
        segment = data(currentPosition:endPosition);
    
    % Store the segment in the cell array
        segments{k,i} = segment;
    
    % Update the current position for the next iteration
        currentPosition = endPosition + 61;
    end
end

% Assuming your cell array is named 'cellArray'
cellArray = segments;

newcell = cell(1,8)

index = 0
% Loop through each column in the cell array
for col = 1:8
    for row = [1:11, 13:19]
        % Extract the column data from the cell array
        columnData = [cellArray{row, col}];
        
        % Store the numeric values in the corresponding column of the matrix
        vector(index+1:index+length(columnData)) = columnData;
        index = index + length(columnData);
    end
    index = 0;
    newcell{1,col} = vector;
    clear vector
end



cellArray =newcell; % 1x8 cell array with double values

% Extract columns 1, 3, 5, 7
columnIndices1 = [1, 3, 5, 7];
vector1 = cell2mat(cellArray(columnIndices1));

% Extract columns 2, 4, 6, 8
columnIndices2 = [2, 4, 6, 8];
vector2 = cell2mat(cellArray(columnIndices2));
matrix(1:24840, end+1) = vector1;
matrix(24841:end, end) = vector2;
