% Assuming you have a 'matrix' with the class labels as the first column

% Calculate minimum and maximum values for each column
minVals = min(matrix);
maxVals = max(matrix);

% Perform min-max normalization
normalized_matrix = (matrix - minVals) ./ (maxVals - minVals);

X = normalized_matrix(:,2:end); % assuming class vector is the first column
y = normalized_matrix(:,1);


% Create a Random Forest model with OOB predictor importance
numTrees = 100;
rf_model = TreeBagger(numTrees, X, y, 'Method', 'classification', 'OOBPredictorImportance', 'on');

% Get feature importances
feature_importances = rf_model.OOBPermutedVarDeltaError;

% Create a dictionary mapping feature indices to their importance scores
feature_importance_dict = containers.Map(1:size(X,2), feature_importances);

% Rank features based on importance
[sorted_importances, sorted_indices] = sort(cell2mat(values(feature_importance_dict)), 'descend');
sorted_features = sorted_indices;

% Display or use sorted_features as needed
disp('Feature ranking based on Random Forest Importance:');
disp(sorted_features);
%
