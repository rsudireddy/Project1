import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from model.LassoHomotopy import LassoHomotopyModel  
import os 

output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Plotting function for small_test.csv
def visualize_test_predict(data_matrix, target_data, fitted_model):
    """ Visualizes results for a small dataset. """
    estimated_outputs = fitted_model.estimate(data_matrix)

    # Plot Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(target_data, estimated_outputs, color='blue', label='Estimated vs True')
    plt.plot([min(target_data), max(target_data)], [min(target_data), max(target_data)], 
             color='red', linestyle='--', label='Ideal Line')
    plt.xlabel('True Values')
    plt.ylabel('Estimated Values')
    plt.title('Estimated vs True Values (small_test.csv)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "small_estimates.png"))    
    plt.close()

    # Plot Model Weights
    plt.figure(figsize=(10, 6))
    weight_count = len(fitted_model.weights)
    position_idx = 0
    bar_positions = []
    while position_idx < weight_count:
        bar_positions.append(position_idx)
        position_idx += 1
    plt.bar(bar_positions, fitted_model.weights, color='purple', width=0.4)
    plt.xlabel('Feature Positions')
    plt.ylabel('Weight Values')
    plt.title('Lasso Weights (small_test.csv)')
    plt.savefig(os.path.join(output_dir,"small_weights.png"))
    plt.close()


# Plotting function for collinear_test.csv
def visualize_lasso_on_collinear_data(feature_set, true_values, trained_model):
    """ Visualizes results for collinear data. """
    predicted_values = trained_model.estimate(feature_set)

    # Plot Predictions vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predicted_values, color='green', label='Predicted vs True')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 
             color='red', linestyle='--', label='Perfect Match')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values (collinear_test.csv)')
    plt.legend()
    plt.savefig(os.path.join(output_dir,"collinear_estimates.png"))
    plt.close()

    # Visualize Collinear Features
    plt.figure(figsize=(10, 6))
    plt.scatter(feature_set[:, 0], feature_set[:, 1], c=true_values, cmap='viridis', edgecolors='k', s=100)
    plt.xlabel('Feature One')
    plt.ylabel('Feature Two')
    plt.title('Collinear Data: Feature One vs Feature Two')
    plt.colorbar(label='Target Values')
    plt.savefig(os.path.join(output_dir,"collinear_feature_scatter.png"))
    plt.close()

    # Plot Model Weights
    plt.figure(figsize=(10, 6))
    num_weights = len(trained_model.weights)
    weight_indices = list(range(num_weights))
    plt.bar(weight_indices, trained_model.weights, color='orange')
    plt.xlabel('Feature Indices')
    plt.ylabel('Weight Magnitude')
    plt.title('Lasso Weights (collinear_test.csv)')
    plt.savefig(os.path.join(output_dir,"collinear_weights.png"))
    plt.close()


# Test for small_test.csv
def test_predict():
    """ Test Lasso Homotopy on a small dataset. """
    estimator = LassoHomotopyModel(reg_strength=0.1)
    records = []
    with open("Tests/small_test.csv", "r") as input_file:
        file_reader = csv.DictReader(input_file)
        row_counter = 0
        while row_counter < sum(1 for _ in file_reader):
            input_file.seek(0)
            next(file_reader)  # Skip header
            for entry in file_reader:
                records.append(entry)
            break
    input_array = np.array([[float(val) for key, val in item.items() if key.startswith('x')] for item in records])
    output_array = np.array([float(item['y']) for item in records])
    estimator.train(input_array, output_array)
    results = estimator.estimate(input_array)
    assert results.shape == output_array.shape
    visualize_test_predict(input_array, output_array, estimator)


# Test for collinear_test.csv
def test_lasso_on_collinear_data():
    """ Test Lasso Homotopy on collinear data. """
    predictor = LassoHomotopyModel(reg_strength=0.5)
    data_entries = []
    with open("Tests/collinear_data.csv", "r") as data_file:
        csv_parser = csv.DictReader(data_file)
        for record in csv_parser:
            data_entries.append(record)
    feature_matrix = np.array([[float(value) for key, value in entry.items() if key.startswith('X')] for entry in data_entries])
    target_vector = np.array([float(entry['target']) for entry in data_entries])
    predictor.train(feature_matrix, target_vector)
    estimates = predictor.estimate(feature_matrix)
    assert estimates.shape == target_vector.shape
    visualize_lasso_on_collinear_data(feature_matrix, target_vector, predictor)


# Test sparsity of coefficients
def test_sparsity():
    analyzer = LassoHomotopyModel(reg_strength=0.5)
    np.random.seed(42)
    num_samples, num_features = 100, 10
    random_inputs = np.random.randn(num_samples, num_features)
    base_weights = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    noisy_outputs = random_inputs @ base_weights + np.random.randn(num_samples) * 0.01
    analyzer.train(random_inputs, noisy_outputs)
    assert np.any(analyzer.weights == 0), "Lasso should produce some zero weights"
    assert not np.all(analyzer.weights == 0), "Lasso should not zero out all weights"


# Testing model coefficient shrinkage
def test_coefficient_shrinkage():
    """ Test that coefficients are shrunk compared to OLS (alpha=0). """
    np.random.seed(42)
    sample_count, variable_count = 50, 5
    input_data = np.random.randn(sample_count, variable_count)
    true_coefficients = np.array([1.5, -2.0, 0.5, 0.0, 1.0])
    response_data = input_data @ true_coefficients + np.random.randn(sample_count) * 0.01
    unregularized_model = LassoHomotopyModel(reg_strength=0.0)
    unregularized_model.train(input_data, response_data)
    regularized_model = LassoHomotopyModel(reg_strength=0.5)
    regularized_model.train(input_data, response_data)
    avg_unreg_magnitude = np.mean(np.abs(unregularized_model.weights))
    avg_reg_magnitude = np.mean(np.abs(regularized_model.weights))
    assert avg_reg_magnitude < avg_unreg_magnitude, "Higher regularization should shrink weights"


# Testing the model with high regularization effect
def test_high_alpha_zeroes_out():
    model_instance = LassoHomotopyModel(reg_strength=100.0)
    np.random.seed(42)
    row_total, col_total = 50, 5
    feature_array = np.random.randn(row_total, col_total)
    target_array = np.random.randn(row_total)
    model_instance.train(feature_array, target_array)
    assert np.all(np.abs(model_instance.weights) < 1e-6), "High regularization should force weights near zero"


# Comparin the model with scikit-learn model
def test_comparison_with_sklearn():
    np.random.seed(42)
    entry_count, feature_count = 50, 5
    raw_features = np.random.randn(entry_count, feature_count)
    known_weights = np.array([1.0, -1.5, 0.0, 2.0, 0.0])
    noisy_targets = raw_features @ known_weights + np.random.randn(entry_count) * 0.01
    feature_mean = np.mean(raw_features, axis=0)
    feature_std = np.std(raw_features, axis=0)
    target_mean = np.mean(noisy_targets)
    scaled_features = (raw_features - feature_mean) / feature_std
    scaled_targets = noisy_targets - target_mean
    custom_model = LassoHomotopyModel(reg_strength=0.1, max_steps=1000, threshold=1e-4)
    custom_model.train(scaled_features, scaled_targets)
    custom_predictions = custom_model.estimate(scaled_features) + target_mean
    sklearn_model = Lasso(alpha=0.1, max_iter=1000, tol=1e-4)
    sklearn_model.fit(scaled_features, scaled_targets)
    sklearn_predictions = sklearn_model.predict(scaled_features) + target_mean
    assert np.allclose(custom_predictions, sklearn_predictions, atol=0.1), "Predictions should match sklearn's Lasso"
    assert np.allclose(custom_model.weights, sklearn_model.coef_, atol=0.1), "Coefficients should align with sklearn"


# Testing the model with single feature case
def test_edge_case_single_feature():
    predictor_instance = LassoHomotopyModel(reg_strength=0.1)
    np.random.seed(42)
    sample_size = 50
    single_feature_input = np.random.randn(sample_size, 1)
    response_values = 2.0 * single_feature_input[:, 0] + np.random.randn(sample_size) * 0.01
    predictor_instance.train(single_feature_input, response_values)
    predicted_results = predictor_instance.estimate(single_feature_input)
    assert predicted_results.shape == response_values.shape, "Prediction shape must equal target shape"
    assert abs(predictor_instance.weights[0]) > 0, "Single feature weight should not be zero"

