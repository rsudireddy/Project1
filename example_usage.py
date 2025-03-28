import numpy as np
from LassoHomotopy.model.LassoHomotopy import LassoHomotopyModel 


def main():
    # Example training data
    feature_matrix = np.array([[1, 2],
                               [2, 3],
                               [3, 4],
                               [4, 5]])
    target_values = np.array([2.5, 3.5, 4.5, 5.5])
    
    # Initialize and train the model
    predictor = LassoHomotopyModel(reg_strength=1.0, max_steps=1000, threshold=1e-6)
    predictor.train(feature_matrix, target_values)
    
    offset = np.mean(target_values) - np.mean(feature_matrix @ predictor.weights)

    # Generating predictions
    estimates = feature_matrix @ predictor.weights + offset  
    print("Predictions:", estimates)
    
    print("Coefficients:", predictor.weights)
    print("Intercept:", offset)


if __name__ == "__main__":
    main()