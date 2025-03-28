import numpy as np
import matplotlib.pyplot as plt

class LassoHomotopyModel:
    def __init__(self, reg_strength=1.0, max_steps=1000, threshold=1e-4):
        self.reg_strength = reg_strength 
        self.max_steps = max_steps  # Maximum iterations
        self.threshold = threshold  
        self.max_reg = None  
        self.weights = None  # Model coefficients
        self.data_mean = None  
        self.data_scale = None  
        self.target_offset = None  

#model training code
    def train(self, features, target):
        """ Train the LASSO model using a path-based approach. """
        num_rows, num_cols = features.shape
        
        self.data_mean = np.mean(features, axis=0)
        self.data_scale = np.std(features, axis=0)
        self.target_offset = np.mean(target)
        
        scaled_features = (features - self.data_mean) / self.data_scale
        adjusted_target = target - self.target_offset

        # Calculate maximum regularization if not set
        if self.max_reg is None:
            self.max_reg = np.max(np.abs(scaled_features.T @ adjusted_target)) / num_rows

        # Initialize coefficients
        coefficients = np.zeros(num_cols)
        step_count = 0
        
        # Iterate until max steps is reached
        while step_count < self.max_steps:
            old_coeffs = coefficients.copy()
            col_idx = 0
            
            # Update each coefficient
            while col_idx < num_cols:
                feature_col = scaled_features[:, col_idx]
                error_term = adjusted_target - scaled_features @ coefficients + feature_col * coefficients[col_idx]
                gradient = np.dot(feature_col.T, error_term) / num_rows
                feature_norm = np.sum(feature_col ** 2) / num_rows
                
                if gradient < -self.reg_strength:
                    coefficients[col_idx] = (gradient + self.reg_strength) / feature_norm
                elif gradient > self.reg_strength:
                    coefficients[col_idx] = (gradient - self.reg_strength) / feature_norm
                else:
                    coefficients[col_idx] = 0
                
                col_idx += 1

            # Checking for convergence
            if np.linalg.norm(coefficients - old_coeffs) < self.threshold:
                break
            
            step_count += 1

        self.weights = coefficients
        return self

    def estimate(self, input_data):
        """ Generate predictions using the trained model. """
        scaled_input = (input_data - self.data_mean) / self.data_scale
        return scaled_input @ self.weights + self.target_offset