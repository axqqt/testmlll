import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

# Function to load JSON data
def load_json_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

# Function to extract relevant data (Horsepower and Miles_per_Gallon)
def extract_data(data):
    return [(item['Horsepower'], item['Miles_per_Gallon']) for item in data]

# Function to preprocess the data (normalization)
def normalize_data(x_data, y_data):
    x_min, x_max = min(x_data), max(x_data)
    y_min, y_max = min(y_data), max(y_data)
    
    x_normalized = [(x - x_min) / (x_max - x_min) for x in x_data]
    y_normalized = [(y - y_min) / (y_max - y_min) for y in y_data]
    
    return x_normalized, y_normalized, (x_min, x_max), (y_min, y_max)

# Function to build and train a simple linear regression model
def build_and_train_model(x_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1], use_bias=True)
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    
    # Convert data to tensors
    x_tensor = np.array(x_train, dtype=np.float32)
    y_tensor = np.array(y_train, dtype=np.float32)

    model.fit(x_tensor, y_tensor, epochs=100, batch_size=25, verbose=0)
    return model

# Function to plot data and predictions
def plot_data(x_data, y_data, model, x_range, x_min, x_max, y_min, y_max):
    # Plot the original data points
    plt.scatter(x_data, y_data, color='blue', label='Original Data')
    
    # Generate predictions from the model
    x_line = np.linspace(x_range[0], x_range[1], 100)
    x_line_normalized = (x_line - x_min) / (x_max - x_min)  # Normalize the input for predictions
    y_line_normalized = model.predict(x_line_normalized.reshape(-1, 1))
    
    # Denormalize the predictions
    y_line_denormalized = y_line_normalized * (y_max - y_min) + y_min
    
    # Plot the regression line
    plt.plot(x_line, y_line_denormalized, color='red', label='Fitted Line')
    
    plt.xlabel('Horsepower')
    plt.ylabel('Miles per Gallon')
    plt.legend()
    plt.show()

# Main function
def main():
    # Load data from JSON file
    data = load_json_data('cardata.json')
    
    # Extract the relevant columns (Horsepower and Miles_per_Gallon)
    x_data, y_data = zip(*extract_data(data))
    
    # Normalize data
    x_normalized, y_normalized, (x_min, x_max), (y_min, y_max) = normalize_data(x_data, y_data)
    
    # Build and train the model
    model = build_and_train_model(x_normalized, y_normalized)
    
    # Plot original data and the model's prediction
    plot_data(x_data, y_data, model, (min(x_data), max(x_data)), x_min, x_max, y_min, y_max)

if __name__ == "__main__":
    main()
