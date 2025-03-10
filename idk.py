import numpy as np
from sklearn.neural_network import MLPClassifier

# Prepare the training data
X_train = np.array([
    [255/255, 255/255, 255/255],  # White
    [192/255, 192/255, 192/255],  # Light grey
    [65/255, 65/255, 65/255],     # Dark grey
    [0, 0, 0]                     # Black
])

y_train = np.array([1, 1, 0, 0])  # 1 for light, 0 for dark

# Initialize and train the neural network
clf = MLPClassifier(hidden_layer_sizes=(3,), max_iter=1000)
clf.fit(X_train, y_train)

# Test with the color Dark Blue (0, 0, 128)
test_color = np.array([[0/255, 0/255, 128/255]])

# Predict the output (light or dark)
prediction = clf.predict(test_color)

# Output the probability of "dark" and "light"
probabilities = clf.predict_proba(test_color)

# Display the result
result = f"Dark probability: {probabilities[0][0]} Light probability: {probabilities[0][1]}"
print(result)
