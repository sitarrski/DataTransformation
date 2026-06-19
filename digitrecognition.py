import joblib
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# Load the digits dataset
digits = datasets.load_digits()

# Print examples with their features
print("Data features:")
print(digits.data)

# Array with output (decision) values
print("\nTarget values:")
print(digits.target)

# The first example
print("\nFirst image matrix:")
print(digits.images[0])

# Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("First Digit")
plt.show()

# Initialize the SVM classifier
clf = svm.SVC(gamma=0.001, C=100)

# Train the classifier on training data (all but the last element)
clf.fit(digits.data[:-1], digits.target[:-1])

# Predict the last digit
prediction = clf.predict(digits.data[-1:])
print("\nPredicted label for the last digit:", prediction[0])

# Display the last digit
plt.figure(2, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Last Digit")
plt.show()

# The true label of the last image
print("The true label of the last image:", digits.target[-1])

# Save the trained model to a file
model_path = r'/Users/jakub/PyCharmMiscProject/clf.joblib'
joblib.dump(clf, model_path)
print(f"\nModel saved to: {model_path}")

# Load the model from the file
clf3 = joblib.load(model_path)

# Predict using the loaded model
loaded_prediction = clf3.predict(digits.data[-1:])
print("Predicted label from the loaded model:", loaded_prediction[0])
