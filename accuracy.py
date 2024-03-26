from sklearn.metrics import accuracy_score, precision_score
import pickle

# Load embeddings and names
data = pickle.load(open('output/embeddings.pickle', 'rb'))
embeddings = data['embeddings']
names = data['names']

# Load label encoder
le = pickle.load(open('output/le.pickle', 'rb'))

# Load recognizer
recognizer = pickle.load(open('output/recognizer.pickle', 'rb'))

# Use embeddings to get confidence scores and predictions
confidence_scores = recognizer.predict_proba(embeddings)
predictions = recognizer.predict(embeddings)

# Decode predicted labels
predicted_names = le.inverse_transform(predictions)

# Calculate accuracy and precision
accuracy = accuracy_score(names, predicted_names)
precision = precision_score(names, predicted_names, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")

# Print confidence scores along with accuracy and precision
for i, (name, confidence) in enumerate(zip(predicted_names, confidence_scores)):
    print(f"Prediction: {name}, Confidence: {max(confidence):.2%}")

# Check if accuracy and precision are greater than 80%
if accuracy > 0.8 and precision > 0.8:
    print("Accuracy and precision meet the desired threshold.")
else:
    print("Accuracy or precision is below the desired threshold. You may need to retrain the model.")

