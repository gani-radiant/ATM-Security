import pickle

# Open the file containing pickled data in binary read mode
with open('output/embeddings.pickle', 'rb') as file:
    # Load the pickled data
    data = pickle.load(file)


# Now 'data' contains the deserialized Python object
print(data)

