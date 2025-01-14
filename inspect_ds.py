from datasets import load_dataset

# Load the dataset
dataset = load_dataset('cfilt/iitb-english-hindi')

# Print the dataset structure
print(dataset)

# Print a sample from the training set
print(dataset['train'][0])