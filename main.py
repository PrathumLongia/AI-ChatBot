import os # Check if file exists    
import json     # For reading and writing JSON files
import random  # For generating random numbers
import nltk  # For natural language processing splittinfgg sentences into tokens and words
import numpy as np  # For numerical operations
import torch # For tensor operations
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms  
import torch.nn.functional as F  # For activation functions and other utilities
from torch.utils.data import Dataset, DataLoader, TensorDataset  # For creating datasets and data loaders


#nltk.download('punkt_tab')  # Download the punkt tokenizer for sentence splitting (ran only once to initlalize)
#nltk.download('wordnet')  # Download the WordNet lemmatizer (ran only once to initlalize)

class ChatbotModel(nn.Module):
    def __init__(self,input_size,output_size): # Initialize the model with input and output sizes of the neural network and the hidden layer size
        super(ChatbotModel,self).__init__() # Initialize the parent class by calling super() and passing the current class and self as arguments

        self.fc1 = nn.Linear(input_size,128) # First fully connected (Dense) layer with input size(array of numbers) and 128 hidden units(neurons) 
        self.fc2 = nn.Linear(128, 64) # Second fully connected layer with 64 hidden units
        self.fc3 = nn.Linear(64, output_size) # Third fully connected layer with output size (number of  intents in intents.json currently 5)
        self.relu = nn.ReLU() # ReLU activation function for introducing non-linearity
        self.dropout = nn.Dropout(p=0.5) # Dropout layer with a dropout probability of 0.5 to prevent overfitting
        

    def forward(self,x): # Define the forward propogation pass of the model to get to the output
        x = self.relu(self.fc1(x)) # Apply first fully connected layer to calculate bias and weights followed by ReLU activation to break linearity
        x = self.dropout(x) # Apply dropout to the output of the first layer to prevent overfitting
        x = self.relu(self.fc2(x)) # Apply Second fully connected layer to calculate bias and weights followed by ReLU activation to break linearity
        x = self.dropout(x)  # Apply dropout to the output of the second layer to prevent overfitting
        x = self.fc3(x) # Apply Third fully connected layer to calculate bias and weights to get the final output and apply softmax activation function to the logits to get probabilities for each class
        return x # Return the output of the model
    
class ChatbotAssistant:
    def __init__(self,intents_path,function_mappings = None): # Initialize the ChatBotAssistant class with the path to the intents file and function mappings which will be called when intents are recognised
        self.model = None # Initialize the model attribute to None
        self.intents_path = intents_path  # Store the path to the intents file
    
        self.documents = [] # List to store the tokenized patterns (numbers) and their corresponding intent sentences
        self.vocabulary = [] # List to store the unique words in the tokenized patterns
        self.intents = [] # List to store the unique intents
        self.intents_responses = {} # Dictionary to map intents to their corresponding responses
        self.function_mappings = function_mappings
        self.X = None
        self.Y = None



    #  A lemmatizer is a tool in natural language processing (NLP) that reduces words to their base or dictionary form, 
    #  known as the “lemma.” Unlike stemming, which just chops off prefixes or suffixes,
    #  lemmatization considers the meaning and context to return a proper root word.
    @staticmethod # Decorator to indicate that the method does not depend on the instance of the class
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()  # Initialize the WordNet lemmatizer
        tokens = nltk.word_tokenize(text)  # Tokenize the input text into individual words
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens] # Lemmatize and convert each token to lowercase

        return tokens  # Return the list of lemmatized tokens
    

    
    def bag_of_tokens(self,tokens):  # Convert a list of tokens into a bag-of-words representation based on the given vocabulary
        return [1 if token in tokens else 0 for token in self.vocabulary] # Return a list of 1s and 0s indicating the presence or absence of each word in the vocabulary
        
    def parse_intents(self): # Parse the intents from the JSON file and populate the documents, vocabulary, intents, and responses lists
        lemmatizer = nltk.WordNetLemmatizer()  # Initialize the WordNet lemmatizer
        if os.path.exists(self.intents_path): # Check if the intents file exists at the specified path using os
            with open(self.intents_path,'r') as f: # Open the intents file in read mode
                intents_data = json.load(f)     # Load the JSON data from the file

                for intent in intents_data['intents']: # Iterate through each intent in the intents data
                    if intent['tag'] not in self.intents: # Check if the intent tag is not already in the intents list
                        self.intents.append(intent['tag']) # Add the intent tag to the intents list
                        self.intents_responses[intent['tag']] =intent['responses'] # Map the intent tag to its corresponding responses 

                    
                    for pattern in intent['patterns']:
                        pattern_tokens = self.tokenize_and_lemmatize(pattern)    # Tokenize and lemmatize the pattern using the static method
                        self.vocabulary.extend(pattern_tokens) # Add the tokens to the vocabulary list
                        self.documents.append((pattern_tokens, intent['tag'])) # Add the tokenized pattern and its corresponding intent tag to the documents list (X,Y)

                    self.vocabulary = sorted(set(self.vocabulary)) # Remove duplicates and sort the vocabulary list using set and sorted functions


    def prepare_data(self):
        bags = []   # List to store the bag-of-words representations of the patterns
        indices = []    # List to store the indices of the corresponding intents

        for documents in self.documents:
            words = documents[0] # Get the tokenized pattern (X)
            bag = self.bag_of_tokens(words) # Convert the tokenized pattern into a bag-of-words representation using the static method
            
            intent_index = self.intents.index(documents[1]) # Get the index of the corresponding intent (Y)

            bags.append(bag) # Add the bag-of-words representation to the bags list
            indices.append(intent_index) # Add the intent index to the indices list
            self.X= np.array(bags)
            self.Y = np.array(indices)


    #A tensor is a mathematical object that generalizes the concepts of scalars, vectors, and matrices to higher dimensions.
    #  In simple terms, in machine learning and deep learning, a tensor is a multi-dimensional array of numbers.

    
    def train_model(self, batch_size,lr,epochs):
        X_tensor = torch.tensor(self.X,dtype=torch.float32) # Convert the input data to a PyTorch tensor of type float32
        Y_tensor = torch.tensor(self.Y,dtype=torch.long) # Convert the target data to a PyTorch tensor of type long (integer)
    
        dataset = TensorDataset(X_tensor,Y_tensor)  # Create a TensorDataset to hold the input and target tensors
        loader = DataLoader(dataset,batch_size=batch_size,shuffle=True) # Create a DataLoader to iterate through the dataset in batches and shuffle the data

        self.model = ChatbotModel(self.X.shape[1], len(self.intents)) # Initialize the ChatbotModel with the input size (number of features) and output size (number of intents currently 5)
        
        criterion = nn.CrossEntropyLoss() # Define the loss function as CrossEntropyLoss for multi-class classification
        optimizer = optim.Adam(self.model.parameters(),lr=lr) # Define the optimizer as Adam with the specified learning rate to update the model parameters during training
         
        for epoch in range(epochs): # Iterate through the specified number of epochs
            running_loss = 0.0 # Initialize the running loss for the epoch

            for batch_X,batch_Y in loader:
                optimizer.zero_grad() # Zero the gradients of the optimizer to prevent accumulation from previous batches
                outputs = self.model(batch_X) # Forward pass: compute the model output for the current batch
                loss = criterion(outputs,batch_Y) # Compute the loss between the model output and the target labels for the current batch
                loss.backward() # Backward pass: compute the gradients
                optimizer.step() # Backward pass: compute the gradients and update the model parameters using the optimizer depending on the learning rate and the loss function

                running_loss += loss.item() # Accumulate the loss for the epoch
            
            print(f"Epoch [{epoch+1}]: Loss : {running_loss/len(loader):.4f}") # Print the average loss for the epoch"

    def save_model(self,model_path,dimensions_path): # Save the trained model and related data to a file
        torch.save(self.model.state_dict(),model_path) # Save the model's state dictionary (parameters) to the specified model path using torch.save
        
        with open(dimensions_path,'w') as f:
            json.dump({ 'input_size' : self.X.shape[1], 'output_size':len(self.intents)},f)
        
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions =  json.load(f)

            self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])  # Initialize the ChatbotModel with the input and output sizes from the dimensions file
            self.model.load_state_dict(torch.load(model_path, weights_only=True))  # Load the model's state dictionary (parameters) from the specified model path using torch.load  
        
    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message) # Tokenize and lemmatize the input message using the static method
        bag = self.bag_of_tokens(words) # Convert the tokenized message into a bag-of-words representation using the static method
        
        bag_tensor = torch.tensor([bag], dtype=torch.float32) # Convert the bag-of-words representation to a PyTorch tensor of type float32
        
        self.model.eval() # Set the model to evaluation mode to disable dropout and batch normalization layers
        with torch.no_grad(): # Disable gradient calculation to save memory and computation during inference
            predictions = self.model(bag_tensor) # Forward pass: compute the model output for the input tensor
        
        predicted_class_index = torch.argmax(predictions , dim=1).item() # Get the index of the class with the highest predicted probability
        predicted_intent = self.intents[predicted_class_index] # Get the predicted intent based on the predicted class index

        if self.function_mappings and predicted_intent in self.function_mappings:
           self.function_mappings[predicted_intent]() # Call the corresponding function if the predicted intent has a mapping in the function_mappings dictionary

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return "I'm sorry, I don't have a response for that."


def get_stocks():
    stocks = {'Apple','Nvidia','Meta','Microsoft','Amazon','Samsung'}
    print( random.sample(stocks,3))


if __name__ == "__main__":
    assistant = ChatbotAssistant('intents.json',function_mappings={'stocks':get_stocks})

    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8,lr=0.001,epochs=100)
    assistant.save_model('chatbot_model.pth','model_dimensions.json')

    while True:
        message = input("Chat with the bot:")

        if message == '/quit':
            break

        print(assistant.process_message(message))







            
            
        
