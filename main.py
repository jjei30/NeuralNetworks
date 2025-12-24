import time #will be used to check how long it took to train the model

import torch
from torch import optim
from torch import nn
import torch.optim as optim

from sklearn.model_selection import train_test_split #will be used to split the data into training and testing sets
#StandardScaler ensures that the number are equally scaled, i.e. if theres a large number it scales it down so it fits with, for example, age
from sklearn.preprocessing import StandardScaler, LabelEncoder #LabelEncoder converts any text within the data to numbers
from sklearn.metrics import accuracy_score, f1_score #f1 score is a type of f score that's more balanced regarding precision and recall

import pandas
import numpy

#setting random seeds while also ensuring that the results are much more producible, i.e., when the network is getting tested it doesnt get random variables each time
#the numbers get changed per every run but while the neural network is running it stays consistent
torch.manual_seed(55)
numpy.random.seed(55)

def loadData(columnsToRemove=None):
    #loading and preparing the csv data
    dataFile = pandas.read_csv("data.csv")

    #removing the first three columns as they're unneeded
    dataFile = dataFile.iloc[:, 3:]
    #incase specific columns are coded to get removed
    if columnsToRemove is not None:
        dataFile = dataFile.drop(columns=columnsToRemove, errors='ignore')


    #getting the x (all columns that are after the region column) and y (the region)
    x = dataFile.iloc[:, 1:].values
    y = dataFile.iloc[:, 0].values
    
    labelEncoder = LabelEncoder()
    y_number = labelEncoder.fit_transform(y)

    #splitting training-testing to 80/20 goldilocks principle-style (and also the most commonly used ratio for training and testing)
    #stratify just ensures that there's the same class distribution in both sets
    X_train, X_test, Y_train, Y_test = train_test_split(x, y_number, test_size=0.2, random_state=55, stratify=y_number)

    #making sure they get scaled to ensure that the neural network performs well
    standardScaler = StandardScaler()
    X_train = standardScaler.fit_transform(X_train) #fit from training, transform the training
    X_test = standardScaler.transform(X_test) #transforming the test using stats gained from the training

    X_train = torch.FloatTensor(X_train) #floattensor is used for continuous features
    X_test = torch.FloatTensor(X_test)
    Y_train = torch.LongTensor(Y_train) #longtensor to store int versions of class labels
    Y_test = torch.LongTensor(Y_test)

    return X_train, X_test, Y_train, Y_test, labelEncoder.classes_



#all the neural networks
class oneD(nn.Module):
    def __init__(self, classes_number):
        super(oneD, self).__init__()

        #first block, 1st channel to 32
        self.conv_layers = nn.Sequential(nn.Conv1d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        #second block, 32 to 64
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        #third block, 64 to 128
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1)
        )
        #connecting layers after convolutional extraction
        self.fc = nn.Sequential(
            nn.Linear(128, 64), #128 features to 64 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, classes_number)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class DFFN(nn.Module):
    def __init__(self, inputSize, classes_number):
        super(DFFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes_number)
        )
    
    def forward(self, x):
        return self.layers(x)
  
class autoEncoder(nn.Module):
    def __init__(self, inputSize, classesNumber, latent_dim=10):
        super(autoEncoder, self).__init__()

        #compressing input into lower-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(inputSize, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        #predicting regions using latent representation
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, classesNumber)
        )

        #decoding to reconstruct the original input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, inputSize),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        classOutput = self.classifier(latent)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, classOutput
        
#using this to track the training of the model
def trainClassificationModel(model, X_train, Y_train, X_test, Y_test, epochs=100):
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr =0.001)

    trainLosses = []
    testAccuracies = []
    testf1Scores = []

    startTime = time.time() #the moment the training starts, the timer starts

    #begin training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train)

        #checking the loss comparing it with actual labels
        loss = criteria(outputs, Y_train)

        loss.backward()

        optimizer.step()

        #evaluation phase
        model.eval()
        with torch.no_grad():
            testOutputs = model(X_test)
            _, prediction = torch.max(testOutputs.data, 1) #predictions
            #also underscore is used for predicted class, aka for throwaway

            #checking the accuracy and f1 score
            accuracy = accuracy_score(Y_test.numpy(), prediction.numpy())
            f1Results = f1_score(Y_test.numpy(), prediction.numpy(), average='weighted')

        trainLosses.append(loss.item())
        testAccuracies.append(accuracy)
        testf1Scores.append(f1Results)

        #printing per 20 epochs
        if(epoch + 1)%20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], loss: {loss.item():.4f}, '
                  f'Test Accuracy: {accuracy:.4f}, F1: {f1Results:.4f}')
    trainingTime = time.time() - startTime
    print(f"Training was completed at: {trainingTime:.2f} seconds")

    return trainLosses, testAccuracies, testf1Scores, trainingTime

#only used to specifically train the auto encoder, the auto encoder would not work with the above function as its usually used as a reconstructor, for the sake of this it would be played with to see if the autoencoder can predict the data
def trainAutoEncoder(model, X_train, Y_train, X_test, Y_test, epochs=100, alpha=0.7): #alpha is used as the weight between class loss & reconstruct loss
    classCriteria = nn.CrossEntropyLoss()
    recCriteria = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainLosses = []
    testAccuracies = []
    testf1Scores = []

    startTime = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        #pass it forward
        reconstructed, _, classOutput = model(X_train)

        classLoss = classCriteria(classOutput, Y_train)
        recLoss = recCriteria(reconstructed, X_train)
        #weighted sum of the losses
        loss = alpha*classLoss + (1-alpha)*recLoss

        loss.backward()
        optimizer.step()

        #evaluating
        model.eval()
        with torch.no_grad():
            _, _, testClass= model(X_test)
            #this is where classification is done 
            _, prediction = torch.max(testClass.data, 1)

            accuracy = accuracy_score(Y_test.numpy(), prediction.numpy())
            f1Results = f1_score(Y_test.numpy(), prediction.numpy(), average='weighted')
        trainLosses.append(loss.item())
        testAccuracies.append(accuracy)
        testf1Scores.append(f1Results)

        if(epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {loss.item():.4f}, '
                    f'Test Accuracy: {accuracy:.4f}, F1: {f1Results:.4f}')
            
    trainingTime = time.time() - startTime
    print(f"Training for Autoencoder was completed at: {trainingTime:.2f} seconds")
    return trainLosses, testAccuracies, testf1Scores, trainingTime

#had some issues when removing data, so this would make the models readjust when data is removed
def createModels(inputSize, numClasses):
    oneDCNN = oneD(numClasses)
    dFFN = DFFN(inputSize, numClasses)
    ae = autoEncoder(inputSize, numClasses)
    return oneDCNN, dFFN, ae


def main():

    #which columns should be removed
    columnsToRemove = ['S_DrugDependency', 'S_AlcoholDependency']
    #Preparing the data
    X_train, X_test, Y_train, Y_test, classNames = loadData(columnsToRemove=columnsToRemove)
    inputSize = X_train.shape[1]
    classes_number = len(classNames)
    
    print(f"Data loaded: {inputSize} features, {classes_number} classes")
    print(f"Columns removed: {columnsToRemove}")
    print(f"Classes: {classNames}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    #putting all of the results in a dictionary
    results = {}

    #creating the models via dynamic input size
    oneDCNN, dFFN, ae = createModels(inputSize, classes_number)

    #Training the model via 1D CNN
    print("\n" + "*"*50)
    print("1D Training Phase")
    print("*"*50)
    oneD_loss, oneD_acc, oneD_f1, oneD_time = trainClassificationModel(
        oneDCNN, X_train, Y_train, X_test, Y_test
    )
    results['oneD'] = {
        'final_accuracy': oneD_acc[-1],
        'final_f1': oneD_f1[-1],
        'training_time': oneD_time,
        'loss_history': oneD_loss,
        'accuracy_history': oneD_acc,
        'f1_history': oneD_f1
    }
    
    #training the model via deep feed
    print("\n" + "*"*50)
    print("Deep Feed-Forward Network Training phase")
    print("*"*50)
    dffn_loss, dffn_acc, dffn_f1, dffn_time = trainClassificationModel(
        dFFN, X_train, Y_train, X_test, Y_test
    )
    results['DFFN'] = {
        'final_accuracy': dffn_acc[-1],
        'final_f1': dffn_f1[-1],
        'training_time': dffn_time,
        'loss_history': dffn_loss,
        'accuracy_history': dffn_acc,
        'f1_history': dffn_f1
    }
    
    #training the model via auto encoder
    print("\n" + "*"*50)
    print("Training Autoencoder")
    print("*"*50)
    ae_train_loss, ae_acc, ae_f1, ae_time = trainAutoEncoder(
        ae, X_train, Y_train, X_test, Y_test
    )
    results['Autoencoder'] = {
        'final_accuracy': ae_acc[-1],
        'final_f1': ae_f1[-1],
        'training_time': ae_time,
        'train_loss_history': ae_train_loss,
        'accuracy_history': ae_acc,
        'f1_history': ae_f1
    }
    
    #printing out the final results after 100 epochs has been reached per 20 epochs including the final accuracy score, the f-1 score, and the time it took to train the neural networks
    print("\n" + "*"*50)
    print("Results Comparison")
    print("*"*50)
    print(f"{'Neural Network':<15} {'Accuracy':<10} {'F1-Score':>9} {'Time (s)':>12}")
    print("=" * 50)
    print(f"{'1D':<15} {results['oneD']['final_accuracy']*100:>8.2f}% {results['oneD']['final_f1']*100:>9.2f}% {results['oneD']['training_time']:>10.2f}")
    print(f"{'DFFN':<15} {results['DFFN']['final_accuracy']*100:>8.2f}% {results['DFFN']['final_f1']*100:>9.2f}% {results['DFFN']['training_time']:>10.2f}")
    print(f"{'Autoencoder':<15} {results['Autoencoder']['final_accuracy']*100:>8.2f}% {results['Autoencoder']['final_f1']*100:>9.2f}% {results['Autoencoder']['training_time']:>10.2f}")
    
    return results

if __name__ == "__main__":
    results = main()



