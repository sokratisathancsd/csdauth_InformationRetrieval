import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

class RBFNetwork:
    
    def __init__(self, k, N, load, save, inter):      
        self.HiddenLayer = HiddenNetworkLayer(k, save)
        self.numOfPerceptrons = N
        self.load, self.save, self.inter = load, save, inter
        
    def train(self, Data, maxIters, diff, epochs, b):
        
        print("\n*************** Training Phase ***************\n")
        print("Training RBF Network for %d Data Samples..."%len(Data))

        startTime = time.time()
        
        if self.load:
            self.HiddenLayer.Centers, self.HiddenLayer.sigmas = loadData()
        else:
            self.HiddenLayer.train(Data, maxIters, diff) # Training Hidden Layer via K-Means
            kMeansMinutes, kMeansSeconds = (time.time() - startTime)/60, (time.time() - startTime)%60
            print("\nHidden Layer Training completed in %d minutes and %d seconds."%(kMeansMinutes, kMeansSeconds))
             
        if self.inter:
            Weights = self.HiddenLayer.interpolation(Data, self.numOfPerceptrons)
            self.OutputLayer = OutputNetworkLayer(self.numOfPerceptrons, len(self.HiddenLayer.Centers), Weights)
        else:          
            print("\nTraining Output Layer via Adaline for %d epochs...\n"%epochs)    
            self.OutputLayer = OutputNetworkLayer(self.numOfPerceptrons, len(self.HiddenLayer.Centers))
        
            Epochs, Errors, errorDiff = [], [], 1e-4
        
            for epoch in range(epochs):
            
                epochError, Correct = 0, 0
                random.shuffle(Data) # Randomly shuffle the samples.
            
                for data in Data:   
                
                    Input, label = self.HiddenLayer.phi(data[0]), data[1]
                    D = getGoalsVector(len(self.OutputLayer.Perceptrons), label)     
                    Y = self.OutputLayer.train(Input, D, b)
                
                    epochError += calculateError(D, Y)
                    prediction = np.argmax(Y)
                
                    if prediction == label:
                        Correct += 1
            
                epochError = epochError/len(Data)
            
                Epochs.append(epoch + 1) 
                Errors.append(epochError)
                
                print("Epoch #%d completed: Error = %f, Accuracy = %.2f%%"%(epoch + 1, epochError, Correct*100/len(Data)))
                  
                if (epoch > 0) and (Errors[epoch - 1] - Errors[epoch] < errorDiff):
                    errorDiff, b = errorDiff/10, b/2
                    print("Learning rate b changing to %.4f."%b)
                        
            plotError(Epochs, Errors)
        
        totalTime = time.time() - startTime
        print("\nTraining Phase completed in %d minutes and %d seconds"%(totalTime/60, totalTime%60))
        
    
    def evaluate(self, TestingSet):
        
        print("\n*************** Testing Phase ***************")
        testSize = len(TestingSet)
        print("\nTesting for %d samples...\n"%testSize)
        Correct = 0
        for Dataset in TestingSet:
            Input, label = Dataset[0], Dataset[1]
            prediction = self.predict(Input)
            if prediction == label:
                Correct += 1
        
        print("Successful Predictions Percentage: %.2f%% of the samples."%(Correct*100/testSize))
        print("Unsucessful Predictions Percentage: %.2f%% of the samples."%((testSize - Correct)*100/testSize))
        
    
    def predict(self, x):
        
        Input = self.HiddenLayer.phi(x)
        self.OutputLayer.feedLayer(Input)
        return np.argmax(self.OutputLayer.Output)


class HiddenNetworkLayer():
    
    def __init__(self, k, save):        
        self.k, self.save = k, save
          
    def train(self, Data, maxIters, diff):
        print("\nTraining Hidden Layer via K-Means for k = %d.\n"%self.k)
        self.Centers, self.sigmas = kMeans(Data, self.k, maxIters, diff, self.save)
            
    def phi(self, x):
        
        Output = []
        
        i = 0
        while i < len(self.Centers):
            y = RBFunction(x, self.Centers[i], self.sigmas[i])
            Output.append(y)
            i += 1
        
        return np.array(Output)
    
    
    def interpolation(self, Data, N):
        
        print("\nCalculating Weights for %d Neurons via Interpolation...\n"%N)
       
        Weights = []
    
        for i in range(N):  
            
            A, d = [], []
            
            for data in Data:
                x, label = data[0], data[1]
                y = self.phi(x)
                A.append(y)
                if label == i:
                    d.append(1)
                else:
                    d.append(0)
            
            A = np.linalg.pinv(np.array(A)) # Get the pseudoinverse k x len(Data)
            d = np.array(d) # len(Data) x 1 vector
            Weights.append(A @ d)
            
            print("Neuron #%d done."%(i+1))

        return Weights
    

class Perceptron():
    
    def __init__(self, n, Weights):
        self.x = np.zeros(n) # Input
        self.Weights = Weights # Weight vector
        self.u = 0 # Activation Value
        self.y = 0 # Output
        self.delta = 0 # delta of Perceptron
        
    def setInput(self, Input):
        self.x = Input
        
    def setActivationValue(self, bias):
        self.u = np.dot(self.x, self.Weights) + bias
    
    def setOutput(self): 
        self.y = sigmoid(self.u)

    def feedPerceptron(self, Input):
        self.setInput(Input)
        self.setActivationValue(0)
        self.setOutput()
        
    def setDelta(self, d):
        error = d - self.y # Error of output Perceptron
        phi = sigmoidDerivative(self.u) # Activation Function's Derivative
        self.delta = error*phi
        
    def updateWeights(self, b):
        """
        b: The learning rate of the network
        preOutput: The preOutput vector of the previous layer. Each value
                Output[i] corresponds to the value perceptron(i).y 
                (except for the first hidden layer) where perceptron(i) 
                is a perceptron of the previous layer.
        """
        i = 0
        while i < len(self.Weights):
            self.Weights[i] = self.Weights[i] + b*(self.delta)*self.x[i]
            i += 1
        
    
class OutputNetworkLayer():
    
    def __init__(self, N, inputSize, Weights = []):
        """ 
        N: Number of perceptrons in layer
        inputSize: The input size for each perceptron of the layer
        """
        if len(Weights) == 0:
            Weights = [np.array([random.randint(-100,100)/1000 for i in range(inputSize)]) for i in range(N)]
            
        self.Perceptrons = [Perceptron(inputSize, Weights[i]) for i in range(N)] # List of length N
        self.Output = np.zeros(N) # Vector 1xN
        
            
    def feedLayer(self, Input):
        
        i = 0
        while i < len(self.Perceptrons):
            self.Perceptrons[i].feedPerceptron(Input)
            self.Output[i] = self.Perceptrons[i].y
            i += 1

        
    def setDeltas(self, D):
        """
        D: The goal vector for the specific input, where D[i]
           is the goal for the specific output perceptron
        """
        i = 0
        while i < len(self.Perceptrons):
            self.Perceptrons[i].setDelta(D[i])
            i += 1
            
    
    def updateWeights(self, b):
        i = 0
        while i < len(self.Perceptrons):
            self.Perceptrons[i].updateWeights(b)
            i += 1

    def train(self, Input, D, b):
        
        self.feedLayer(Input)
        self.setDeltas(D)
        self.updateWeights(b)
                
        return self.Output


def kMeans(Data, k, maxIters, diff, save):
    
    nextCenters = chooseCenters(Data, k)
    
    Data = [sample[0] for sample in Data] # Discard the labels
    
    d, iters = 100, 0
    
    while d > diff and iters < maxIters:
        
        Centers = np.array(nextCenters)
        AssignedData = [[] for i in range(k)]
        
        # Search for each sample's closest center
        for x in Data:
            Distances = []
            i = 0
            while i < k:
                Distances.append((i, np.linalg.norm(x - Centers[i]))) # (center, distance)
                i += 1
            
            Distances.sort(key = lambda tup: tup[1])
            cluster = Distances[0][0] # cluster of the minimum distance
            AssignedData[cluster].append(x)
        
        
        # Update Centers
        i = 0
        while i < len(nextCenters):
            if len(AssignedData[i]) > 0:
                nextCenters[i] = sum(AssignedData[i])/len(AssignedData[i])
            i += 1
            
        # Get New Centers difference from the last.
        d = getCentersDifference(Centers, nextCenters, k)
        iters += 1     
        print("K-Means Iteration #%d: Centers Mean Difference = %.3f"%(iters, d))
              
    
    finalCenters, sigmas = [], []
     
    i = 0
    while i < len(nextCenters):
        if len(AssignedData[i]) > 1: # Each center must have at least two samples assigned to it.
            finalCenters.append(nextCenters[i])
            sigmas.append(getSigma(nextCenters[i], AssignedData[i]))
        i += 1
    
    if save:
        saveData(finalCenters, sigmas)
    
    print("\n%d/%d Centers are being used."%(len(finalCenters), k))
    print("Average Center Radius (sigma): %.2f"%(sum(sigmas)/len(sigmas)))
    
    return finalCenters, sigmas


def getCentersDifference(C1, C2, k):
    
    error = 0
    
    for i in range(k):
        error += np.linalg.norm(C1[i] - C2[i])
    
    return error/len(C1)


def getSigma(Center, Data):

    sigma = 0

    for sample in Data:
        temp = np.linalg.norm(Center - sample)
        if temp > sigma:
            sigma = temp    
     
    return sigma


def chooseCenters(Data, k):
    
    Centers, Counters = [], [0 for i in range(2)]

    i = 0
    
    while i < k:
        
        sample = Data[random.randint(0, len(Data) - 1)]
        
        if Counters[sample[1]] <= math.ceil(k//2) and checkIfNotIn(sample, Centers):
            Centers.append(sample[0]) # Append vector
            Counters[sample[1]] += 1 # Increase label's counter
        else:
            continue
        i += 1
    
    return Centers
    

def checkIfNotIn(sample, Centers):
    
    if len(Centers) == 0:
        return True
    
    for center in Centers:
        if np.array_equal(sample, center):
            return False
        
    return True
   
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoidDerivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

def RBFunction(X, C, sigma):
    
    try:
        return math.exp(((-1) * math.pow(np.linalg.norm(X - C), 2))/(math.pow(sigma, 2)))
    except:
        return math.exp(((-1) * math.pow(np.linalg.norm(X - C), 2))/(0.001))
                               
def getGoalsVector(Out, label):
    
    d = np.zeros(Out)
    i = 0
    while i < len(d):
        if i == label:
            d[i] = 0.95
        else:
            d[i] = 0.05
        i += 1
    return np.array(d)

def calculateError(D, Y):
    errorVector = D - Y
    error = math.pow(np.linalg.norm(errorVector), 2)
    return error/2
   
def saveData(Centers, sigmas):
    
    print("\nSaving Data...")
    
    with open('Centers.txt', 'wb') as file:
        pickle.dump(Centers, file)
        
    with open('Sigmas.txt', 'wb') as file:
        pickle.dump(sigmas, file)

def loadData():
   
    print("\nLoading Data...")
    
    try:
        with open('Centers.txt', 'rb') as file:
            Centers = pickle.load(file)
        with open('Sigmas.txt', 'rb') as file:
            sigmas = pickle.load(file)
    except:
        Centers = []
        print("Error: Files not found.")
        
    return Centers, sigmas
           
def plotError(Epochs, Errors):
    
    plt.plot(Epochs, Errors)
    plt.title("Error per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show() 
    time.sleep(2)
    