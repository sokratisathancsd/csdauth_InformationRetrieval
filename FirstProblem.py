import DataManipulation as dm
import numpy as np
import time
import RBFNN as rbf

def main():
    
    Data = dm.extractData()
    Vectors = dm.dataToVectors(Data[:2000]) # Read First 2000 lines

    TrainingSet = Vectors[:1500] # First 1500 Training Data
    TestingSet = Vectors[1500:] # Last 500 Testing Data

    #testRBF(TrainingSet, TestingSet) # Test RBF Network
    kNearestNeighbors(TrainingSet, TestingSet, 50) # Test kNearestNeighbors


def kNearestNeighbors(TrainingSet, TestingSet, k):
    
    startTime = time.time()
     
    loops, Correct = 0, 0
      
    for test in TestingSet:
        
        Classes = np.zeros(2) # Possible Classes |Even or Odd| = 2
        Distances = []
        for train in TrainingSet:
            Distances.append((np.linalg.norm(train[0] - test[0]), train[1])) # (distance, label)
        Distances.sort(key = lambda tup: tup[0]) # sort the tuples by the distance
        for d in Distances[0:k]: # for k nearest neighbours
            Classes[d[1]] += 1 # find which class most neighbours belong in
        prediction = np.argmax(Classes) # get the label of the class with most k neighbours in it
        if prediction == test[1]: # if label of test is equal to label of class
            Correct += 1
        loops += 1
        
        if loops == 30:
            runtimeEstimation(time.time() - startTime, len(TestingSet), loops)
    
    totalTime = time.time() - startTime
    
    if totalTime >= pow(60,2):
        print("\nk Nearest Neighbors Algorithm completed in %d hours and %d minutes"%(totalTime/pow(60,2), (totalTime/60)%60))
    else:
        print("\nk Nearest Neighbors Algorithm completed in %d minutes and %d seconds"%(totalTime/60, totalTime%60))
    
    testSize = len(TestingSet)
    print("\nSuccessful Predictions Percentage: %.2f%% of the samples"%(Correct*100/testSize))
    print("Unsucessful Predictions Percentage: %.2f%% of the samples"%((testSize - Correct)*100/testSize))



def runtimeEstimation(estTime, setSize, loops):
    
    estTime = estTime*setSize/loops
    
    if estTime >= pow(60,2):
        print("\nRuntime Estimation: %d hours and %d minutes"%(estTime/pow(60,2), (estTime/60)%60))
    else:
        print("\nRunime Estimation: %d minutes and %d seconds"%(estTime/60, estTime%60))
        
        
def testRBF(TrainingSet, TestingSet):
    
    RB = rbf.RBFNetwork(100, 2, load = False, save = False, inter = False)
    RB.train(TrainingSet, 20, 0.001, 30, 0.5)
    RB.evaluate(TestingSet)


if __name__== "__main__":
    main()