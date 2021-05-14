import DataManipulation as dm
import numpy as np
import math

def main():
    
    Data = dm.extractData()

    Vectors = dm.dataToVectors(Data[:2000]) # First 2000 rows
    predict(Vectors)


def predict(Vectors):
    
    accuracy, d = 0, math.sqrt(5)
    
    for v in Vectors:
        
        p = np.linalg.norm(v[0])/d
        accuracy += CrossEntropy(p, v[1])

    print("\nAccuracy: %.2f%%"%(accuracy*100/len(Vectors)))


def CrossEntropy(p, y):
    
    if p == 1:
        p = 0.999999
   
    return -(y*math.log(p) + (1-y)*math.log(1-p))


if __name__== "__main__":
    main()