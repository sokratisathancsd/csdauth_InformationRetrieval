import csv
from textblob import TextBlob
import math
import numpy as np

# Data[i] = [id, q1_id, q2_id2, q1, q2, label]

Banned = ["the", "a", "an", "of", "is", "are", "do", "does", "so", "to", 
          "very", "i", "you", "am", "my", "be", "how", "what", "can",
          "which", "this", "that", "may", "could", "would", "in", "on", 
          "by", "if", "while", "it", "and", "why", "will", "when", "me",
          "was", "were", "your", "its", "us", "for", "should", "'s", "s",
          "at", "her", "she", "his", "he", "our", "we", "those", "these",
          "there", "any", "some", "much", "or", "in", "out", "where", "with",
          "as", "with", "who", "did", "do"]


def extractData():
    
    ''' Read Data from .CSV File '''
    print("\nReading data from .csv file...")
    with open('train_original.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')  
        Data = [row for row in reader]
      
    print("Total Entries Read: %d"%(len(Data) - 1))
    return Data[1:]  # Don't need the first row
        


def readInvertedIndex():
    
    InvertedIndex = {}
    
    with open('InvertedIndex.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            word, queries = line.split("::")
            Q = [int(n) for n in queries.split(",")]
            InvertedIndex.update({word:Q})
        
    return InvertedIndex



def writeInvertedIndex(Data):
      
    Queries, InvertedIndex = {}, {}
    
    for data in Data:
        
        for i in range(1, 3):
            qid, query = data[i], data[i+2]
            if qid not in Queries:
                Queries.update({qid:query})
         
    for qid in Queries:
        
        words = TextBlob(Queries[qid]).lower().words
        
        for w in words:
            if w not in InvertedIndex:
                InvertedIndex.update({w:[qid]})
            else:
                InvertedIndex[w].append(qid)
                    
    
    with open('InvertedIndex.txt', 'w', encoding='utf-8') as textFile:
        for word in InvertedIndex:
            line = ','.join(str(x) for x in InvertedIndex[word])
            textFile.write("%s::%s"%(word, line+"\n"))
  


def dataToVectors(Data):
    
    print("\nConverting %d Data samples to vectors..."%len(Data))
    
    Vectors = []
    InvertedIndex = readInvertedIndex()
    
    Terms = [term for term in InvertedIndex]
    i = 0
    
    for data in Data:
        
        blob1, blob2 = TextBlob(data[3]).lower(), TextBlob(data[4]).lower()     
        
        d1 = queryLength(blob1, blob2)
        d2 = cosineSimilarity(blob1, blob2, InvertedIndex, Terms)
        d3, d4, d5 = compareTags(blob1, blob2)
        Vectors.append((np.array((d1, d2, d3, d4, d5)), int(data[5])))
        
    
    print("Data to Vector Convertion completed.")
    
    return Vectors
          
            

def queryLength(blob1, blob2) :
    
    c1, c2 = len(blob1.tokens), len(blob2.tokens)       
    return 1 - abs(c1 - c2)/max(c1, c2)
           


def cosineSimilarity(blob1, blob2, InvertedIndex, Terms):
    
    N = 537933 # Total Number of Queries

    Blobs = [blob1, blob2]
    
    Vectors = []
    
    for i in range(2):
        
        v = np.zeros(len(InvertedIndex)) # Initialize Vector

        Words = Blobs[i].words # Dictionary word -> frequency
        
        try:
            maxf = max(Words.count(w) for w in Words) # Max Frequency of a term in the query
        except:
            return 0.5 # Corrupted Data

        for w in Words:
            
            f, n = Words.count(w)/maxf, len(InvertedIndex[w]) # f(t), n(t)
            
            idf = math.log(N/n)/math.log(N) # IDF(t)
            
            v[Terms.index(w)] = f*idf 
        
        Vectors.append(v)
                
    return (Vectors[0] @ Vectors[1])/(np.linalg.norm(Vectors[0])*np.linalg.norm(Vectors[1]))
        
    

def compareTags(blob1, blob2):
    
    Tags, NN, VB, JJ = [blob1.tags, blob2.tags], [[], []], [[], []], [[], []]
    
    for i in range(2):
        for t in Tags[i]:
            if "NN" in t[1] and t[0] not in NN[i]:
                NN[i].append(t[0])        
        NN[i] = set(NN[i])
    
    try:
        d3 = len(NN[0].intersection(NN[1]))/len(NN[0].union(NN[1]))
    except:
        d3 = 0.5
    
    for i in range(2):
        
        for t in Tags[i]:
            if "VB" in t[1] and t[0] not in VB[i]:
                VB[i].append(t[0])        
        VB[i] = set(VB[i])
    
    
    try:
        d4 = len(VB[0].intersection(VB[1]))/len(VB[0].union(VB[1]))
    except:
        d4 = d3
    
    for i in range(2):
        
        for t in Tags[i]:
            if "JJ" in t[1] and t[0] not in JJ[i]:
                JJ[i].append(t[0])        
        JJ[i] = set(JJ[i])
    
    try:
        d5 = len(JJ[0].intersection(JJ[1]))/len(JJ[0].union(JJ[1]))
    except:
        d5 = (d3 + d4)/2
    
    return d3, d4, d5
