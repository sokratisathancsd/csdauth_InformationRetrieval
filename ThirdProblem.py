import math
import DataManipulation as dm
from textblob import TextBlob
import pickle
import time

def main():
    
    start = time.time()
    
    Queries = loadQueriesDictionary()
    HashBands = loadHashBands()
    
    qid = "1"
    
    NearDuplicates = findNearDuplicates(qid, Queries, HashBands, 0.05)
    
    print("\nFor query:\n\n\t%s"%Queries[qid][0])
    
    if len(NearDuplicates) > 0:
        print("\nNear Duplicates:\n")
        for qid in NearDuplicates:
            print("\t"+Queries[qid][0])
            print()
            
    else:
        print("\nNo Near Duplicates found. Try increasing the threshold.")
        
    print("Execution Time: %.2f seconds"%(time.time() - start))
    
    
              
    
def loadQueriesDictionary():
     
    try:
        with open('Queries.txt', 'rb') as file:
            Queries = pickle.load(file)
    except:
        Queries = []
        print("Error: Files not found.")
        
    return Queries
     
    

def createQueriesDictionary(Data):
    
    InvertedIndex, Queries = dm.readInvertedIndex(), {}

    N = 537933 # Total Number of Queries
    
    for data in Data:
        
        for i in range(1, 3):
            qid, query = data[i], [data[i+2]]
            if qid not in Queries:
                Queries.update({qid:query})
        
    for qid in Queries:

        Words = TextBlob(Queries[qid][0]).lower().words # Dictionary word -> frequency
        
        Hashes, Weights = [], []
        
        try:
            maxf = max(Words.count(w) for w in Words) # Max Frequency of a term in the query
        except:
            continue # Corrupted Data
            
        for w in Words:
            
            Hashes.append(hashFunction(w, 64))
            
            f, n = Words.count(w)/maxf, len(InvertedIndex[w]) # f(t), n(t)
            
            idf = math.log(N/n)/math.log(N) # IDF(t)
            
            Weights.append(f*idf)
        
        
        queryHash = HashQuery(Hashes, Weights)
        Queries[qid].append(queryHash)
        
    
    with open('Queries.txt', 'wb') as file:
        pickle.dump(Queries, file)
            
        
def HashQuery(Hashes, Weights):
    
    QueryHash = [0 for i in range(len(Hashes[0]))]
    
    i = 0
    while i < len(Hashes):
        weight = Weights[i]
        j = 0
        while j < len(Hashes[i]):
            if Hashes[i][j] == "1":
                QueryHash[j] += weight
            else:
                QueryHash[j] -= weight
            j += 1
        i += 1
        
    
    QueryHashString = ""
    
    for h in QueryHash:
        if h >= 0:
            QueryHashString += "1"
        else:
            QueryHashString += "0"
    
    return QueryHashString             
    
         
def hashFunction(word, n):
    
    h = (bin(hash(word))[2:])[1:]
    
    for i in range(len(h), n):
        h = "0" + h # Fill in the zeros
        
    return h





def loadHashBands():
    
    try:
        with open('HashBands.txt', 'rb') as file:
            HashBands = pickle.load(file)
    except:
        HashBands = []
        print("Error: Files not found.")
        
    return HashBands
    
    
def createHashBands(Queries, b):
    
    QueryBands = splitToBands(Queries, b) # b is the number of bands
    
    HashBands = [dict() for i in range(b)]
    
    for qid in QueryBands:
        
        i = 0
        while i < len(QueryBands[qid]):
            
            bandHash = bandHashFunction(QueryBands[qid][i], 16)
            
            if bandHash in HashBands[i]: # If bucket already exists
                HashBands[i][bandHash].append(qid) # Place qid in backet
            else: # Else create Bucket       
                HashBands[i].update({ bandHash: [qid] }) # Containing the qid
            
            i += 1
    
    with open('HashBands.txt', 'wb') as file:
        pickle.dump(HashBands, file)
        

def splitToBands(Queries, b):
    
    r = len(Queries["1"][1])//b # Number of rows in each band
    
    QueryBands = {}
    
    for qid in Queries:
        
        try:
            band, qhash = [], Queries[qid][1]
        except: 
            continue

        for i in range(b):
            band.append(qhash[i*r:(i+1)*r])
        
        QueryBands.update({qid : band})
        
    return QueryBands


def bandHashFunction(bitHash, n):

    toNumber = int(bitHash, 2) + 3651
    
    h = bin(int(toNumber % math.pow(2, n)))[2:]
    
    for i in range(len(h), n):
        h = "0" + h # Fill in the zeros
    
    return h    


def findNearDuplicates(qid, Queries, HashBands, threshold):
    
    b, r = 2, 32
    qhash = Queries[qid][1]
    
    band = []
    
    for i in range(b):
        band.append(qhash[i*r:(i+1)*r])
        
    i = 0
        
    Duplicates = set()
    
    while i < len(band):
            
        bandHash = bandHashFunction(band[i], 16)
        
        if bandHash in HashBands[i]:
            PossibleDuplicates = Duplicates.union(set(HashBands[i][bandHash]))
            
        i += 1
     
    NearDuplicates = []
    
    for dupQid in PossibleDuplicates:
        d = HammingDistance(Queries[qid][1], Queries[dupQid][1])
        if d < threshold and (qid != dupQid):
            NearDuplicates.append(dupQid)
        
            
    return NearDuplicates
     
        

def HammingDistance(h1, h2):
    
    diff, i = 0, 0

    while i < len(h1):
        if h1[i] != h2[i]:
            diff += 1
        i += 1
        
    return diff/len(h1)
    



if __name__== "__main__":
    main()