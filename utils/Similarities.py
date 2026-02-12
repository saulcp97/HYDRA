# Similarity Measures
import torch 
from torch import nn
import torch.nn.functional as F
import Config

#Bruteforced calculation of similarity:
#sim = 0
#for e in list(A.keys()):
#   if len(A[e].shape) > 1: #Esto elimina el Bias
#      sim += Similarities.euclideanDistance(A[e], B[e]).item()

#Apply normalization converts the dictionaries of weights and biases in to flattened vectors.ç
#We need to add the bias to the resulting vector for more accurate calculations
def applyNormalization(dictionary_A, dictionary_B):
    vector_A = torch.tensor([])
    vector_B = torch.tensor([])

    for e in list(dictionary_A.keys()):
        tensor_A = dictionary_A[e]
        tensor_B = dictionary_B[e]

        #Check that all tensors are on the same device
        if tensor_A.device != vector_A.device:
            vector_A = vector_A.to(tensor_A.device)
        if tensor_B.device != vector_B.device: 
            vector_B = vector_B.to(tensor_A.device)
        
        flattened_A = torch.flatten(tensor_A)
        flattened_B = torch.flatten(tensor_B)
        
        vector_A = torch.cat((vector_A, flattened_A), 0)
        vector_B = torch.cat((vector_B, flattened_B), 0)

    return vector_A, vector_B


def cosineSimilarity(A, B):
    normalized_A = F.normalize(A, dim=0)
    normalized_B = F.normalize(B, dim=0)

    output = F.cosine_similarity(normalized_A, normalized_B)
    return torch.mean(torch.flatten(output))

def euclideanDistance(A, B):
    #normalized_A = torch.flatten(F.normalize(A, dim=0))
    #normalized_B = torch.flatten(F.normalize(B, dim=0))
    return torch.sum((A-B)**2)

def pureJaccard(A, B):
    intersections = (A * B).sum()
    union = A.sum() + B.sum() - intersections
    return intersections / union

def indexOfJaccard(A, B, validation, epsilon):
    A.apply_(lambda x: abs(x) > epsilon)
    B.apply_(lambda x: abs(x) > epsilon)

    similar = pureJaccard(A,B)

    return similar

def crossEntropy(A, B):
    output = F.cross_entropy(A, B)
    return output


def applySimilarity(A_dict, B_dict, similarity):
    #The simulation doesnt need normalization only be pytorch.
    if not Config.SIMULATION_MODE:
        A_vector, B_vector = applyNormalization(A_dict, B_dict)
    else:
        A_vector = torch.tensor(A_dict)
        B_vector = torch.tensor(B_dict)
    #Format of the SIMILARITY_MEASURE = "COSINE_SIMILARITY"
    if similarity == "COSINE_SIMILARITY":
        #To avoid the error of "Expected 1-dimensional target for 1-dimensional input, but got target of size [N]" we need to add a dummy dimension to B_vector
        B_vector = B_vector.unsqueeze(0)
        A_vector = A_vector.unsqueeze(0)
        return cosineSimilarity(A_vector, B_vector)
    elif similarity == "EUCLIDEAN_DISTANCE":
        return euclideanDistance(A_vector, B_vector)
    elif similarity == "JACCARD_INDEX":
        return indexOfJaccard(A_vector, B_vector, None, Config.EPSILON)
    elif similarity == "CROSS_ENTROPY":
        return crossEntropy(A_vector, B_vector)


#A = torch.tensor([[0.9091,  0.1296], [-0.3108, -2.4423]])    
#B = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423]])

#print(A)
#print(euclideanDistance(A, B))
#print(cosineSimilarity(A, B))

#A.apply_(lambda x: x > 0.5)

#print(A)