# Similarity Measures
import torch 
from torch import nn
import torch.nn.functional as F

from utils.similarity_2 import *
SimilarityMeasures = {
    "COSINE_SIMILARITY": CosineSimilarityMeasure(),
    "EUCLIDEAN_DISTANCE": EuclideanDistanceMeasure(),
    "NORM_EUCLIDEAN": NormalizedEuclideanDistanceMeasure(),
    "MANHATTAN": ManhattanDistanceMeasure(),
    "PEARSON": PearsonCorrelationMeasure(),
    "ANGULAR":AngularDistanceMeasure(),
}

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

#For design this should in theory only being used by
def singleNormalization(dictionary_A):
    vector_A = torch.tensor([])
    for e in list(dictionary_A.keys()):
        tensor_A = dictionary_A[e]
        if tensor_A.device != vector_A.device:
            vector_A = vector_A.to(tensor_A.device)
        flattened_A = torch.flatten(tensor_A)
        vector_A = torch.cat((vector_A, flattened_A), 0)
    return vector_A

def cosineSimilarity(A, B):
    normalized_A = F.normalize(A, dim=0)
    normalized_B = F.normalize(B, dim=0)

    output = F.cosine_similarity(normalized_A, normalized_B)
    return torch.mean(torch.flatten(output))

def euclideanDistance(A, B):
    return torch.sqrt(torch.sum((A-B)**2))

def normalizedEuclidean(A, B):
    A_norm = F.normalize(A, p=2, dim=0)
    B_norm = F.normalize(B, p=2, dim=0)
    return torch.sqrt(torch.sum((A_norm - B_norm) ** 2))

def manhattanDistance(A, B):
    output = torch.abs(A - B)
    return torch.sum(output)



def pureJaccard(A, B):
    intersections = (A * B).sum()
    union = A.sum() + B.sum() - intersections
    return intersections / union

def indexOfJaccard(A, B, validation, epsilon):
    Ab = A.cpu()
    Bb = B.cpu()
    Ab.apply_(lambda x: abs(x) > epsilon)
    Bb.apply_(lambda x: abs(x) > epsilon)
    similar = pureJaccard(Ab, Bb)

    return similar

def crossEntropy(A, B):
    output = F.cross_entropy(A, B)
    return output


def mse(A, B):
    return F.mse_loss(A, B)

from scipy.stats import wasserstein_distance
import numpy as np
def normalized_gemd(w1, w2):
    # Normalización Min-Max para asegurar que los pesos 
    # se comporten como una distribución "comparable"
    w1_norm = (w1 - np.min(w1)) / (np.max(w1) - np.min(w1))
    w2_norm = (w2 - np.min(w2)) / (np.max(w2) - np.min(w2))
    return wasserstein_distance(w1_norm, w2_norm)

def applySimilarity(A_dict, B_dict, similarity, simulation_mode):
    # The simulation mode should be passed explicitly from the FederatedModel config.
    if not simulation_mode:
        A_vector, B_vector = applyNormalization(A_dict, B_dict)
    else:
        A_vector = torch.tensor(A_dict)
        B_vector = torch.tensor(B_dict)
    # Format of the SIMILARITY_MEASURE = "COSINE_SIMILARITY"
    return SimilarityMeasures[similarity].compute(A_vector, B_vector)


#A = torch.tensor([[0.9091,  0.1296], [-0.3108, -2.4423]])    
#B = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423]])

#print(A)
#print(euclideanDistance(A, B))
#print(cosineSimilarity(A, B))

#A.apply_(lambda x: x > 0.5)

#print(A)