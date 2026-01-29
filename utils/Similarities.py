# Similarity Measures
import torch 
from torch import nn
import torch.nn.functional as F

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

#A = torch.tensor([[0.9091,  0.1296], [-0.3108, -2.4423]])    
#B = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423]])

#print(A)
#print(euclideanDistance(A, B))
#print(cosineSimilarity(A, B))

#A.apply_(lambda x: x > 0.5)

#print(A)