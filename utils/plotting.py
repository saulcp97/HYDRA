import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.Similarities as Similarities


def loadWeights(path_a: str, path_b: str):
    A = torch.load(path_a)
    B = torch.load(path_b)

    method = "ADD"
    mesure = "euclidean"

    if mesure == "cosine":
        if method == "ADD":
            sim = 0
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    sim += Similarities.cosineSimilarity(A[e], B[e]).item()

            return sim
        elif method == "AVERAGE":
            results = []
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    results.append(Similarities.cosineSimilarity(A[e], B[e]).item())
            return np.average(results)
        
    elif mesure == "euclidean":
        if method == "ADD":
            sim = 0
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    sim += Similarities.euclideanDistance(A[e], B[e]).item()

            return -sim
        elif method == "AVERAGE":
            results = []
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    results.append(Similarities.euclideanDistance(A[e], B[e]).item())
            return -np.average(results)

    elif mesure == "jaccard":
        if method == "ADD":
            sim = 0
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    sim += Similarities.indexOfJaccard(A[e], B[e], 0, 1e-6)

            return sim
        elif method == "AVERAGE":
            results = []
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    results.append(Similarities.indexOfJaccard(A[e], B[e], 0, 1e-6))
            return np.average(results)
    
    elif mesure == "ENTROPY":
        if method == "ADD":
            sim = 0
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    sim += Similarities.crossEntropy(A[e], B[e])
            return sim
        
        elif method == "AVERAGE":
            results = []
            for e in list(A.keys()):
                if len(A[e].shape) > 1: #Esto elimina el Bias
                    results.append(Similarities.crossEntropy(A[e], B[e]))


def distDifference(x, y):
    return -abs(x-y)

def plotCovariance(elements, maxNum: int, differenceFunction, graphName):
    """
    Genera y visualiza un Mapa de Calor de similitud/diferencia entre elementos.

    :param elements: Lista de los 100 elementos a comparar.
    :param maxNum: El número total de elementos (debe ser len(elements)).
    :param differenceFunction: La función para calcular la relación entre dos elementos.
    :param graphName: Título para el gráfico.
    """
    if len(elements) < maxNum:
        print(f"Error: La lista de elementos debe tener {maxNum} elementos, no {len(elements)}.")
        return


    # 1. Crear la Matriz de Relación (100x100)
    # Inicializa la matriz con ceros
    relation_matrix = np.zeros((maxNum, maxNum))

    # Itera para calcular la relación para CADA par de elementos (i, j)
    for i in range(maxNum):
        for j in range(maxNum):
            # Tu función se usa para llenar la matriz
            relation_matrix[i, j] = differenceFunction(elements[i], elements[j])

    # 2. Visualizar la Matriz con un Mapa de Calor (Heatmap)
    
    plt.figure(figsize=(10, 8)) # Ajusta el tamaño para 100 elementos

    # Mapa de calor
    plt.imshow(relation_matrix)
    
    plt.title(f'Mapa de Calor de la Matriz de Relación: {graphName}')
    plt.xlabel('Elemento (j)')
    plt.ylabel('Elemento (i)')
    plt.xticks(np.arange(0, maxNum, 10), np.arange(1, maxNum + 1, 10))
    plt.yticks(np.arange(0, maxNum, 10), np.arange(1, maxNum + 1, 10), rotation=0)
    plt.show()


from sklearn.manifold import MDS
#import sklearn

def plot_MDS_Points(distanceMatrix, colorList):
    mds = MDS(n_components=2, metric='precomputed', random_state=42)
    posiciones_2d = mds.fit_transform(distanceMatrix)

    plt.figure(figsize=(10, 7))

    
    #TODO in theory we can take the positions of every agent, get the original graph and draw the connections
    for i in range(len(posiciones_2d)):
        plt.scatter(posiciones_2d[i, 0], posiciones_2d[i, 1], 
                    color=colorList[i], s=100, label=f'Punto {i+1}')
        plt.text(posiciones_2d[i, 0] + 0.05, posiciones_2d[i, 1] + 0.05, f'P{i+1}')

    plt.title('Representación Relativa de Puntos (MDS)')
    plt.xlabel('Dimensión proyectada 1')
    plt.ylabel('Dimensión proyectada 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()