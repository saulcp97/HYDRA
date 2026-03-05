import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.Similarities as Similarities


def get_cmap(n, name='viridis'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

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


import Config
#Do a 3D MDS plot using the 'iteration' variable as the x axis, and y and z are the projected dimensions.
def plot_MDS_3D():
    #We prepare the data for the MDS.
    prefixDir = "outputs/" + Config.experiment_name + "/"
    hist_list = []

    #For loop for the Config.agent_num agents
    for agent in range(len(Config.AGENT_NAMES)):
        auxiliar = []
        for epoch in range(Config.EPOCH_NUM):
            path = prefixDir + "scp_" + str(agent) + "/iteration_" + str(epoch) + "/weights.pth"
            weights = torch.load(path)
            flat_weights = []
            for key in weights.keys():
                flat_weights.extend(weights[key].flatten().tolist())
            auxiliar.append(flat_weights)
        hist_list.append(auxiliar)

    all_weights = np.vstack(hist_list)

    #Aplicamos el MDS solamente a las dimensiones de los pesos, sin la iteración.
    mds = MDS(n_components=2, metric='euclidean', random_state=42)

    mds.fit(all_weights)

    #We create the matplotlib 3D plot.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    #We plot the points, using the iteration as the x axis, and the MDS dimensions as y and z.
    for i in range(len(all_weights)):
        iteration = i % Config.EPOCH_NUM
        agent = i // Config.EPOCH_NUM

        #x, y, z = iteration, mds.embedding_[i, 0], mds.embedding_[i, 1]

        ax.scatter(iteration, mds.embedding_[i, 0], mds.embedding_[i, 1], 
                    color=Config.coalition_Color_Dictionary[agent], s=100, label=f'Agente {agent+1}' if iteration == 0 else "")
        if iteration == 0:
            ax.text(iteration, mds.embedding_[i, 0] + 0.05, mds.embedding_[i, 1] + 0.05, f'A{agent+1}')
    ax.set_title('Evolución de Pesos en el Espacio MDS')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Dimensión MDS 1')
    ax.set_zlabel('Dimensión MDS 2')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_PCA_3D():
    #We prepare the data for the PCA.
    prefixDir = "outputs/" + Config.experiment_name + "/"
    hist_list = []

    #For loop for the Config.agent_num agents
    for agent in range(len(Config.AGENT_NAMES)):
        auxiliar = []
        for epoch in range(Config.EPOCH_NUM):
            path = prefixDir + "scp_" + str(agent) + "/iteration_" + str(epoch) + "/weights.pth"
            weights = torch.load(path)
            flat_weights = []
            for key in weights.keys():
                flat_weights.extend(weights[key].flatten().tolist())
            auxiliar.append(flat_weights)
        hist_list.append(auxiliar)

    all_weights = np.vstack(hist_list)

    #Aplicamos el PCA solamente a las dimensiones de los pesos, sin la iteración.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(all_weights)


    #X_subset_2d = pca.transform(X_subset)


    #We create the matplotlib 3D plot.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    #We plot the points, using the iteration as the x axis, and the PCA dimensions as y and z.
    for i in range(len(all_weights)):
        iteration = i % Config.EPOCH_NUM
        agent = i // Config.EPOCH_NUM

        ax.scatter(iteration, pca.components_[0][i], pca.components_[1][i], 
                    color=Config.coalition_Color_Dictionary[agent], s=100, label=f'Agente {agent+1}' if iteration == 0 else "")
        if iteration == 0:
            ax.text(iteration, pca.components_[0][i] + 0.05, pca.components_[1][i] + 0.05, f'A{agent+1}')
    ax.set_title('Evolución de Pesos en el Espacio PCA')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Dimensión PCA 1')
    ax.set_zlabel('Dimensión PCA 2')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

#Simulation Plot 3D MDS.
import pandas as pd
def plot_MDS_3D_Simulation_csv():
    cmap = get_cmap(len(Config.AGENT_NAMES))

    hist_list = []
    all_data = []
    #For loop for the Config.agent_num agents

    for agent in range(len(Config.AGENT_NAMES)):
        pathAgent = "outputs/" + Config.experiment_name + "/" + "scp_" + str(agent) + "/expedient.csv"

        df = pd.read_csv(pathAgent)
        
        x_mds = df.iloc[:Config.EPOCH_NUM, 2:]
        hist_list.append([x_mds])
        
        print(f"Agent {agent} loaded.")

    all_weights = np.vstack(hist_list)

    print("Shape:", all_weights.shape)

    mds = MDS(n_components=2, metric='euclidean', init='random', random_state=42, n_init=4, max_iter=300)

    subset = all_weights[:, 1, :]

    print(subset.shape)

    mds.fit(subset)

    #We create the matplotlib 3D plot.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for agent in range(len(Config.AGENT_NAMES)):
        agent_points = all_weights[agent, :, :][:Config.EPOCH_NUM]

        agent_mds = mds.fit_transform(agent_points)
        ax.plot(range(Config.EPOCH_NUM), agent_mds[:, 0], agent_mds[:, 1], c=cmap(agent+1), alpha=0.5)

    ax.set_title('Evolution of Weights in the MDS space')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MDS Dimension 1')
    ax.set_zlabel('MDS Dimension 2')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def plot_spread_optimized():
    all_weights = []
    hist_list = []
    cmap = get_cmap(len(Config.AGENT_NAMES), "hot")
    for agent in range(len(Config.AGENT_NAMES)):
        pathAgent = "outputs/" + Config.experiment_name + "/" + "scp_" + str(agent) + "/expedient.csv"

        #Lecture format:
        #loss (0), density (1), num_neighbors (2), neighbor_names (3-3+num_neighbors), similarities (4+num_neighbors, 4+2*num_neighbors)
        # coalition_length (5+2*num_neighbors), bin_masks (6+2*num_neighbors, 6+3*num_neighbors)

        df = pd.read_csv(pathAgent)
        start_idx = 4 + 3*int(df.iloc[0, 2])

        x_mds = df.iloc[1:1+Config.EPOCH_NUM, start_idx:]
        hist_list.append([x_mds])
        
        #print(f"Agent {agent} loaded with shape:{x_mds.shape}")

    all_weights = np.vstack(hist_list)
    n_agents, n_epochs, n_dims = all_weights.shape
    
    # PASO 1: Centrado respecto al objetivo final
    final_consensus = all_weights[:, -1, :].mean(axis=0)
    centered = all_weights - final_consensus
    
    # PASO 2: Aplanado y Escalado
    # Esto iguala la importancia de todas las dimensiones vectoriales
    flattened = centered.reshape(-1, n_dims)
    scaler = StandardScaler()
    flattened_scaled = scaler.fit_transform(flattened)
    
    # PASO 3: PCA con enfoque en la dispersión
    # Reducimos a 2 componentes para los ejes Y y Z
    pca = PCA(n_components=2)
    reduced_points = pca.fit_transform(flattened_scaled)
    
    # Re-formateamos
    trajectories = reduced_points.reshape(n_agents, n_epochs, 2)
    
    # --- Visualización ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    eps = 1e-16


    # PASO 4: Multiplicador de dispersión (Opcional)
    # Si aún quieres más separación visual, puedes aplicar un factor de escala
    spread_factor = 10
    ax.set(xlim=(0, 100), ylim=(-300, 300), zlim=(-100, 100))
    maxY = -1
    maxZ = -1
    minY = 1000
    minZ = 1000
    for i in range(n_agents):
        iYMin = np.min(trajectories[i, :, 0])
        iZMin = np.min(trajectories[i, :, 1])
        iYMax = np.max(trajectories[i, :, 0])
        iZMax = np.max(trajectories[i, :, 1])
        if minY > iYMin:
            minY = iYMin

        if minZ > iZMin:
            minZ = iZMin

        if maxY < iYMax:
            maxY = iYMax

        if maxZ < iZMax:
            maxZ = iZMax


    for i in range(n_agents):
        x = np.arange(n_epochs)

        #vector_normY = 2 * (trajectories[i, :, 0] - minY) / (maxY - minY) - 1
        #vector_normZ = 2 * (trajectories[i, :, 1] - minZ) / (maxZ - minZ) - 1
        #y = vector_normY * spread_factor
        #z = vector_normZ * spread_factor
        
        y = trajectories[i, :, 0] * spread_factor
        z = trajectories[i, :, 1] * spread_factor
        ax.plot(x, y, z, alpha=0.2, c=cmap(i))

    cmap2 = plt.get_cmap('hot')
    norm = Normalize(vmin=0, vmax=n_agents - 1)

    sm = ScalarMappable(cmap=cmap2, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, aspect=20)
    cbar.set_label('Agent Index', rotation=270, labelpad=15)
    
    ax.set_title('Evolution of Weights in the PCA space')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PCA Dimension 1')
    ax.set_zlabel('PCA Dimension 2')
    
    plt.show()