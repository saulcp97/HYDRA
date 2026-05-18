import Config
import torch
import pandas as pd
import numpy as np


def singleNormalization(dictionary_A):
    vector_A = torch.tensor([])
    for e in list(dictionary_A.keys()):
        tensor_A = dictionary_A[e].cpu()
        flattened_A = torch.flatten(tensor_A)
        vector_A = torch.cat((vector_A, flattened_A), 0)
    return vector_A

def calculate_federated_metrics():
    all_agents_data = {}
    
    # 1. Carga masiva de la última fila de cada agente
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        
        try:
            df = pd.read_csv(path)
            # Nos quedamos solo con la última fila
            last_row = df.iloc[-1]
            all_agents_data[agent_name] = last_row
        except Exception as e:
            print(f"Error cargando datos de {agent_name}: {e}")

    # Variables para acumular resultados
    total_coalition_size = 0
    total_error = 0
    reciprocity_matches = 0
    total_possible_links = 0
    coaltion_Sizes = []
    listOfErrors = []
    # 2. Procesamiento de métricas
    for agent_name, data in all_agents_data.items():
        # A. Average Coalition Size
        total_coalition_size += data['coalition_length']
        coaltion_Sizes.append(data['coalition_length'])
        # B. Average Final Error (asumiendo que la columna se llama 'loss')
        total_error += data['loss']
        listOfErrors.append(data['loss'])
        # C. Reciprocity Percentage
        # Necesitamos saber quiénes son sus vecinos en la coalición
        # Según tu formato: bin_masks empiezan en 6 + 2*num_neighbors
        num_n = int(data['num_neighbors'])
        graph_grade = Config.GRAPH_GRADE
        
        # Extraemos los nombres de los vecinos (están en las columnas 3 a 3+num_n)
        # Es más seguro usar nombres de columnas si los tienes, o iloc:
        neighbor_names_list = data.iloc[3 : 3 + num_n].values
        
        # Extraemos la máscara de coalición (está después de similarities y coal_len)
        mask_start = 4 + 2 * num_n
        coalition_mask = data.iloc[mask_start : mask_start + num_n].values
        
        for i, is_in_coalition in enumerate(coalition_mask):
            if is_in_coalition == 1:
                neighbor_name = neighbor_names_list[i]
                total_possible_links += 1
                
                # Comprobamos reciprocidad: ¿El vecino tiene a este agente en su coalición?
                if neighbor_name in all_agents_data:
                    neighbor_data = all_agents_data[neighbor_name]
                    n_num_n = int(neighbor_data['num_neighbors'])
                    
                    # Buscamos al agente original en la lista de vecinos del vecino
                    n_neighbors_list = list(neighbor_data.iloc[3 : 3 + n_num_n].values)
                    
                    if agent_name in n_neighbors_list:
                        idx_in_neighbor = n_neighbors_list.index(agent_name)
                        n_mask_start = 4 + 2 * n_num_n
                        # Miramos el bit correspondiente en la máscara del vecino
                        if neighbor_data.iloc[n_mask_start + idx_in_neighbor] == 1:
                            reciprocity_matches += 1

    # 3. Cálculos finales
    num_agents = len(all_agents_data)
    avg_coalition_size = total_coalition_size / num_agents
    avg_final_error = total_error / num_agents
    reciprocity_pct = (reciprocity_matches / total_possible_links * 100) if total_possible_links > 0 else 0

    print(f"--- Final Results ---")
    print(f"Average Coalition Size: {avg_coalition_size:.4f}±{np.std(coaltion_Sizes):.4f}")
    print(f"Average Final Error: {avg_final_error:.4f}±{np.std(listOfErrors):.4f}")
    print(f"Reciprocity Percentage: {reciprocity_pct:.4f}%")

    return avg_coalition_size, avg_final_error, reciprocity_pct

def calculate_advanced_metrics():
    all_agents_data = {}
    agent_weights = {} # Guardaremos los pesos por separado para velocidad
    
    # 1. Carga de datos
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[-1]
        all_agents_data[agent_name] = last_row
        
        # Extraer pesos (últimas N columnas)
        agent_weights[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)

    # --- MÉTRICA 1: Distancia media intra-coalición por agente ---
    # Para cada agente, calculamos la distancia media con sus aliados actuales
    agent_intra_coalition_distances = {}

    for agent_name, row in all_agents_data.items():
        my_weights = agent_weights[agent_name]
        n_count = int(row['num_neighbors'])
        distances = []
        
        for i in range(n_count):
            # Si el vecino i está en la coalición
            if row[f'isCoalition{i}'] == 1:
                neighbor_name = row[f'nNames{i}']
                if neighbor_name in agent_weights:
                    neighbor_w = agent_weights[neighbor_name]
                    # Distancia Euclídea
                    dist = np.linalg.norm(my_weights - neighbor_w)
                    distances.append(dist)
        
        # Guardamos la media para este agente (0 si está solo)
        agent_intra_coalition_distances[agent_name] = np.mean(distances) if distances else 0.0

    avg_intra_dist_global = np.mean(list(agent_intra_coalition_distances.values()))

    # --- MÉTRICA 2: Distancia entre Centros de Convergencia (Clusters) ---
    # Primero: Encontrar qué agentes forman parte de qué "manada" (comunidades)
    # Una forma sencilla es calcular el "modelo promedio" de cada coalición local
    convergence_vectors = {}

    for agent_name, row in all_agents_data.items():
        coalition_members_weights = [agent_weights[agent_name]] # Incluirse a sí mismo
        n_count = int(row['num_neighbors'])
        
        for i in range(n_count):
            if row[f'isCoalition{i}'] == 1:
                name_n = row[f'nNames{i}']
                if name_n in agent_weights:
                    coalition_members_weights.append(agent_weights[name_n])
        
        # El "punto de convergencia" para este agente es el promedio de su coalición
        convergence_vectors[agent_name] = np.mean(coalition_members_weights, axis=0)

    # Ahora calculamos la distancia entre estos vectores de convergencia
    # Esto muestra qué tan lejos están las distintas "ideas" o grupos de la red
    conv_names = list(convergence_vectors.keys())
    conv_matrix = []
    for i in range(len(conv_names)):
        for j in range(i + 1, len(conv_names)):
            d = np.linalg.norm(convergence_vectors[conv_names[i]] - convergence_vectors[conv_names[j]])
            conv_matrix.append(d)
    
    avg_dist_between_convergences = np.mean(conv_matrix) if conv_matrix else 0.0

    print(f"--- Métricas Avanzadas ---")
    print(f"Distancia media intra-coalición (Cohesión): {avg_intra_dist_global:.4f}±{np.std(list(agent_intra_coalition_distances.values())):.4f}")
    print(f"Distancia media entre puntos de convergencia (Separación): {avg_dist_between_convergences:.4f}±{np.std(conv_matrix):.4f}")
    
    return avg_intra_dist_global, avg_dist_between_convergences


def calculate_coalition_divergence():
    all_agents_data = {}
    weights_dict = {} # Guardaremos los pesos por separado para velocidad
    # 1. Carga de datos
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[-1]
        all_agents_data[agent_name] = last_row
        
        # Extraer pesos (últimas N columnas)
        weights_dict[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)
    
    coalition_energies = [] # Aquí guardaremos el punto 1 para cada coalición

    for agent_name, row in all_agents_data.items():
        my_weights = weights_dict[agent_name]
        n_count = int(row['num_neighbors'])
        
        # Identificar miembros de la coalición
        coal_members_weights = []
        for i in range(n_count):
            if row[f'isCoalition{i}'] == 1:
                neighbor_name = row[f'nNames{i}']
                if neighbor_name in weights_dict:
                    coal_members_weights.append(weights_dict[neighbor_name])
        
        # 1) Cálculo de la "Energía" de la coalición (Divergencia local)
        if coal_members_weights:
            # Suma de distancias al cuadrado de cada agente con sus vecinos de coalición
            # Nota: Incluimos al propio agente en el grupo para la media
            group_weights = coal_members_weights + [my_weights]
            sum_sq_dist = 0
            
            # Para cada agente en la coalición, calculamos su distancia al cuadrado con el resto
            for member_w in group_weights:
                for peer_w in group_weights:
                    sum_sq_dist += np.linalg.norm(member_w - peer_w)**2
            
            # Dividido por el número de agentes en la coalición (punto 1 de tu explicación)
            energy = sum_sq_dist / len(group_weights)
            coalition_energies.append(energy)
        else:
            coalition_energies.append(0)

    # 2) Media de las densidades coalicionales
    # Suma de todas las energías dividida por la cantidad de coaliciones (agentes)
    mean_coalitional_divergence = np.mean(coalition_energies)

    print(f"Coalition Divergence (Media de densidades): {mean_coalitional_divergence:.4f}±{np.std(coalition_energies):.4f}")
    return mean_coalitional_divergence


def calculateExperimentResults():
    # 0. Average data loading.
    all_agents_data = {}
    agent_weights = {} 
    listOfErrors = []
    listOfAccuracies = []
    listOfAucs = []
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[-1]
        all_agents_data[agent_name] = last_row
        
        listOfErrors.append(np.sum(df["loss"]))
        
        # Load accuracy and AUC if they exist in the CSV
        if "accuracy" in df.columns:
            listOfAccuracies.append(last_row["accuracy"])
        if "auc" in df.columns:
            listOfAucs.append(last_row["auc"])

        # 'Crude' Weight extraction, we
        if Config.SIMULATION_MODE:
            agent_weights[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)
        else:
            pathWeights = f"outputs/{Config.experiment_name}/{agent_name}/iteration_{str(Config.EPOCH_NUM-1)}/weights.pth"
            weights = torch.load(pathWeights)
            agent_weights[agent_name] = singleNormalization(weights)

    # 0.5 Determine the Metrics to return
    finalCoalitionSize, finalCoalitionSizeSTD = 0, 0
    intraCoalitionDist, intraCoalitionDistSTD = 0, 0
    coalitionDivergence, coalitionDivergenceSTD = 0, 0
    convergedDistance, convergedDistanceSTD = 0, 0
    finalTrainError, finalTrainErrorSTD = 0, 0
    finalAccuracy, finalAccuracySTD = 0, 0
    finalAuc, finalAucSTD = 0, 0
    reciprocityPercentage = 0

    # Axuiliar Variables
    total_coalition_size = 0
    reciprocity_matches = 0
    total_possible_links = 0
    coaltion_Sizes = []

    agent_intra_coalition_distances = []

    coalition_energies = []

    convergence_vectors = {}
    # 1. First data processing 
    for agent_name, rows in all_agents_data.items():
        # A. Average Coalition Size
        total_coalition_size += rows['coalition_length']
        coaltion_Sizes.append(rows['coalition_length'])
    
        # C. Reciprocity Percentage
        num_n = int(rows['num_neighbors'])
        neighbor_names_list = rows.iloc[3 : 3 + num_n].values

        # D. Intracoalition Distance
        my_weights = agent_weights[agent_name]

        # E. Convergence Points
        coalition_members_weights = [agent_weights[agent_name]] 
        distances = []
        coal_members_weights = []
        for i in range(num_n):
            neighbor_name = neighbor_names_list[i]
            if rows[f'isCoalition{i}'] == 1:
                #Only increase the possible link if it is par of the coalition.
                total_possible_links += 1

                neighbor_w = agent_weights[neighbor_name]
                dist = np.linalg.norm(my_weights - neighbor_w)
                distances.append(dist)

                coal_members_weights.append(agent_weights[neighbor_name])

                #Check if the neighbor has this agent as a coalition member too.
                neighbor_data = all_agents_data[neighbor_name]
                n_num_n = int(neighbor_data['num_neighbors'])
                n_neighbors_list = list(neighbor_data.iloc[3 : 3 + n_num_n].values)

                #There is no case where the actual agent isnt a neighbor of the coalitioned neighbor.
                idx_in_neighbor = n_neighbors_list.index(agent_name)
                n_mask_start = 4 + 2 * n_num_n
                if neighbor_data.iloc[n_mask_start + idx_in_neighbor] == 1:
                    reciprocity_matches += 1

                coalition_members_weights.append(agent_weights[neighbor_name])
        convergence_vectors[agent_name] = np.mean(coalition_members_weights, axis=0)


        if coal_members_weights:
            # Suma de distancias al cuadrado de cada agente con sus vecinos de coalición
            # Nota: Incluimos al propio agente en el grupo para la media
            group_weights = coal_members_weights + [my_weights]
            sum_sq_dist = 0
            # Para cada agente en la coalición, calculamos su distancia al cuadrado con el resto
            for member_w in group_weights:
                for peer_w in group_weights:
                    sum_sq_dist += np.linalg.norm(member_w - peer_w)**2
            
            # Dividido por el número de agentes en la coalición (punto 1 de tu explicación)
            energy = sum_sq_dist / len(group_weights)
            coalition_energies.append(energy)
        else:
            coalition_energies.append(0)

        agent_intra_coalition_distances.append(np.mean(distances) if distances else 0.0)

    # Final calculations
    num_agents = len(all_agents_data)
    finalCoalitionSize = total_coalition_size / num_agents 
    finalCoalitionSizeSTD = np.std(coaltion_Sizes)
           
    finalTrainError = np.mean(listOfErrors)
    finalTrainErrorSTD = np.std(listOfErrors)
    
    # Calculate accuracy and AUC metrics if available
    if listOfAccuracies:
        finalAccuracy = np.mean(listOfAccuracies)
        finalAccuracySTD = np.std(listOfAccuracies)
    if listOfAucs:
        finalAuc = np.mean(listOfAucs)
        finalAucSTD = np.std(listOfAucs)

    reciprocityPercentage = (reciprocity_matches / total_possible_links * 100) if total_possible_links > 0 else 0

    intraCoalitionDist = np.mean(agent_intra_coalition_distances)
    intraCoalitionDistSTD = np.std(agent_intra_coalition_distances)

    coalitionDivergence = np.mean(coalition_energies)
    coalitionDivergenceSTD = np.std(coalition_energies)

    conv_names = list(convergence_vectors.keys())
    conv_matrix = []
    for i in range(len(conv_names)):
        for j in range(i + 1, len(conv_names)):
            d = np.linalg.norm(convergence_vectors[conv_names[i]] - convergence_vectors[conv_names[j]])
            conv_matrix.append(d)
    
    convergedDistance = np.mean(conv_matrix)
    convergedDistanceSTD = np.std(conv_matrix)

    # Printing the results
    print(f"--- Final Results ---")
    print(f"Average Coalition Size: {finalCoalitionSize:.4f}±{finalCoalitionSizeSTD:.4f}")

    print(f"Distancia media intra-coalición (Cohesión): {intraCoalitionDist:.4f}±{intraCoalitionDistSTD:.4f}")
    
    print(f"Coalition Divergence (Media de densidades): {coalitionDivergence:.4f}±{coalitionDivergenceSTD:.4f}")
    print(f"Distancia media entre puntos de convergencia (Separación): {convergedDistance:.4f}±{convergedDistanceSTD:.4f}")

    print(f"Average Final Error: {finalTrainError:.4f}±{finalTrainErrorSTD:.4f}")
    if listOfAccuracies:
        print(f"Average Accuracy: {finalAccuracy:.4f}±{finalAccuracySTD:.4f}")
    if listOfAucs:
        print(f"Average AUC: {finalAuc:.4f}±{finalAucSTD:.4f}")
    print(f"Reciprocity Percentage: {reciprocityPercentage:.4f}%")
    
    # Return metrics for use in other functions
    return (finalCoalitionSize, finalCoalitionSizeSTD, finalTrainError, finalTrainErrorSTD,
            intraCoalitionDist, intraCoalitionDistSTD, coalitionDivergence, coalitionDivergenceSTD,
            convergedDistance, convergedDistanceSTD, reciprocityPercentage,
            finalAccuracy, finalAccuracySTD, finalAuc, finalAucSTD)


import networkx as nx
def contarCoalicionesOfIter(iter):
    #Count coalitions based on the Bron and Kerbosch algorithm implemented on networkX, find_cliques.

    #Once again we reload the results of the experiments and pour the data on a dictionary
    all_agents_data = {}
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[iter]
        all_agents_data[agent_name] = last_row


    isMutuallyCoallitioned = {}
    nodeNames = []
    for agent_name, rows in all_agents_data.items():
        nodeNames.append(agent_name)
        isMutuallyCoallitioned[agent_name] = []
        num_n = int(rows['num_neighbors'])
        neighbor_names_list = rows.iloc[3 : 3 + num_n].values
        for i in range(num_n):
            neighbor_name = neighbor_names_list[i]
            if rows[f'isCoalition{i}'] == 1:
                neighbor_data = all_agents_data[neighbor_name]
                n_num_n = int(neighbor_data['num_neighbors'])
                n_neighbors_list = list(neighbor_data.iloc[3 : 3 + n_num_n].values)

                #There is no case where the actual agent isnt a neighbor of the coalitioned neighbor.
                idx_in_neighbor = n_neighbors_list.index(agent_name)
                n_mask_start = 4 + 2 * n_num_n
                if neighbor_data.iloc[n_mask_start + idx_in_neighbor] == 1:
                    isMutuallyCoallitioned[agent_name].append(neighbor_name)

    auxG = nx.Graph()
    auxG.add_nodes_from(nodeNames)

    for agent, neighbors in isMutuallyCoallitioned.items():
    # Aquí añadirías solo las conexiones positivas de tu loop original
    # G_amigos.add_edge(agent, amigo_confirmado
        for confirmedNeighbor in neighbors:
            auxG.add_edge(agent, confirmedNeighbor)

    connectedComponents = nx.number_connected_components(auxG)
    print(f"Number of unique Coalitions: {connectedComponents} in epoch {iter} connected components way")

    #comunidades = nx.community.louvain_communities(auxG)
    #num_grupos = len(comunidades)
    #print(f"Number of unique Coalitions: {num_grupos} in epoch {iter} louvain communities way")
    return connectedComponents

def contarCoalicionesOfIterForExp(iter, exp):
    #Count coalitions based on the Bron and Kerbosch algorithm implemented on networkX, find_cliques.

    #Once again we reload the results of the experiments and pour the data on a dictionary
    all_agents_data = {}
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{exp}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[iter]
        all_agents_data[agent_name] = last_row


    isMutuallyCoallitioned = {}
    nodeNames = []
    for agent_name, rows in all_agents_data.items():
        nodeNames.append(agent_name)
        isMutuallyCoallitioned[agent_name] = []
        num_n = int(rows['num_neighbors'])
        neighbor_names_list = rows.iloc[3 : 3 + num_n].values
        for i in range(num_n):
            neighbor_name = neighbor_names_list[i]
            if rows[f'isCoalition{i}'] == 1:
                neighbor_data = all_agents_data[neighbor_name]
                n_num_n = int(neighbor_data['num_neighbors'])
                n_neighbors_list = list(neighbor_data.iloc[3 : 3 + n_num_n].values)

                #There is no case where the actual agent isnt a neighbor of the coalitioned neighbor.
                if agent_name in n_neighbors_list:
                    idx_in_neighbor = n_neighbors_list.index(agent_name)
                    n_mask_start = 4 + 2 * n_num_n
                    if neighbor_data.iloc[n_mask_start + idx_in_neighbor] == 1:
                        isMutuallyCoallitioned[agent_name].append(neighbor_name)

    auxG = nx.Graph()
    auxG.add_nodes_from(nodeNames)

    for agent, neighbors in isMutuallyCoallitioned.items():
    # Aquí añadirías solo las conexiones positivas de tu loop original
    # G_amigos.add_edge(agent, amigo_confirmado
        for confirmedNeighbor in neighbors:
            auxG.add_edge(agent, confirmedNeighbor)

    connectedComponents = nx.number_connected_components(auxG)
    print(f"Number of unique Coalitions: {connectedComponents} in epoch {iter} connected components way")

    #comunidades = nx.community.louvain_communities(auxG)
    #num_grupos = len(comunidades)
    #print(f"Number of unique Coalitions: {num_grupos} in epoch {iter} louvain communities way")
    return connectedComponents


import matplotlib.pyplot as plt
def plotCoalitionsOfExperiment():
    dataExperiment = []

    for i in range(Config.EPOCH_NUM):
        dataExperiment.append(contarCoalicionesOfIter(i))

    fig, ax = plt.subplots()
    ax.plot(dataExperiment)
    plt.show()


def plotCoalitionsOfExperimentList():

    measuresNames = ["MSE"]
    experimentList = [
        "experiment_FMNIST_MSE"
    ]

    dataExperiment = {}
    for mesureIndex in range(len(measuresNames)):
        dataExperiment[measuresNames[mesureIndex]] = []
        for i in range(Config.EPOCH_NUM):
            dataExperiment[measuresNames[mesureIndex]].append(contarCoalicionesOfIterForExp(i, experimentList[mesureIndex]))

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(Config.EPOCH_NUM)
    for mesureIndex in range(len(measuresNames)):
        m_name = measuresNames[mesureIndex]
        ax.plot(epochs, dataExperiment[m_name], label=m_name, linewidth=2)


    # --- ESTÉTICA Y LEYENDA ---
    ax.set_title("Evolution of number of Coalitions", fontsize=14)
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Number of Coalitions", fontsize=12)
    
    # Añadimos rejilla para seguir mejor los cambios
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # La leyenda: loc='best' busca el hueco vacío automáticamente
    ax.legend(title="Similarity Measures", loc='upper left', frameon=True)

    plt.tight_layout() # Ajusta márgenes automáticamente

    plt.show()