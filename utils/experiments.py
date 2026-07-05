import Config
import os
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


def _read_saved_distances_from_row(row):
    result = {}
    for col in row.index:
        if not isinstance(col, str):
            continue
        if col.startswith("distance_"):
            key = col.replace("distance_", "")
            result[key] = float(row[col])
        elif col.startswith("distance") and col[8:].isdigit():
            key = col.replace("distance", "")
            result[key] = float(row[col])
    return result


def _get_pairwise_distance(agent_i, agent_j, agent_weights, saved_distances):
    if agent_i == agent_j:
        return 0.0

    if agent_i in agent_weights and agent_j in agent_weights:
        return float(np.linalg.norm(agent_weights[agent_i] - agent_weights[agent_j]))

    if agent_i in saved_distances and agent_j in saved_distances[agent_i]:
        return float(saved_distances[agent_i][agent_j])

    if agent_j in saved_distances and agent_i in saved_distances[agent_j]:
        return float(saved_distances[agent_j][agent_i])

    raise KeyError(f"No saved distance between {agent_i} and {agent_j}")


def _coalition_energy_from_names(member_names, agent_weights, saved_distances):
    if not member_names:
        return 0.0

    n = len(member_names)
    sum_sq = 0.0
    for i in range(n):
        for j in range(n):
            d = _get_pairwise_distance(member_names[i], member_names[j], agent_weights, saved_distances)
            sum_sq += d * d
    return sum_sq / n


def _centroid_distance_between_groups(names_a, names_b, agent_weights, saved_distances):
    if not names_a or not names_b:
        return 0.0

    m = len(names_a)
    n = len(names_b)
    sum_a = 0.0
    sum_b = 0.0
    sum_ab = 0.0

    for i in range(m):
        for j in range(m):
            d = _get_pairwise_distance(names_a[i], names_a[j], agent_weights, saved_distances)
            sum_a += d * d

    for i in range(n):
        for j in range(n):
            d = _get_pairwise_distance(names_b[i], names_b[j], agent_weights, saved_distances)
            sum_b += d * d

    for i in range(m):
        for j in range(n):
            d = _get_pairwise_distance(names_a[i], names_b[j], agent_weights, saved_distances)
            sum_ab += d * d

    val = sum_ab / (m * n) - sum_a / (2 * m * m) - sum_b / (2 * n * n)
    return float(np.sqrt(max(val, 0.0)))


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
                            #calcular numero coaliciones reciprocity_matches / 2
                            
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
    agent_distances = {}
    agent_coalitions = {}
    
    # 1. Carga de datos
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[-1]
        all_agents_data[agent_name] = last_row
        
        pathWeights = f"outputs/{Config.experiment_name}/{agent_name}/iteration_{str(Config.EPOCH_NUM-1)}/weights.pth"
        if Config.SIMULATION_MODE:
            agent_weights[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)
        elif os.path.exists(pathWeights):
            weights = torch.load(pathWeights)
            agent_weights[agent_name] = singleNormalization(weights)
        else:
            distance_columns = [col for col in df.columns if isinstance(col, str) and col.startswith("distance_")]
            if distance_columns:
                agent_distances[agent_name] = _read_saved_distances_from_row(last_row)
            else:
                agent_weights[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)

    # --- MÉTRICA 1: Distancia media intra-coalición por agente ---
    # Para cada agente, calculamos la distancia media con sus aliados actuales
    agent_intra_coalition_distances = {}

    for agent_name, row in all_agents_data.items():
        n_count = int(row['num_neighbors'])
        distances = []
        coalition_members = [agent_name]

        for i in range(n_count):
            if row[f'isCoalition{i}'] == 1:
                neighbor_name = row[f'nNames{i}']
                coalition_members.append(neighbor_name)
                try:
                    dist = _get_pairwise_distance(agent_name, neighbor_name, agent_weights, agent_distances)
                    distances.append(dist)
                except KeyError:
                    pass

        agent_coalitions[agent_name] = coalition_members
        agent_intra_coalition_distances[agent_name] = np.mean(distances) if distances else 0.0

    avg_intra_dist_global = np.mean(list(agent_intra_coalition_distances.values()))

    # --- MÉTRICA 2: Distancia entre Centros de Convergencia (Clusters) ---
    # Primero: Encontrar qué agentes forman parte de qué "manada" (comunidades)
    # Una forma sencilla es calcular el "modelo promedio" de cada coalición local
    convergence_vectors = {}

    for agent_name, row in all_agents_data.items():
        coalition_members = [agent_name]
        n_count = int(row['num_neighbors'])
        
        for i in range(n_count):
            if row[f'isCoalition{i}'] == 1:
                neighbor_name = row[f'nNames{i}']
                coalition_members.append(neighbor_name)
        
        agent_coalitions[agent_name] = coalition_members

    # Ahora calculamos la distancia entre estos vectores de convergencia
    # Esto muestra qué tan lejos están las distintas "ideas" o grupos de la red
    conv_names = list(agent_coalitions.keys())
    conv_matrix = []
    for i in range(len(conv_names)):
        for j in range(i + 1, len(conv_names)):
            d = _centroid_distance_between_groups(
                agent_coalitions[conv_names[i]],
                agent_coalitions[conv_names[j]],
                agent_weights,
                agent_distances,
            )
            conv_matrix.append(d)
    
    avg_dist_between_convergences = np.mean(conv_matrix) if conv_matrix else 0.0

    print(f"--- Métricas Avanzadas ---")
    print(f"Distancia media intra-coalición (Cohesión): {avg_intra_dist_global:.4f}±{np.std(list(agent_intra_coalition_distances.values())):.4f}")
    print(f"Distancia media entre puntos de convergencia (Separación): {avg_dist_between_convergences:.4f}±{np.std(conv_matrix):.4f}")
    
    return avg_intra_dist_global, avg_dist_between_convergences


def calculate_coalition_divergence():
    all_agents_data = {}
    weights_dict = {} # Guardaremos los pesos por separado para velocidad
    distances_dict = {}
    # 1. Carga de datos
    for agent_id in range(len(Config.AGENT_NAMES)):
        agent_name = f"scp_{agent_id}"
        path = f"outputs/{Config.experiment_name}/{agent_name}/expedient.csv"
        df = pd.read_csv(path)
        last_row = df.iloc[-1]
        all_agents_data[agent_name] = last_row
        
        pathWeights = f"outputs/{Config.experiment_name}/{agent_name}/iteration_{str(Config.EPOCH_NUM-1)}/weights.pth"
        if Config.SIMULATION_MODE:
            weights_dict[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)
        elif os.path.exists(pathWeights):
            weights = torch.load(pathWeights)
            weights_dict[agent_name] = singleNormalization(weights)
        else:
            distance_columns = [col for col in df.columns if isinstance(col, str) and col.startswith("distance_")]
            if distance_columns:
                distances_dict[agent_name] = _read_saved_distances_from_row(last_row)
            else:
                weights_dict[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)
    
    coalition_energies = [] # Aquí guardaremos el punto 1 para cada coalición

    for agent_name, row in all_agents_data.items():
        has_full_weights = agent_name in weights_dict
        n_count = int(row['num_neighbors'])
        
        # Identificar miembros de la coalición
        coalition_members = []
        for i in range(n_count):
            if row[f'isCoalition{i}'] == 1:
                neighbor_name = row[f'nNames{i}']
                coalition_members.append(neighbor_name)

        # 1) Cálculo de la "Energía" de la coalición (Divergencia local)
        if coalition_members:
            if has_full_weights:
                my_weights = weights_dict[agent_name]
                coal_members_weights = [weights_dict[n] for n in coalition_members if n in weights_dict]
                group_weights = coal_members_weights + [my_weights]
                sum_sq_dist = 0
                for member_w in group_weights:
                    for peer_w in group_weights:
                        sum_sq_dist += np.linalg.norm(member_w - peer_w)**2
                energy = sum_sq_dist / len(group_weights)
            else:
                group_names = [agent_name] + coalition_members
                energy = _coalition_energy_from_names(group_names, weights_dict, distances_dict)
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
    agent_distances = {}
    listOfErrors = []
    listOfAccuracies = []
    listOfAucs = []

    total_coalition_changes = 0
    total_possible_changes = 0
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

        # 'Crude' Weight or distance extraction
        if Config.SIMULATION_MODE:
            agent_weights[agent_name] = last_row[-Config.VECTOR_DIMENSION:].values.astype(float)
        else:
            pathWeights = f"outputs/{Config.experiment_name}/{agent_name}/iteration_{str(Config.EPOCH_NUM-1)}/weights.pth"
            if os.path.exists(pathWeights):
                weights = torch.load(pathWeights)
                agent_weights[agent_name] = singleNormalization(weights)
            else:
                distance_columns = [col for col in df.columns if isinstance(col, str) and col.startswith("distance_")]
                if distance_columns:
                    agent_distances[agent_name] = _read_saved_distances_from_row(last_row)
                else:
                    weight_columns = [col for col in df.columns if isinstance(col, str) and col.startswith("w")]
                    if weight_columns:
                        agent_weights[agent_name] = df.iloc[-1][weight_columns].astype(float).values
                    else:
                        raise FileNotFoundError(
                            f"Missing weights for {agent_name}: {pathWeights} and no flattened CSV weight or distance columns found"
                        )

        coalitions = df.iloc[:,-Config.GRAPH_GRADE:]
        for ep in range(1, Config.EPOCH_NUM):
            if not coalitions.iloc[ep-1].equals(coalitions.iloc[ep]):
                total_coalition_changes += 1
            total_possible_changes += 1 #Se podria precalcular pero lo dejo asi


    # 0.5 Determine the Metrics to return
    finalCoalitionSize, finalCoalitionSizeSTD = 0, 0
    intraCoalitionDist, intraCoalitionDistSTD = 0, 0
    coalitionDivergence, coalitionDivergenceSTD = 0, 0
    convergedDistance, convergedDistanceSTD = 0, 0
    finalTrainError, finalTrainErrorSTD = 0, 0
    reciprocityPercentage = 0

    finalAccuracy, finalAccuracySTD, finalAuc, finalAucSTD = 0, 0, 0, 0


    # Axuiliar Variables
    total_coalition_size = 0
    reciprocity_matches = 0
    total_possible_links = 0
    coaltion_Sizes = []

    agent_intra_coalition_distances = []

    coalition_energies = []

    potential_isolated = set(Config.AGENT_NAMES) # Empezamos con todos

    agent_coalitions = {}
    convergence_vectors = {}
    
    reprocityDict = {}
    # 1. First data processing 
    for agent_name, rows in all_agents_data.items():
        # A. Average Coalition Size
        total_coalition_size += rows['coalition_length']
        coaltion_Sizes.append(rows['coalition_length'])
    
        # B. Reciprocity Percentage
        num_n = int(rows['num_neighbors'])
        neighbor_names_list = rows.iloc[3 : 3 + num_n].values

        if agent_name in agent_weights:
            my_weights = agent_weights[agent_name]
        else:
            my_weights = None

        has_full_weights = my_weights is not None
        has_saved_distances = agent_name in agent_distances

        coalition_members_weights = [my_weights] if has_full_weights else []
        distances = []
        coal_members_weights = []
        for i in range(num_n):
            neighbor_name = neighbor_names_list[i]
            if rows[f'isCoalition{i}'] == 1:
                # Kick out of the probably orphaned agents
                potential_isolated.discard(neighbor_name)
                reprocityDict[neighbor_name] = reprocityDict.get(neighbor_name, 0) + 1

                # Only increase the possible link if it is part of the coalition.
                total_possible_links += 1

                if has_full_weights:
                    neighbor_w = agent_weights[neighbor_name]
                    dist = np.linalg.norm(my_weights - neighbor_w)
                elif has_saved_distances:
                    dist = float(agent_distances[agent_name][neighbor_name])
                else:
                    raise RuntimeError(
                        f"Unable to calculate distance for {agent_name} with neighbor {neighbor_name}: no weights or saved distances available"
                    )

                distances.append(dist)

                if has_full_weights:
                    coal_members_weights.append(agent_weights[neighbor_name])

                # Check if the neighbor has this agent as a coalition member too.
                neighbor_data = all_agents_data[neighbor_name]
                n_num_n = int(neighbor_data['num_neighbors'])
                n_neighbors_list = list(neighbor_data.iloc[3 : 3 + n_num_n].values)

                idx_in_neighbor = n_neighbors_list.index(agent_name)
                n_mask_start = 4 + 2 * n_num_n
                if neighbor_data.iloc[n_mask_start + idx_in_neighbor] == 1:
                    reciprocity_matches += 1

                if has_full_weights:
                    coalition_members_weights.append(agent_weights[neighbor_name])

        coal_members = [neighbor_name for i, neighbor_name in enumerate(neighbor_names_list) if rows[f'isCoalition{i}'] == 1]
        agent_coalitions[agent_name] = [agent_name] + coal_members

        if has_full_weights:
            convergence_vectors[agent_name] = np.mean(coalition_members_weights, axis=0) if coalition_members_weights else None
        else:
            convergence_vectors[agent_name] = None

        if has_full_weights and coal_members_weights:
            group_weights = coal_members_weights + [my_weights]
            sum_sq_dist = 0
            for member_w in group_weights:
                for peer_w in group_weights:
                    sum_sq_dist += np.linalg.norm(member_w - peer_w)**2
            energy = sum_sq_dist / len(group_weights)
            coalition_energies.append(energy)
        elif not has_full_weights and has_saved_distances and coal_members:
            group_names = [agent_name] + coal_members
            energy = _coalition_energy_from_names(group_names, agent_weights, agent_distances)
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

    reciprocityPercentage = (reciprocity_matches / total_possible_links * 100) if total_possible_links > 0 else 0

    intraCoalitionDist = np.mean(agent_intra_coalition_distances)
    intraCoalitionDistSTD = np.std(agent_intra_coalition_distances)

    coalitionDivergence = np.mean(coalition_energies)
    coalitionDivergenceSTD = np.std(coalition_energies)

    # Calculate accuracy and AUC metrics if available
    if listOfAccuracies:
        finalAccuracy = np.mean(listOfAccuracies)
        finalAccuracySTD = np.std(listOfAccuracies)

    if listOfAucs:
        finalAuc = np.mean(listOfAucs)
        finalAucSTD = np.std(listOfAucs)

    conv_names = list(agent_coalitions.keys())
    conv_matrix = []
    for i in range(len(conv_names)):
        for j in range(i + 1, len(conv_names)):
            name_i = conv_names[i]
            name_j = conv_names[j]
            d = _centroid_distance_between_groups(
                agent_coalitions[name_i], agent_coalitions[name_j], agent_weights, agent_distances
            )
            conv_matrix.append(d)
    
    convergedDistance = np.mean(conv_matrix) if conv_matrix else np.nan
    convergedDistanceSTD = np.std(conv_matrix) if conv_matrix else np.nan

    reprocityList = [val / Config.GRAPH_GRADE for val in reprocityDict.values()]


    # Printing the results
    print(f"--- Final Results ---")
    print(f"Average Coalition Size: {finalCoalitionSize:.4f}±{finalCoalitionSizeSTD:.4f}")

    print(f"Distancia media intra-coalición (Cohesión): {intraCoalitionDist:.4f}±{intraCoalitionDistSTD:.4f}")
    
    print(f"Coalition Divergence (Media de densidades): {coalitionDivergence:.4f}±{coalitionDivergenceSTD:.4f}")
    print(f"Distancia media entre puntos de convergencia (Separación): {convergedDistance:.4f}±{convergedDistanceSTD:.4f}")

    print(f"Average Final Error: {finalTrainError:.4f}±{finalTrainErrorSTD:.4f}")
    print(f"Reciprocity Percentage: {reciprocityPercentage:.4f}%")

    if listOfAccuracies:
        print(f"Average Accuracy: {finalAccuracy:.4f}±{finalAccuracySTD:.4f}")
    if listOfAucs:
        print(f"Average AUC: {finalAuc:.4f}±{finalAucSTD:.4f}")

    #training_loss hecho
    #auc (mira papers de FL sobre clasificación)
    return (finalCoalitionSize, finalCoalitionSizeSTD, intraCoalitionDist, intraCoalitionDistSTD,
            coalitionDivergence, coalitionDivergenceSTD, convergedDistance, convergedDistanceSTD,
            finalTrainError, finalTrainErrorSTD, reciprocityPercentage, 0, len(potential_isolated), 0,
            total_coalition_changes/total_possible_changes, 0, np.mean(reprocityList), 0,
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

reportColumnNames = [
    "Similarity", 
    "Percentage", 

    "Avg. Coalition Size",
    "std Avg. Coalition Size",

    "Intra-Coalition Distance",
    "std Intra-Coalition Distance",

    "Coalition Divergence",
    "std Coalition Divergence",
    "Avg. Distance Between Convergence Points",
    "std Avg. Distance Between Convergence Points",
    
    "Avg. Training Error",
    "std Avg. Training Error",
    "Reciprocity (%)",
    "std Reciprocity (%)",
    
    "Isolated Agents",
    "std Isolated Agents",
    
    "Frequence of Changes",
    "std Frequence of Changes",

    "Reprocity",
    "std Reprocity",

    "Accuracy",
    "std Accuracy", 
    "AUC",
    "std AUC",
]
def experimentResultsCSV(header:list, experimentRows:list, nameSetExperiments:str):
    usedData = []

    for i in range(len(experimentRows)):
       usedData.append(header[i] + list(experimentRows[i]))

    df = pd.DataFrame(usedData, columns=reportColumnNames)
    df.to_csv(f"outputs/{nameSetExperiments}.csv", index=False)


def condenseManyExperiments(nameOutput:str, listCSVFiles:list):
    usedData = []

    df_list = [pd.read_csv(f) for f in listCSVFiles]

    df_total = pd.concat(df_list)
    group_cols = ["Similarity", "Percentage"]

    #Mean calculations
    df_avg = df_total.groupby(group_cols).mean().reset_index()
    #Std Calculations
    df_std = df_total.groupby(group_cols).std().reset_index()

    #This is another TODO, but here we filter the columns with std in the name
    for col in df_avg.columns:
        if col.startswith("std "):
            base_col = col.replace("std ", "")
            if base_col in df_std.columns:
                df_avg[col] = df_std[base_col].values

    
    # Calculate the std of Isolated Agents, Frequence of Changes and Reciprocity
    # These are global metrics per experiment, so their std is the std across executions
    isolated_agents_std = df_total.groupby(group_cols)["Isolated Agents"].std().reset_index()
    freq_changes_std = df_total.groupby(group_cols)["Frequence of Changes"].std().reset_index()
    reciprocity_std = df_total.groupby(group_cols)["Reciprocity (%)"].std().reset_index()
    
    reprocity_std = df_total.groupby(group_cols)["Reprocity"].std().reset_index()


    # Update the std columns for these specific metrics
    for idx, row in df_avg.iterrows():
        mask = (isolated_agents_std["Similarity"] == row["Similarity"]) & (isolated_agents_std["Percentage"] == row["Percentage"])
        if mask.any():
            df_avg.loc[idx, "std Isolated Agents"] = isolated_agents_std.loc[mask, "Isolated Agents"].values[0]
        
        mask = (freq_changes_std["Similarity"] == row["Similarity"]) & (freq_changes_std["Percentage"] == row["Percentage"])
        if mask.any():
            df_avg.loc[idx, "std Frequence of Changes"] = freq_changes_std.loc[mask, "Frequence of Changes"].values[0]

        mask = (reciprocity_std["Similarity"] == row["Similarity"]) & (reciprocity_std["Percentage"] == row["Percentage"])
        if mask.any():
            df_avg.loc[idx, "std Reciprocity (%)"] = reciprocity_std.loc[mask, "Reciprocity (%)"].values[0]
 
        mask = (reprocity_std["Similarity"] == row["Similarity"]) & (reprocity_std["Percentage"] == row["Percentage"])
        if mask.any():
            df_avg.loc[idx, "std Reprocity"] = reprocity_std.loc[mask, "Reprocity"].values[0]

    df_avg.to_csv(f"outputs/{nameOutput}_CONSOLIDATED.csv", index=False)