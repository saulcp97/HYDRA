
import Config

import random
import numpy as np
import torch
import pandas as pd
import mesa

def set_seed(seed: int):
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
 
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SIMULATION_SEED)

from architecture.model import FederatedModel
from utils.plotting import *
from utils.experiments import *

if __name__ == "__main__":
    print("Experimental runs")

    column_names = [
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


    #Config.experiment_name = "Cosine_0.25_2"
    #plotCoalitionStability(0.25, 0)

    NumberExperiments = 5
    nameSetExperiments = "multiTableResults"
    experimentsName = ["Cosine", "Euclidean", "Normalized Euclidean", "Manhattan", "Pearson Correlation", "Angular"]
    similarityM = ["COSINE_SIMILARITY", "EUCLIDEAN_DISTANCE", "NORM_EUCLIDEAN", "MANHATTAN", "PEARSON", "ANGULAR"]
    coalPercentage = [0.1, 0.25, 0.5] #[0] 
    Config.FIXED_THRESHOLD = False
    Config.IID = True
    Config.EPOCH_NUM = 100
    RELOAD = True
    for k in range(NumberExperiments):
        all_rows = []
        for i in range(len(experimentsName)):
            Config.SIMILARITY_MEASURE = similarityM[i]
            Config.IS_SIMILARITY = not Similarities.SimilarityMeasures[similarityM[i]].is_distance
            for j in range(len(coalPercentage)):
                Config.experiment_name = experimentsName[i] + "_" + str(coalPercentage[j]) + "_" + str(k)
                Config.coalition_percentage = coalPercentage[j]

                accuracyPlotViolin(Config.coalition_percentage)
                
                row = [experimentsName[i], coalPercentage[j]]

                set_seed(Config.SIMULATION_SEED)

                starter_model = FederatedModel(len(Config.AGENT_NAMES))
                print("Ammount of agents: ", len(starter_model.agents))
                if RELOAD:
                    # Reload the weights of nnModel inside of every starter_model.agents[i]
                    import os
                    for agent in starter_model.agents:
                        agent_dir = os.path.join("outputs", Config.experiment_name, agent.agent_name)
                        
                        # Find the last iteration directory
                        if os.path.exists(agent_dir):
                            iterations = [d for d in os.listdir(agent_dir) if d.startswith("iteration_")]
                            if iterations:
                                # Sort by iteration number and get the last one
                                iterations.sort(key=lambda x: int(x.split("_")[1]))
                                last_iteration = iterations[-1]
                                weights_path = os.path.join(agent_dir, last_iteration, "weights.pth")
                                
                                # Load the weights if the file exists
                                if os.path.exists(weights_path):
                                    state_dict = torch.load(weights_path, map_location=agent.nnModel.device if hasattr(agent.nnModel, 'device') else torch.device("cpu"))
                                    agent.nnModel.load_state_dict(state_dict)
                                    print(f"Loaded weights for {agent.agent_name} from iteration {last_iteration}")
                else:
                    for epoch in range(Config.EPOCH_NUM):
                        starter_model.step()

                #Experiment interpretation:
                row += calculateExperimentResults()

                print(f"Average Accuracy {row[-4]} +- STD {row[-3]}")
                print(f"Average Area Under the Curve {row[-2]} +- STD {row[-1]}")
                all_rows.append(row)
        df = pd.DataFrame(all_rows, columns=column_names)
        df.to_csv(f"outputs/{nameSetExperiments}_{k}.csv", index=False)

        print("Tests Finished")
        Config.SIMULATION_SEED += 1
    
    files = [f"outputs/{nameSetExperiments}_{i}.csv" for i in range(NumberExperiments)]
    df_list = [pd.read_csv(f) for f in files]

    # 2. Concatenar todos los dataframes en uno solo
    df_total = pd.concat(df_list)

    # 3. Agrupar por el experimento y el umbral para promediar las ejecuciones
    # Agrupamos por las columnas que identifican al experimento de forma única
    group_cols = ["Similarity", "Percentage"]

    # Calculamos la media de todas las columnas numéricas para las 5 ejecuciones
    df_avg = df_total.groupby(group_cols).mean().reset_index()

    # Calculamos la desviación estándar real entre las 5 ejecuciones para las columnas de interés
    df_std = df_total.groupby(group_cols).std().reset_index()

    # 4. Sustituir las columnas que contienen 'std' por la desviación real de las ejecuciones
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
 


    df_avg.to_csv(f"outputs/{nameSetExperiments}_FINAL_CONSOLIDATED.csv", index=False)

    print("Final report generated.")
    #plotCoalitionsOfExperimentList()
    #plotCoalitionsOfExperimentList()
    if Config.SIMULATION_MODE:
        #plot_fast_federated_3d()
        #plot_spread_optimized()
        pass
        #plot_MDS_3D_Simulation_csv()
    else:
        pass
        #plot_MDS_3D()
#f.write('%s\n' %items)


#κ