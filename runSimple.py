
import Config
from architecture.model import FederatedModel
from utils.plotting import plot_MDS_3D, plot_MDS_3D_Simulation, plot_PCA_3D, plot_MDS_3D_Simulation_csv

if __name__ == "__main__":
    print("Experimental runs")

    starter_model = FederatedModel(len(Config.AGENT_NAMES))

    print("Ammount of agents: ", len(starter_model.agents))
    
   
    for epoch in range(Config.EPOCH_NUM):
        starter_model.step()

    
    if Config.SIMULATION_MODE:
        plot_MDS_3D_Simulation_csv()
    else:
        plot_MDS_3D()