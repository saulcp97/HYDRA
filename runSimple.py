
import Config
from architecture.model import FederatedModel
from utils.plotting import plot_MDS_3D, plot_MDS_3D_Simulation, plot_PCA_3D

if __name__ == "__main__":
    print("Experimental runs")

    starter_model = FederatedModel(len(Config.AGENT_NAMES))

    print("Ammount of agents: ", len(starter_model.agents))
    
   
    for epoch in range(Config.EPOCH_NUM):
        starter_model.step()

    #Plot using the MDS 3D we just got.
    #plot_MDS_3D()
    #plot_PCA_3D()
    if Config.SIMULATION_MODE:
        plot_MDS_3D_Simulation()
    else:
        plot_MDS_3D()