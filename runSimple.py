
import Config
from architecture.model import FederatedModel
from utils.plotting import *
from utils.experiments import *
if __name__ == "__main__":
    print("Experimental runs")

    #starter_model = FederatedModel(len(Config.AGENT_NAMES))
    #print("Ammount of agents: ", len(starter_model.agents))
    #for epoch in range(Config.EPOCH_NUM):
    #    starter_model.step()
    #Experiment interpretation:

    calculateExperimentResults()
    plotCoalitionsOfExperimentList()
    if Config.SIMULATION_MODE:
        #plot_fast_federated_3d()
        #plot_spread_optimized()
        pass
        #plot_MDS_3D_Simulation_csv()
    else:
        plot_MDS_3D()
#f.write('%s\n' %items)