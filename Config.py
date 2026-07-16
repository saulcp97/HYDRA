# Configuración para el sistema.
import networkx as nx

xmpp_server = "gtirouter.dsic.upv.es"
xmpp_server = "localhost"

DEFAULT_TIMER = 20

BATCH_SIZE = 1024

dirichlet_alpha=0.3

# Max SPADE message body length (aioxmpp limit is 256 * 1024)
max_message_body_length = 150_000

web_port = 5222  
url = "localhost"
jid_domain = "@" + xmpp_server

# FSM name states Central Federado
SETUP_STATE_CFDL = "SETUP_STATE"
STOP_STATE_CFDL = "STOP_STATE"
SEND_STATE_CFDL = "SEND_STATE"

# Data-Set Path
data_set_path = "../data"


# LOGGERS
CONSENSUS_LOGGER = "CONSENSUS_LOGGER"
MESSAGE_LOGGER = "MESSAGE_LOGGER"
WEIGHT_LOGGER = "WEIGHT_LOGGER"
TRAINING_LOGGER = "TRAINING_LOGGER"
EPSILON_LOGGER = "EPSILON_LOGGER"
TRAINING_TIME_LOGGER = "TRAINING_TIME_LOGGER"
COALITION_LOGGER = "COALITION_LOGGER"
EPOCH_NUM = 100
EPOCH_SHARE = 1
NUMBER_OF_AGENTS = 8
coalition_index1_Classes = [0,1,2,3,4]
coalition_index2_Classes = [5,6,7,8,9]

SAVE_FULL_WEIGHTS = False

#   Network Structures (10 miembros)
# Coalition properties
# 2 coalitions 5 members each coalition
n_coalitons = 2
coalition_probability = 0.8 # 0.8  # -1: ACoL; >0: ACoaL
DINAMIC_COALITIONS = True
SIMILARITY_MEASURE = "COSINE_SIMILARITY"
#Boolean var to show if the measure is similarity or distance, to know if we want to maximize or minimize it.
IS_SIMILARITY = False
#Threshold variables
FIXED_THRESHOLD = False
threshold_similarity = 0.9
# K-Neighbors coalition algorithm: percentage of neighbors to include (0-100)
# If 10, only the 10% most similar neighbors are kept for the coalition
coalition_percentage = 0.1

EPSILON = 0.1
DEFAULT_SIMILARITY = 0
coalitions = [["0", "1", "2", "3", "4"], ["5", "6", "7", "8", "9"]]

#IID or Non-IID configuration variables
IID = False

experiment_name = "SINGLE_RUN_EXPERIMENT"
log_Experiment = True
PREFIX = "scp_"

AGENT_NICKNAMES = {}
AGENT_NAMES = [] #["scp_0", "scp_1", "scp_2", "scp_3", "scp_4", "scp_5"]
NEIGHBOURS = {}
Iteration_FALL = 50

#AGENT_NAMES = ["a0", "a1"]
SAVE_NEGATIVE_EPOCH = False


coalition_Color_Dictionary = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "yellow",
    5: "purple",
    6: "cyan",
    7: "magenta",
    8: "brown",
    9: "pink"
}

#Configuration variables for the Simulation Mode.
#Simulation mode changes the agent behavior, so instead of doing actual training they just simulate a fake version
#of gradient descent.
SIMULATION_MODE = False
VECTOR_DIMENSION = 100
RANDOMNESS_SCALE = 0.5
SIMULATION_SEED = 42

#Learning rate
ETA = 0.01

#Network Structures configuration variables
# Net Configuration Path File
path_csv = 'Network_Structures/Connection_1.csv'
network_graph = "graphs/complete.gml"
USE_FILE_GRAPH = False
COMPLETED_GRAPH = True
GRAPH_SEED = 43

#Global variables to use
H = None #The networkX graph
Is_Fall_Node = False
Fall_Node = None

GRAPH_GRADE = 8 #Normally 8
def calcNewGraph():
    global H, GRAPH_GRADE, AGENT_NAMES, NEIGHBOURS, AGENT_NICKNAMES
    global Is_Fall_Node, Fall_Node

    if USE_FILE_GRAPH:
        H = nx.read_graphml(network_graph)
    else:
        if COMPLETED_GRAPH:
            H = nx.complete_graph(NUMBER_OF_AGENTS)
            GRAPH_GRADE = NUMBER_OF_AGENTS - 1
        else:
            H = nx.random_regular_graph(GRAPH_GRADE, NUMBER_OF_AGENTS, seed=GRAPH_SEED)
    
    for x in H.nodes:
        AGENT_NAMES.append(PREFIX+str(x))
        neigh = list(H.neighbors(x))
        NEIGHBOURS[AGENT_NAMES[-1]] = [PREFIX+str(y) for y in neigh]
        AGENT_NICKNAMES[PREFIX+str(x)] = str(x)

    Is_Fall_Node = False
    Fall_Node = AGENT_NAMES[3]

calcNewGraph()