import numpy as np
import matplotlib.pyplot as plt
import matplotlib


cm = plt.cm.get_cmap('viridis')
bi = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ff0080","#ff0080","#a349a4","#0000ff","#0000ff"]) 
tr = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#55CDFC","#55CDFC","#F7A8B8","#F7A8B8","#FFFFFF","#FFFFFF"])
ls = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D62900","#FF9B55","#FFFFFF","#D461A6","#A50062"])


"""
x1 = np.linspace(-10.0, 10.0, 100)
x2 = np.linspace(-10.0, 10.0, 100)

X1, X2 = np.meshgrid(x1, x2)
Y = np.sqrt(np.square(X1) + np.square(X2))


# I dont like the viridis colormap but this https://www.youtube.com/watch?v=xAoljeRJ3lU convinced me for now.
cm = plt.cm.get_cmap('viridis')
bi = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ff0080","#ff0080","#a349a4","#0000ff","#0000ff"]) 
tr = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#55CDFC","#55CDFC","#F7A8B8","#F7A8B8","#FFFFFF","#FFFFFF"])
ls = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#D62900","#FF9B55","#FFFFFF","#D461A6","#A50062"])


cp = plt.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
plt.clabel(cp, inline=1, fontsize=10)
cp = plt.contourf(X1, X2, Y, cmap=tr)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch

pesos_history = None  # Simulated weights history (100 epochs, 10 parameters)
#load the weights history from outputs/baseline/weights and loop for the 100 first epochs to fill the array with the flattened dictionary weights.

prefixDir = "outputs/baseline/weights/"
prefixFile = "weight_"
colorList =[]
auxiliar = []
for i in range(100):
    path = prefixDir + prefixFile + str(i) + ".pth"
    weights = torch.load(path)
    flat_weights = []
    for key in weights.keys():
        flat_weights.extend(weights[key].flatten().tolist())
    auxiliar.append(flat_weights)  # Assuming we want the first 1000 parameters for simplicity
    colorList.append((1, 0.1, 0.1, 1.0/(100-i)))
# Convert auxiliar list to numpy array
pesos_history = np.array(auxiliar)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(pesos_history)

# 1. 'pesos_history' es un array de forma (num_epocas, num_parametros)
pca = PCA(n_components=2)
pesos_2d = pca.fit_transform(pesos_history)

# 2. Ahora pesos_2d[:, 0] y pesos_2d[:, 1] son tus coordenadas X e Y

final_point = pesos_2d[-1, :]

def calcular_superficie(X, Y, centro):
    return np.sqrt((X - centro[0])**2 + (Y - centro[1])**2)

# 3. Crear el grid para el fondo
x_grid = np.linspace(pesos_2d[:,0].min()-0.2, pesos_2d[:,0].max()+0.2, 200)
y_grid = np.linspace(pesos_2d[:,1].min()-0.2, pesos_2d[:,1].max()+0.2, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = calcular_superficie(X, Y, final_point)

# 4. CALAVE: Calcular los valores de Z exactamente en los puntos de la trayectoria
# Estos serán nuestros niveles de contorno "obligatorios"
niveles_exactos = [calcular_superficie(p[0], p[1], final_point) for p in pesos_2d]
# Los ordenamos de menor a mayor para que plt.contour no se queje
niveles_exactos = sorted(list(set(niveles_exactos)))

cm = plt.cm.get_cmap('viridis')

plt.contourf(X, Y, Z, levels=niveles_exactos, cmap=cm, alpha=0.3)
plt.contour(X, Y, Z, levels=niveles_exactos, colors='black', linewidths=0.5)
plt.plot(pesos_2d[:, 0], pesos_2d[:, 1], marker=None, linestyle='-') # La trayectoria
#Scatter all the points with the color list
plt.scatter(pesos_2d[:, 0], pesos_2d[:, 1], color=colorList, s=50, edgecolors='blue', linewidth=2) # Los puntos con colores progresivos
plt.show()

"""
coalition_Color_Dictionary = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "yellow",
}


import Config

#This time instead of using the baseline, we use the weights of the lossLandscape experiment
prefixDir = "outputs/" + Config.experiment_name + "/"
hist_list = []

edge_ColorList = []
#For loop for the 10 agents
for agent in range(10):
    auxiliar = []
    auxColorList = []
    for epoch in range(30):
        path = prefixDir + "scp_" + str(agent) + "/iteration_" + str(epoch) + "/weights.pth"
        weights = torch.load(path)
        flat_weights = []
        for key in weights.keys():
            flat_weights.extend(weights[key].flatten().tolist())
        auxiliar.append(flat_weights)
        #We check the coalition of the agent and match it
        with open(prefixDir + "scp_" + str(agent) + "/iteration_" + str(epoch) + "/coalition.txt", "r") as f:
            coalition = int(f.read())
        auxColorList.append(coalition_Color_Dictionary[coalition])
    edge_ColorList.append(auxColorList)
    hist_list.append(auxiliar)

all_weights = np.vstack(hist_list)

pca = PCA(n_components=2)
pca.fit(all_weights)


all_weights_2d = pca.transform(all_weights)

proyecciones = [pca.transform(h) for h in hist_list]
punto_inicial = proyecciones[0][0, :] # Es el mismo para todos
puntos_finales = [p[-1, :] for p in proyecciones]

# 2. Definir el Horizonte Sintético Multimodal
def loss_multimodal(X, Y, finales, inicio):
    # La "pérdida" es la distancia al valle más cercano (mínimo de distancias)
    dist_valles = []
    for f in finales:
        # Valle: aumenta la pérdida según te alejas del final
        dist_valles.append(np.sqrt((X - f[0])**2 + (Y - f[1])**2))
    
    Z = np.min(dist_valles, axis=0)
    
    # Añadimos la "colina" en el inicio (opcional pero visualmente potente)
    # Una gaussiana que eleva el centro
    colina = 2 * np.exp(-((X - inicio[0])**2 + (Y - inicio[1])**2) / 2)
    return Z + colina

# 3. Crear el Grid
x_min, x_max = all_weights_2d[:,0].min()-1, all_weights_2d[:,0].max()+1
y_min, y_max = all_weights_2d[:,1].min()-1, all_weights_2d[:,1].max()+1
X, Y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = loss_multimodal(X, Y, puntos_finales, punto_inicial)

# 4. Los Niveles: para que "corten" los puntos
# Tomamos una muestra representativa de todos los valores de 'Z' en las trayectorias
niveles = np.unique(np.percentile([loss_multimodal(p[:,0], p[:,1], puntos_finales, punto_inicial) 
                                 for p in proyecciones], np.linspace(0, 100, 20)))

# 5. Plot
plt.figure(figsize=(12, 8))
plt.contourf(X, Y, Z, levels=sorted(niveles), cmap='terrain', alpha=0.4)
plt.contour(X, Y, Z, levels=sorted(niveles), colors='black', linewidths=0.3, alpha=0.5)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i, p in enumerate(proyecciones):
    plt.plot(p[:, 0], p[:, 1], color=colors[i], label=f'Modelo {i+1}', alpha=0.8, lw=2)
    
    #Now we scatter the points with the same color but the edge color is taken from the edge_ColorList, which depends on the coalition of the agent in that epoch.
    for j in range(len(p)):
        plt.scatter(p[j, 0], p[j, 1], color=colors[i], edgecolors=edge_ColorList[i][j], s=50, zorder=5) # Los puntos con colores progresivos
    
    plt.scatter(p[-1, 0], p[-1, 1], color=colors[i], edgecolors=edge_ColorList[i][-1], s=100, zorder=5)

plt.scatter(punto_inicial[0], punto_inicial[1], c='white', edgecolors='black', s=200, marker='*', label='Inicio Común')
plt.legend()
plt.show()
"""