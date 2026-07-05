import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def analizar_sensibilidad_capas(model, dataset_loader, delta=10):
    """
    Calcula la relevancia/sensibilidad de cada capa lineal modificando sus pesos por un 'delta'.
    Devuelve las importancias normalizadas para que sumen 1.
    """
    model.eval()  # Importante: Desactivar Dropout para que la comparación sea justa
    
    # 1. Obtener las predicciones base (originales) del dataset
    predicciones_base = []
    inputs_list = []
    
    with torch.no_grad():
        for inputs, _ in dataset_loader:
            outputs = model(inputs)
            predicciones_base.append(outputs)
            inputs_list.append(inputs)
            
    predicciones_base = torch.cat(predicciones_base, dim=0)
    
    # Identificar solo las capas que tienen pesos (nn.Linear)
    # Ignoramos ReLU y Dropout porque no tienen parámetros alterables
    capas_lineales = [modulo for modulo in model.modules() if isinstance(modulo, nn.Linear)]
    cambios_por_capa = []
    
    criterion = nn.MSELoss()
    
    # 2. Perturbar cada capa por separado
    for idx, capa in enumerate(capas_lineales):
        # Guardamos los pesos originales para restaurarlos después
        peso_original = capa.weight.data.clone()
        bias_original = capa.bias.data.clone() if capa.bias is not None else None
        
        # Aplicamos la perturbación delta (puedes optar por sumarle delta estático 
        # o un ruido aleatorio proporcional a delta. Aquí sumamos delta directamente)
        delt = 2 * (np.random.random(1) - 1)[0] *  delta
        capa.weight.data += delt
        print(delt)
        if bias_original is not None:
            capa.bias.data += delta
            
        # Calcular nuevas predicciones con la capa perturbada
        predicciones_perturbadas = []
        with torch.no_grad():
            for inputs in inputs_list:
                outputs = model(inputs)
                predicciones_perturbadas.append(outputs)
        predicciones_perturbadas = torch.cat(predicciones_perturbadas, dim=0)
        
        # Medir el impacto: ¿Cuánto cambiaron las predicciones respecto a las originales?
        impacto = criterion(predicciones_perturbadas, predicciones_base).item()
        cambios_por_capa.append(impacto)
        
        # Restaurar los pesos originales de la capa
        capa.weight.data.copy_(peso_original)
        if bias_original is not None:
            capa.bias.data.copy_(bias_original)
            
    # 3. Normalizar los resultados para que sumen 1
    suma_total = sum(cambios_por_capa)
    
    # Control de excepciones por si la red colapsó a outputs constantes donde el impacto es 0
    if suma_total == 0:
        importancias_normalizadas = [1.0 / len(cambios_por_capa)] * len(cambios_por_capa)
    else:
        importancias_normalizadas = [cambio / suma_total for cambio in cambios_por_capa]
        
    return importancias_normalizadas



def layer_fisher_importances(model, dataloader, device='cpu', max_batches=None, agg='sum', eps=1e-12):
    # ¡CORRECCIÓN 3! Usar eval() para congelar Dropout, pero permitiendo gradientes
    model.to(device).eval()
    
    param_to_layer = {}
    for module_name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            param_to_layer[p] = module_name or "(root)"
            
    accum = {p: torch.zeros_like(p.data, device=device) for p in param_to_layer}
    n = 0
    
    # Habilitamos gradientes explícitamente ya que estamos en modo eval
    with torch.enable_grad():
        for i, (x, _) in enumerate(dataloader): # Ignoramos 'y' real para la Fisher verdadera
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            
            model.zero_grad()
            logits = model(x)
            
            # ¡CORRECCIÓN 1 y 2! 
            # Calculamos la log-probabilidad y usamos la distribución del modelo.
            # Además, calculamos la pérdida por muestra (no el promedio del lote).
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1).detach()
            
            # La Fisher para CrossEntropy usando las predicciones del modelo como "targets"
            # ponderadas por su propia probabilidad:
            loss = -(probs * log_probs).sum() 
            
            loss.backward()
            
            for p in accum.keys():
                if p.grad is not None:
                    # Acumulamos el cuadrado del gradiente directo (ya no está sesgado por el tamaño del lote)
                    accum[p] += p.grad.detach().pow(2)
                    
            n += x.size(0)
            
    if n == 0:
        raise RuntimeError("No data seen.")
        
    # El resto del tipado y agregación por capas está estructuralmente bien
    layer_vals = {}
    for p, layer in param_to_layer.items():
        val = (accum[p] / float(n)).clamp(min=eps)
        if agg == 'sum':
            s = val.sum().item()
        elif agg == 'mean':
            s = val.mean().item()
        else:
            raise ValueError("agg must be 'sum' or 'mean'")
        layer_vals[layer] = layer_vals.get(layer, 0.0) + s
        
    names = list(layer_vals.keys())
    vals = [layer_vals[nm] for nm in names]
    total = sum(vals)
    
    if total == 0:
        norms = [1.0/len(vals)]*len(vals)
    else:
        norms = [v/total for v in vals]
        
    return tuple(names), tuple(norms)




# ==========================================
# EJEMPLO DE USO (Simulación)
# ==========================================
if __name__ == "__main__":
    # Definir tu arquitectura exacta
    in_dim = 20
    num_classes = 3
    
    mi_red = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(64, num_classes)
    )
    
    # Crear un dataset sintético para el ejemplo (X, y)
    X_falso = torch.randn(100, in_dim)
    y_falso = torch.randint(0, num_classes, (100,))
    dataset_falso = torch.utils.data.TensorDataset(X_falso, y_falso)
    loader_falso = torch.utils.data.DataLoader(dataset_falso, batch_size=32, shuffle=True)

    names, relevancia = layer_fisher_importances(mi_red, loader_falso, device='cpu', agg='sum', eps=1e-2)
    print("Layers:", names)
    print("Importances (normalized):", relevancia)


    # Calcular relevancias
    relevancia = analizar_sensibilidad_capas(mi_red, loader_falso, delta=1e-2)
    
    print("Relevancia por capa (Capa 1, Capa 2):")
    print(tuple(relevancia))