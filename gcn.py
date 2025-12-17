import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import glob
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import radians, sin, cos, asin, sqrt
import glob
import os

# coordends de estaciones
stations = {
    "GYR": (4.78375, -74.04414),   # Guaymaral - Suba
    "USQ": (4.71035, -74.03042),   # Usaquén
    "SUB": (4.76125, -74.09346),   # Suba
    "BOL": (4.73580, -74.12589),   # Bolívia - Engativá	
    "LFR": (4.69070, -74.08248),   # Las Ferias - Engativá
    "CDAR": (4.65847, -74.08400),  # Centro de alto rendimiento - Engativá
    "MAM": (4.62549, -74.06698),   # Min ambiente - santa fe
    "FTB": (4.67824, -74.14382),   # Fontibón
    "PTE": (4.63177, -74.11748),   # Puente Aranda
    "KEN": (4.62505, -74.16133),   # Kennedy
    "CSE": (4.59583, -74.14850),   # Carvajal-Sevillana - Kennedy
    "TUN": (4.57623, -74.13101),   # Tunal - Tunjuelito
    "SCR": (4.57255, -74.08381),   # San Cristóbal
    "JAZ": (4.60850, -74.11494),   # Jazmín - 	Puente Aranda
    "USM": (4.53206, -74.11714),   # Usme
    "MOV7": (4.645194, -74.061556),# Móvil 7 - chapinero
    "CBV": (4.59447, -74.16627),   # Ciudad Bolívar  
    "COL": (4.73719, -74.06947),   # Colina - Suba
    "MOV2": (4.66799, -74.14850)   # Móvil Fontibón
}
# extrar las llaves de las estaciones
station_list = list(stations.keys())

# carpet donde estbn ls estaciones
path = r"C:\Users\aleja\Desktop\3 MESES ESTACIONES"
# encuentra todos los archivos de excel de la crpet y crea un archivo temporal
files = [f for f in glob.glob(path + "/*.xlsx") if not os.path.basename(f).startswith("~$")]
df_all = None # inicalizamos varible para guardart el dtaframe

# pasa por los archivos de excel uno por uno
for file in files:
    df = pd.read_excel(file) # lee ela rchivode escel 
    df.rename(columns={"DateTime": "datetime", "PM2.5": "pm25"}, inplace=True) # # renombra ls columns por s vienen modificds
    df["datetime"] = pd.to_datetime(df["datetime"])  # datatime a formato de fecha python
    station_name = file.split("\\")[-1].split(".")[0] # obtiene el nombre de la estacion desde el archiuvo omitendo la ruta
    df.rename(columns={"pm25": station_name}, inplace=True) # renombra la columna pm 2.5 con el nombre de la estcion
    if df_all is None:
        df_all = df[["datetime", station_name]] # agregamos al dataset las columnas de la estacion
    else:
        df_all = df_all.merge(df[["datetime", station_name]], on="datetime", how="outer") # si ya existe grega l;as nuevas estaciones l dataset


df_all.set_index("datetime", inplace=True) # coinvierte datatime a indice de dataframe
df_all = df_all.sort_index()


# Filtrar estaciones disponibles
station_list = [st for st in station_list if st in df_all.columns]
N = len(station_list) # numero de estaciones
# imprimir informacion
print(f"Estaciones disponibles en datos: {station_list}") 
print(f"Total estaciones: {N}")
print(f"Timesteps: {len(df_all)}") # cunts filas por serie temporal hy

# creamos el grafo
# obtenemos la distanc real en la tierr entre 2 estciones
def haversine(lat1, lon1, lat2, lon2): # distancia en kilometros 
    R = 6371  # radio de la tierra en km
    # convertimos la diferencias de grados a radines 
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    # formula haversine
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2 # calcula el angulo central entre 2 puntos en la tierra
    return 2 * R * asin(sqrt(a))# distancia real entre estaciones

# Distancia entre estaciones
dist_matrix = np.zeros((N, N)) # matriz tamaño N X N el numero de estaciones disponibles
for i in range(N):
    for j in range(N):
        # recorremos la matriz  con las posibles combinaciones
        lat1, lon1 = stations[station_list[i]] # obtenemos ls coordenadas de cada estacion
        lat2, lon2 = stations[station_list[j]]# obtenemos ls coordenadas de cada estacion
        dist_matrix[i, j] = haversine(lat1, lon1, lat2, lon2) # calculamos la distancias con la funcion haversine 

# Normalización por nodo
scalers = {} # cremos un diccionario vacio
data_scaled = np.zeros((len(df_all), N)) # matriz con los datos  timesteps x estaciones
for i, st in enumerate(station_list):  # recorremos las estaciones y obtenemos un numero por cada estacion
    scaler = MinMaxScaler() # creamos el normalizador
    data_scaled[:, i] = scaler.fit_transform(df_all[[st]]).flatten() # tomamops los datos por cada estacion y los normalizamosd
    scalers[st] = scaler # guardamos los valores nomrmalizados para cada estcion

# Correlación entre estciones
corr_matrix = np.corrcoef(data_scaled.T)

K = min(6, N-1) # conexiones entre cada estaciones
alpha = 0.5 # peso de limportancia de la distancia y relcion
edges = [] # list vcia para las conexiones entre estaciones
edge_weights = [] # pesos de esas conexiones

for i in range(N): # recorremos cad estacion
    neighbors = np.argsort(dist_matrix[i, :])[1:K+1] # hgarram,oss las estaciones ms cercans por distancia 
    for j in neighbors: # recorremos las estciones ms cercanas
        edges.append([i, j]) # agreasmos un arista dirigiada a esa estancion
        w_dist = np.exp(-dist_matrix[i,j]/5.0)  # calculamos el peso entre estciones  1 si es poco y 0 si es mucho
        w_corr = (corr_matrix[i,j]+1)/2 # convierte la correlacion entre un valor de 0 y 1
        edge_weights.append(alpha * w_dist + (1-alpha) * w_corr)  # calcula el peso final de la arista 

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # convertimos las lists con las conwexiones y la convertimos en un tesnor
edge_weight = torch.tensor(edge_weights, dtype=torch.float32) # pesos para cada conexion 

print(f"Grafo  creado: {N} nodos, {len(edges)} aristas")



WINDOW = 24 # horas psadas
X_sequences = [] # ventan de entrad 
Y_sequences = [] # venbtn de salida

for t in range(WINDOW, len(data_scaled)): # recorremos los datops 
    x_window = data_scaled[t-WINDOW:t].T  # 24 hors previas
    X_node_features = [x_window[node_idx] for node_idx in range(N)] # toammos las 24 hors de cada nodo
    X_sequences.append(X_node_features) # guardamos el conjunto de ventanas de cada nodo
    Y_sequences.append(data_scaled[t]) # guardmos el valor  predecr

X_sequences = np.array(X_sequences)  # convertimos a matriz las hros previs, numero de ventanas, numero de estaciones, ventn
Y_sequences = np.array(Y_sequences)  # valor a predecir 

print(f"Secuencias: {len(X_sequences)} samples, ventana {WINDOW}h")


# usamos el 80% de los datos para el entrenamiento
split = int(len(X_sequences) * 0.8)
X_train, X_test = X_sequences[:split], X_sequences[split:] # datos de entrenmiento 
Y_train, Y_test = Y_sequences[:split], Y_sequences[split:] # datos de prueba



class GCN(nn.Module): # clse pr el modelo
    def __init__(self, input_dim, hidden_dims=[48,32,24,16]): # carcterisctics t numero de capas internas 
        super().__init__() # consutructor
        self.temporal_encoder = nn.Sequential( # procemos ls secuencias de datos antes de psarls l gcn
            nn.Linear(input_dim, hidden_dims[0]), # capa lineal, cada vector de entrada a las neurnos asignadas
            nn.ReLU(), # activamos no linelidad paraa las relaaciones complejs
            nn.Dropout(0.2) #apagamos aletorimente 
        )
        self.gcn_layers = nn.ModuleList() # creamos listas pr almacenar ls capas gcn
        self.residual_layers = nn.ModuleList() # caps para el atajo residual
        self.dropout_layers = nn.ModuleList() # drop out de cada capa
        for i in range(len(hidden_dims)-1): # iteramos por las capas
            # creamos un capa gcn y la guardamos par el modelo, aprenndemos los datos de l propia estacion y de ls que estan cerca
            self.gcn_layers.append(GCNConv(hidden_dims[i], hidden_dims[i+1], normalize=True)) # creamos la capa 
          # creamos una cap residual para evitr perdida de informacion
            self.residual_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]) # tmño de entrd y tamaño de salida
                if hidden_dims[i]!=hidden_dims[i+1] # soi son diferentes l entrd y slida l cconvierte usndo lines
                else nn.Identity() # sino us identy psndo los dros sun cmbirlo  
            )
            self.dropout_layers.append(nn.Dropout(0.2)) # regmos  cad cp drop out par pagar letoriamente 20% neurons
        #capa densa
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]), # capa densa full conectad
            nn.ReLU(), # ctivmos no lienelidd
            nn.Dropout(0.2), # apaga el 20% de neuronas
            nn.Linear(hidden_dims[-1], 1) # pconvertismos la entrada  1 valor por nodo
        )
    def forward(self, x, edge_index, edge_weight): # tensor que recibe numero de muestras en el lote, nuemerod de nodos y ventan conexiones entre el grafo pesos/distncis de nodos
        batch_size, num_nodes, window = x.shape # extraemos la informacion
        batch_predictions = [] # lista para laa prediccion de cada elemneto 
        for b in range(batch_size): # procesmos cda elemento del batch
            x_graph = x[b] # gurdamos los datos historicos de cda nodo estcion y venna
            h = self.temporal_encoder(x_graph) # vector de las ultims horas de cad  nodo
            # bucle sobre las caps gcn juntanto ls listas
            for gcn, res, drop in zip(self.gcn_layers, self.residual_layers, self.dropout_layers):
                identity = res(h) # aplcimos la cap residual  h y gurdamos el resultdo
                h = gcn(h, edge_index, edge_weight) # fetures de cd nodo por un capa gcn 
                # algo asi como trsnformar l infromacion sin olvidar como era ntes
                h = F.relu(h + identity) # para modelo no lieneal y suma cn con la version residual

                h = drop(h) # apagAmos un porcentaje de neurons
            pred = self.decoder(h).squeeze(-1) # reliz l prediccion final
            batch_predictions.append(pred) # gurd la prediccion del batch en l lista
        return torch.stack(batch_predictions, dim=0)

model = GCN(input_dim=WINDOW, hidden_dims=[48,32,24,16]) # llmmos a la clase pr crear el modelo ye nviamos ls neurons por capa
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5) # actulizamos los pesos del modelo y evitmops sobreajuste

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts( # ajust el lernng rate automticmanete
    optimizer,
    T_0=20,                # primer ciclo de entrenamiento dura 20 epochs
    T_mult=2,              # cada ciclo siguiente dura el doble del anterior
    eta_min=1e-6           # valor mínimo al que puede bajar el learning rate
)

# contamos los datos por estcion 
station_counts = np.sum(~np.isnan(data_scaled), axis=0)
weights = 1/(station_counts + 1e-6) # asigamos persos altos pr estciones con pocos datos
weights = weights / weights.sum() # summos los pesos pr que los peson sumen 1
weights_t = torch.tensor(weights, dtype=torch.float32) # convertimos el rray en un tesnor

def weighted_loss(pred, target):
    mse = (pred - target)**2 # clculmos el error cudrticopor cad estcion 
    return (mse * weights_t).sum() / weights_t.sum() # optenemos el vlor de l perdida ponderado 

loss_fn = weighted_loss # guardamos la funcion 


EPOCHS = 200 # numero de epocas
BATCH_SIZE = 16 # numero demuestras pr cada entrenmiento
# conviertimos los datos de entrenamientos a tensores 
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
# conviertimos los datos de test a tensores 
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

best_r2 = -float('inf')  # mejor r2 en infinito negativo
patience_counter = 0

for epoch in range(EPOCHS):# bucle por epocas
    model.train() # mctivamos el modelo de entrenamiento 
    epoch_loss = 0 # varible para la perdida de epochs
    for i in range(0, len(X_train_t), BATCH_SIZE): #
        x_batch = X_train_t[i:i+BATCH_SIZE] #  conjunto de datos de entrda
        y_batch = Y_train_t[i:i+BATCH_SIZE] # conjunto de datos de salida
        pred = model(x_batch, edge_index, edge_weight)  # clculmos las predicciones con el numero de nodos, conexiones y pesos de rists
        loss = loss_fn(pred, y_batch) # clculmos l operdid entre l prediccion y el vlor rel
        optimizer.zero_grad() # limpimos gradiantes 
        loss.backward() #calculamos los gradintes respecto a la perdid par justr los pesos
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # evitamos que las gradiantes se vuelvan muy grande  sobre justen el modelo
        optimizer.step() # ajustamos los pesos del modelo aprtir de los gradiantes 
        scheduler.step() # actualiza el learng rate 
        epoch_loss += loss.item() # comulmos l perdid de todos los batch en  esa epoca
    avg_loss = epoch_loss / (len(X_train_t)/BATCH_SIZE) #clculmos l perdid promedio del epoch
    if (epoch+1)%10==0:
        model.eval() # evlumos el modelo
        with torch.no_grad(): # evitmos gurdr rdientes pr l evlucion 
            val_pred = model(X_test_t, edge_index, edge_weight) # mndmos los vlores  predecir
            val_loss = loss_fn(val_pred, Y_test_t).item() # calculamos la perdida entre 
            val_r2 = r2_score(Y_test_t.numpy().flatten(), val_pred.numpy().flatten()) # coeficente de relctividd
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {avg_loss:.6f} | Val: {val_loss:.6f} | Val R²: {val_r2:.4f}")
       # mirmos si el r2 es mejor
        if val_r2 > best_r2:
            best_r2 = val_r2 # lo urdmos s es mejor
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gcn_adaptativo.pth') # guardamos los parametros del mejor modelo
        else:
            # si no meor summos
            patience_counter += 1
            if patience_counter >= 20: # mayor a 20 lo detenemos 
                break



model.load_state_dict(torch.load('best_gcn_adaptativo.pth')) # cargamos el modelo guardado
model.eval() # activamos el modo evaluacion
with torch.no_grad(): # evitmos gurdr rdientes pr l evlucion 
    Y_pred = model(X_test_t, edge_index, edge_weight).numpy() # mandamos los vlores  predecir
    Y_true = Y_test # datos verdaderos

Y_pred_real = np.zeros_like(Y_pred) # matriz con las predicciones del modelo
Y_true_real = np.zeros_like(Y_true) # matriz con el vlor rel
for i, st in enumerate(station_list): # iteramos por cada estacion 
    # desnormalizmos
    Y_pred_real[:, i] = scalers[st].inverse_transform(Y_pred[:, [i]]).flatten() #seleccion tods las fils y una column pra tomas los vlores predichos
    Y_true_real[:, i] = scalers[st].inverse_transform(Y_true[:, [i]]).flatten()#seleccion tods las fils y una column pra tomas los vlores relaes

mae_global = mean_absolute_error(Y_true_real.flatten(), Y_pred_real.flatten()) # calculamos el error absoluto promedio de todosas las estaciones
mse_global = mean_squared_error(Y_true_real.flatten(), Y_pred_real.flatten()) # calculamos el error absoluto promedio de todosas las estacione
r2_global = r2_score(Y_true_real.flatten(), Y_pred_real.flatten()) # coeficiente de determinación de todosas las estacione

print(" RESULTADOS GLOBALES")
print(f"MAE:  {mae_global:.2f} µg/m³")
print(f"MSE: {mse_global:.2f} µg/m³")
print(f"R²:   {r2_global:.4f}")

# Resultados por estación
print(" RESULTADOS POR ESTACIÓN")
print(f"{'Estación':<10} {'MAE':>8} {'MSE':>8} {'R²':>8}")
for i, st in enumerate(station_list):
    mae = mean_absolute_error(Y_true_real[:, i], Y_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(Y_true_real[:, i], Y_pred_real[:, i]))
    r2 = r2_score(Y_true_real[:, i], Y_pred_real[:, i])
    print(f"{st:<10} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f}")


# hor aa predecir
target_datetime = pd.to_datetime("2025-07-16 08:00:00")

# Verificar que la fecha esté en el dataset y en el conjunto de test
if target_datetime in df_all.index[WINDOW:]:
    idx_full = df_all.index.get_loc(target_datetime) - WINDOW # encontramos el indice de la hora y restmos la ventana
    if idx_full >= split:  # lka fech esta en el conjunto del test
        test_idx = idx_full - split # encontramos la fecha a revsar del conjunto del etest
        real_vals = Y_true_real[test_idx]  # toma el valor de esa posicion
        pred_vals = Y_pred_real[test_idx] # toma el valor de esa posicion 
        
        print(f"\nPM2.5 real vs predicho - {target_datetime}")
        print(f"{'Estación':<10} {'Real':>8} {'Predicho':>10} {'Error':>8}")
        for st, real, pred in zip(station_list, real_vals, pred_vals):
            print(f"{st:<10} {real:8.2f} {pred:10.2f} {abs(real-pred):8.2f}")

    else:
        print("La fecha/hora está en el conjunto de entrenamiento, no en test.")
else:
    print("La fecha/hora no está en el dataset.")
