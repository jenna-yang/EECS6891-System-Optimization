import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data_path = './machine.data'
columns = ['Vendor', 'Model', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
df = pd.read_csv(data_path, names=columns, header=None)

# Drop non-predictive columns
df.drop(columns=['Model', 'ERP'], inplace=True)

# Preprocessing
categorical_features = ['Vendor']
numeric_features = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']

ohe = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', scaler, numeric_features),
    ('cat', ohe, categorical_features)
])

X = df.drop(columns=['PRP'])
y = df['PRP']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

'''
# layers
layer_count = [1, 2, 3]
layer_histories = []
layer_prp = []

for layer in layer_count:
    model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(X_train.shape[1],))])
                    
    for _ in range(layer):
        model.add(tf.keras.layers.Dense(64, activation='relu'))
            
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                       loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=1)
    layer_histories.append((f'{layer} layers', history))  

    # Evaluate model
    y_pred = model.predict(X_test).flatten()
    layer_prp.append((y_test, y_pred))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("layers", layer)
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')


# units per layer
feature_sizes = [64, 128, 256]
feature_histories = []
feature_prp = []

for units in feature_sizes:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(units, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=1)
    feature_histories.append((f'{units} units', history))

    # Evaluate model
    y_pred = model.predict(X_test).flatten()
    feature_prp.append((y_test, y_pred))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("units", units)
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')
'''

learning_rates = [0.1, 0.01, 0.001, 0.0005]
lr_histories = []
lr_prp = []

for lr in learning_rates:
    model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(1)  # Regression output
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, verbose=1)
    lr_histories.append((f'{lr} rates', history))

    # Evaluate model
    y_pred = model.predict(X_test).flatten()
    lr_prp.append((y_test, y_pred))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')


def plot_results(histories, prps):
    colors = plt.cm.tab10.colors

    plt.subplot(1, 3, 1)
    for i, (name, history) in enumerate(histories):
        plt.plot(history.history['mae'], color=colors[i], linestyle='-', linewidth=5, alpha=0.7, label=name)
    
    plt.title('Training MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.subplot(1, 3, 2)
    for i, (name, history) in enumerate(histories):
        plt.plot(history.history['val_mae'], color=colors[i], linestyle='-', linewidth=5, alpha=0.7, label=name)
    plt.title('Validation MAE')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')


    # Plot actual vs predicted
    data_list = []
    for i, (y_test, y_pred) in enumerate(prps):
        df_temp = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        #df_temp['Feature_Size'] = feature_sizes[i]
        #df_temp['Layers'] = layer_count[i]
        df_temp["LRs"] = learning_rates[i]
        data_list.append(df_temp)

    prps = pd.concat(data_list, ignore_index=True)
    plt.subplot(1, 3, 3)
    #sns.scatterplot(x=prps['Actual'], y=prps['Predicted'], hue=prps['Feature_Size'], palette="coolwarm")
    #sns.scatterplot(x=prps['Actual'], y=prps['Predicted'], hue=prps['Layers'], palette="coolwarm")
    sns.scatterplot(x=prps['Actual'], y=prps['Predicted'], hue=prps['LRs'], palette="coolwarm")

    plt.xlabel("Actual PRP")
    plt.ylabel("Predicted PRP")
    plt.title("Actual vs Predicted Relative CPU Performance")
    plt.legend(loc='lower right')
    plt.show()

plot_results(lr_histories, lr_prp)
#plot_results(layer_histories, layer_prp)
#plot_results(feature_histories, feature_prp)
