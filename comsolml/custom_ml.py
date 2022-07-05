import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras import layers
from tensorflow import keras
import keras.layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def make_autoencoder(input_dim, encoding_dim, activation):
    num_layers = int(np.floor(np.log(input_dim/encoding_dim)/np.log(2)))-1
    layer_list = list(range(num_layers))
    input_trace = keras.Input(shape=(input_dim,))
    encoded = Dense(encoding_dim*2**layer_list[-1], activation=activation)(input_trace)
    for i in reversed(layer_list[:-1]):
        encoded = Dense(encoding_dim*2**i, activation=activation)(encoded)
    decoded = Dense(encoding_dim*2, activation=activation)(encoded)
    for j in (layer_list[2:]):
        decoded = Dense(encoding_dim*2**j, activation=activation)(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = keras.Model(input_trace, decoded)
    encoder = keras.Model(input_trace, encoded)

    return autoencoder, encoder

def make_regression_model(input_dim, activation):
    add_another = True
    if input_dim > 64:
        next_dim = 64
    else:
        next_dim = int(np.floor(input_dim/2))
    model = Sequential()
    if activation == 'LeakyReLU':
        LeakyReLU = keras.layers.LeakyReLU
        #model.add(Dense(input_dim))
        #model.add(LeakyReLU())
        while add_another:
            model.add(Dense(next_dim))
            model.add(LeakyReLU())
            next_dim = int(np.floor(next_dim/2))
            print(f"Layer added; size: {next_dim}")
            if next_dim <= 2:
                add_another = False
        model.add(Dense(1, activation='linear'))

    else:
        #model.add(Dense(input_dim, activation=activation))
        while add_another:
            model.add(Dense(next_dim, activation=activation))
            next_dim = int(np.floor(next_dim/2))
            print(f"Layer added; size: {next_dim}")
            if next_dim <= 2:
                add_another = False
        model.add(Dense(1, activation='linear'))
    model.build(input_shape=(None, input_dim))
    return model

def do_regression(X_sample, label_sample, attributes):
    sns.set_style('whitegrid')
    df_loss = pd.DataFrame(columns=attributes)
    df_test = pd.DataFrame(columns=attributes)
    df_predict = pd.DataFrame(columns=attributes)
    regression_models = {}
    min_max_scalers = {}
    for attribute in attributes:
        print(attribute)
        y_reg = label_sample[[attribute]]

        min_max_scaler = MinMaxScaler()
        y_reg = min_max_scaler.fit_transform(y_reg)
        x_reg_train, x_reg_test, y_reg_train, y_reg_test_norm = train_test_split(X_sample, y_reg, test_size=0.2,
                                                                                 random_state=42)
        x_reg_train_flat = x_reg_train.reshape((len(x_reg_train), np.prod(x_reg_train.shape[1:])))
        x_reg_test_flat = x_reg_test.reshape((len(x_reg_test), np.prod(x_reg_test.shape[1:])))
        regression_model = make_regression_model(x_reg_train_flat.shape[1], 'tanh')
        # direct_regression_model = make_direct_regressor()
        regression_model.compile(loss='mae', optimizer='adam')
        history = regression_model.fit(x_reg_train_flat, y_reg_train.astype('float32'),
                                              epochs=50,
                                              batch_size=20,
                                              shuffle=True,
                                              verbose=0,
                                              validation_data=(x_reg_test_flat, y_reg_test_norm.astype('float32')))

        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Loss', 'Validation loss'])
        plt.title(attribute)
        plt.show()

        y_predict_norm = regression_model.predict(x_reg_test_flat)
        y_predict = min_max_scaler.inverse_transform(y_predict_norm)
        y_reg_test = min_max_scaler.inverse_transform(y_reg_test_norm)
        loss = regression_model.evaluate(x_reg_test_flat, y_reg_test_norm)
        print("loss: " + str(loss))
        df_loss.at[0, attribute] = loss
        df_test[attribute] = np.squeeze(y_reg_test)
        df_predict[attribute] = np.squeeze(y_predict)
        regression_models[attribute] = regression_model
        min_max_scalers[attribute] = min_max_scaler

    return df_loss, df_test, df_predict, regression_models, min_max_scalers

def plot_reg_results(df_test, df_predict, df_loss, df_scales, perc=True, hue=None, title=None):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    attributes = df_test.columns
    plt.figure(figsize=(20, 12))
    for i, attribute in enumerate(attributes):
        y_test_merge = np.squeeze(np.stack([df_test[[attribute]].to_numpy(), df_predict[[attribute]].to_numpy()]))
        df_test_merge = pd.DataFrame(y_test_merge.T, columns=['True [mm]', 'Predicted [mm]'])
        df_test_merge['abs_diff'] = abs(df_test_merge['True [mm]'] - df_test_merge['Predicted [mm]'])
        df_test_merge.quantile(q=0.75)['abs_diff']
        df_test_merge['75th percentile'] = np.where(
            df_test_merge['abs_diff'] < df_test_merge.quantile(q=0.75)['abs_diff'], True, False)
        plt.subplot(2, 3, i + 1)
        if perc:
            sns.scatterplot(data=df_test_merge, x='True [mm]', y='Predicted [mm]', hue='75th percentile', alpha=0.5)
            plt.legend(loc='upper left', title='75th percentile')
        elif hue:
            sns.scatterplot(data=df_test_merge, x='True [mm]', y='Predicted [mm]', hue=hue, alpha=0.5, cmap="jet")
            plt.legend([], [], frameon=False)
        else:
            sns.scatterplot(data=df_test_merge, x='True [mm]', y='Predicted [mm]', alpha=0.5)
        textposx = 1 * (df_scales.at[0, attribute] + (df_scales.at[1, attribute] - df_scales.at[0, attribute]) * 0.5)
        textposy = 1 * (df_scales.at[0, attribute] + (df_scales.at[1, attribute] - df_scales.at[0, attribute]) * 0.05)
        textstring = "Normalized loss (mae): {:.6f}".format(df_loss.at[0, attribute])
        plt.text(textposx, textposy, textstring, fontsize=11, bbox=dict(facecolor='grey', alpha=0.5))
        plt.plot([df_test_merge['True [mm]'].min(), df_test_merge['True [mm]'].max()],
                 [df_test_merge['True [mm]'].min(), df_test_merge['True [mm]'].max()], 'r--')
        plt.title(attribute)
        plt.xlim(df_scales.at[0, attribute], df_scales.at[1, attribute])
        plt.ylim(df_scales.at[0, attribute], df_scales.at[1, attribute])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if title:
        plt.suptitle(title, fontsize=12)

    plt.show()