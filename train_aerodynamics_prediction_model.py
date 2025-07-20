"""
Tak Yeung, tyeung@umd.edu
VECTORIZED IMPLEMENTATION FOR AIRFOIL POLAR PREDICTION
"""
import os
import pickle as pk
import random
import time as t
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch

import fit_spline as fs
from plotting_functions import *


# Define your model with configurable number of hidden layers and activation functions
class MLP_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, activation_function_name):
        super(MLP_model, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        # Add hidden layers
        for _ in range(n_layers):
            self.hidden_layers.append(torch.nn.Linear(input_size if len(self.hidden_layers) == 0 else hidden_size, hidden_size))
            self.dropouts.append(torch.nn.Dropout(p=0.0))  # Improving neural networks by preventing co-adaptation of feature detectors

        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        # Set activation function
        self.activation = self._get_activation_function(activation_function_name)

    def _get_activation_function(self, name):
        activations = {
            'relu': torch.nn.ReLU(),
            'sigmoid': torch.nn.Sigmoid(),
            'tanh': torch.nn.Tanh(),
            'leaky_relu': torch.nn.LeakyReLU()
        }
        return activations[name]

    def forward(self, x):
        for layer, dropout in zip(self.hidden_layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)

        x = self.output_layer(x)
        return x


def main():
    # ======================= Training Parameters =======================
    # data contain 25 unique aoa values, spanning from aoa1 to aoa25. This control how many aoa the training data see.
    mach = 0.6
    mod_folder = f'Model/APM'

    if mach == 0.6:
        aoa_sweep = range(1, 32)
        re = 4.000e6
    elif mach == 0.7:
        aoa_sweep = range(1, 26)
        re = 4.667e6
    else:
        raise ValueError('INCORRECT MACH INPUT')

    # model directory
    dataset = f'Datasets/M{mach}-Re{re:.2e}-clcdcm'.replace('.', '_') + '.xlsx'

    # set up model directory and files' path
    mod_fname, xsc_fname, ysc_fname, pca_fname = setup_model_files(mod_folder)

    # ======================= Generating Model =======================
    # Start timing
    st = t.time()

    # get dataframe based on provided dataset
    df = get_df(dataset, aoa_sweep)

    # read the pdata into a dataframe
    input_size = len(df['X'][0])
    hidden_size = 80  # hidden size
    n_layers = 3  # number of hidden layers
    output_size = 25  # output size (n modes for output vector)
    activation_func = 'relu'  # activation function

    # Train the model
    mod, xsc, ysc, pca, dataframes = train_model(df, input_size, hidden_size, n_layers, output_size, activation_func,
                                                 visualize_training=True, mod_folder=mod_folder)

    # Save model and scaler
    save_files(input_size, hidden_size, n_layers, output_size, activation_func, mod, mod_fname,
               xsc, xsc_fname, ysc, ysc_fname, pca, pca_fname, dataframes, mod_folder)

    print(f'runtime: {t.time() - st}sec!\n')

    # ==================== Result Processing ====================
    # mod_folder = 'Model/aero_mod_f_m06'
    mod, xsc, ysc, pca = get_model(mod_folder)

    dataframes = get_train_valid_test_set(mod_folder)
    find_error(dataframes, mod, xsc, ysc, pca, True)
    plt.show()

    # dataset = 'Datasets/M0_7-Re4_67e+06-test.xlsx'
    dataframes = get_df(dataset, aoa_sweep)
    # find_airfoil_error(dataframes, mod, xsc, ysc, pca, True)

    run_case(dataframes[dataframes['airfoil name'] == 'RC410_0001'], mod, xsc, ysc, pca, mod_folder, True)
    run_case(dataframes[dataframes['airfoil name'] == 'NACA0012_0001'], mod, xsc, ysc, pca, mod_folder, True)
    run_case(dataframes[dataframes['airfoil name'] == 'NACA64A015_0001'], mod, xsc, ysc, pca, mod_folder, True)
    # run_case(dataframes[dataframes['airfoil name'] == 'NACA0009'], mod, xsc, ysc, pca, mod_folder, True)
    # run_case(dataframes[dataframes['airfoil name'] == 'OA212'], mod, xsc, ysc, pca, mod_folder, True)
    # run_case(dataframes[dataframes['airfoil name'] == 'NACA63215'], mod, xsc, ysc, pca, mod_folder, True)

    # run_unknown('Raw_Data/NACA0009.txt', mod, xsc, ysc, pca)
    plt.show()

    # endlessly test airfoil cases from the validation test set
    [_1, _2, dataframes] = get_train_valid_test_set(mod_folder)
    while True:
        run_df = dataframes.sample(n=1)
        print(run_df['airfoil name'])
        run_case(run_df, mod, xsc, ysc, pca, mod_folder, True)
        plt.show()
    return


def setup_model_files(mod_folder_path):
    if not os.path.isdir(f'{mod_folder_path}'):
        os.mkdir(f'{mod_folder_path}')
    mod_fname = f'{mod_folder_path}/mod.pth'
    xsc_fname = f'{mod_folder_path}/xsc.save'
    ysc_fname = f'{mod_folder_path}/ysc.pkl'
    pca_fname = f'{mod_folder_path}/pca.pkl'
    return mod_fname, xsc_fname, ysc_fname, pca_fname


def get_df(file_paths, aoa_sweep=range(1, 26)):
    # initialize an empty dataframe
    unsorted_df = pd.DataFrame()

    # read data file
    if isinstance(file_paths, str):
        _df = pd.read_excel(file_paths)
        unsorted_df = pd.concat([unsorted_df, _df], axis=0)

    elif isinstance(file_paths, list):
        for path in file_paths:
            _df = pd.read_excel(path)
            unsorted_df = pd.concat([unsorted_df, _df], axis=0)

    else:
        raise TypeError('INVALID INPUT')

    # extract features, labels, and availability factor
    X, y, f = [], [], []
    cl, cd, cm = [], [], []

    CST_coef = [f'a{j}' for j in range(1, 20)]
    for i in range(len(unsorted_df)):
        row = unsorted_df.iloc[i]

        # airfoil parameters as feature
        X.append(np.array(row[CST_coef]))

        # airfoil performance as labels (before vectorization and POD)
        cl_curve = row[[f'cl{j}' for j in aoa_sweep]].to_numpy()
        cd_curve = row[[f'cd{j}' for j in aoa_sweep]].to_numpy()
        cd_curve = np.log10(cd_curve.astype(float))
        cm_curve = row[[f'cm{j}' for j in aoa_sweep]].to_numpy()

        line = np.hstack((cl_curve, cd_curve, cm_curve))
        y.append(np.array(line))
        cl.append(cl_curve)
        cd.append(cd_curve)
        cm.append(cm_curve)

        try:
            conv_class = row[[f'class{j}' for j in aoa_sweep]].to_numpy()
            f.append(np.array(conv_class))
        except Exception:
            # no parameters for class
            f.append(np.ones(len(cl_curve)))
            pass

    return pd.DataFrame({'X': X, 'y': y, 'f': f,
                         'airfoil name': unsorted_df['airfoil name'],
                         'cl': cl, 'cd': cd, 'cm': cm})


def train_model(df, input_size, hidden_size, n_layers, output_size, activation_func,
                visualize_training=False, mod_folder=None):
    # ############################################## HYPERPARAMETERS ##############################################
    learning_rate = 0.0005
    beta1 = 0.9
    beta2 = 0.99
    lambda_ = 5e-4  # L2 regularization
    num_epochs = 2000
    tol = 1e-5

    # Split dataset into training, validation, and testing set (60, 20, 20)
    train_df, _df = train_test_split(df, test_size=0.4)
    validation_df, test_df = train_test_split(_df, test_size=0.5)

    # ==================================== Standardize and scale the data sets ====================================
    Scalers = {
        'X': StandardScaler().fit(train_df['X'].to_list()),
        'cl': StandardScaler().fit(train_df['cl'].to_list()),
        'cd': StandardScaler().fit(train_df['cd'].to_list()),
        'cm': StandardScaler().fit(train_df['cm'].to_list())
    }
    xsc = StandardScaler()
    X_train_std = xsc.fit_transform(train_df['X'].to_list())
    X_validation_std = xsc.transform(validation_df['X'].to_list())
    X_test_std = xsc.transform(test_df['X'].to_list())

    # ysc = RobustScaler()
    y_train, y_validation, y_test = train_df['y'].to_list(), validation_df['y'].to_list(), test_df['y'].to_list()
    # y_train_std = ysc.fit_transform(y_train)
    # y_validation_std = ysc.transform(y_validation)
    # y_test_std = ysc.transform(y_test)

    y_train_std = scale_label(train_df, Scalers)
    y_validation_std = scale_label(validation_df, Scalers)
    y_test_std = scale_label(test_df, Scalers)

    # since dataframe only accept 1D list, we must turn a 2D list into a 1D list with numpy array stored inside
    train_df['y_std'] = [np.array(sublist) for sublist in y_train_std]

    # filter out incomplete polar for y_std before POD fitting
    f_polar = np.sum(train_df['f'].to_list(), axis=1)
    train_df['f_polar'] = [int((var == 31)) for var in f_polar]
    filtered_y_std = np.array(train_df.loc[train_df['f_polar'] == 1, 'y_std'].to_list())

    # apply POD to all data based on the complete polars in training set
    pca = PCA(n_components=output_size)
    pca.fit_transform(filtered_y_std)
    y_train_pca = pca.transform(y_train_std)
    y_validation_pca = pca.transform(y_validation_std)
    y_test_pca = pca.transform(y_test_std)

    # ================================== Convert numpy arrays to PyTorch tensors ==================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'USING DEVICE = {device}.')
    
    X_train_std = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_validation_std = torch.tensor(X_validation_std, dtype=torch.float32, device=device)
    X_test_std = torch.tensor(X_test_std, dtype=torch.float32, device=device)

    y_train_pca = torch.tensor(y_train_pca, dtype=torch.float32, device=device)
    y_validation_pca = torch.tensor(y_validation_pca, dtype=torch.float32, device=device)
    y_test_pca = torch.tensor(y_test_pca, dtype=torch.float32, device=device)

    # ===================================== Set-up and train the PyTorch model =====================================
    mod = MLP_model(input_size, hidden_size, n_layers, output_size, activation_func).to(device)

    criterion = torch.nn.HuberLoss()
    # criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(mod.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=lambda_)

    # apply data puncturing technique
    tr_availability_factor = [np.repeat(sub_f, 3) for sub_f in train_df['f'].to_numpy()]
    vd_availability_factor = [np.repeat(sub_f, 3) for sub_f in validation_df['f'].to_numpy()]
    tt_availability_factor = [np.repeat(sub_f, 3) for sub_f in test_df['f'].to_numpy()]

    pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
    
    tr_availability_factor = torch.tensor(np.array(tr_availability_factor, dtype=float), dtype=torch.float32,
                                          device=device)
    vd_availability_factor = torch.tensor(np.array(vd_availability_factor, dtype=float), dtype=torch.float32,
                                          device=device)
    tt_availability_factor = torch.tensor(np.array(tt_availability_factor, dtype=float), dtype=torch.float32,
                                          device=device)

    train_output = torch.matmul(y_train_pca, pca_components) + pca_mean
    train_output = train_output * tr_availability_factor

    validation_output = torch.matmul(y_validation_pca, pca_components) + pca_mean
    validation_output = validation_output * vd_availability_factor

    test_output = torch.matmul(y_test_pca, pca_components) + pca_mean
    test_output = test_output * tt_availability_factor

    # =========== Training Loop ===========
    print(f"Training \n")
    epoch, training_loss_list, validation_loss_list, testing_loss_list = 0, [], [], []
    while epoch < num_epochs:
        epoch += 1
        optimizer.zero_grad()

        outputs = mod(X_train_std)
        predicted_output = torch.matmul(outputs, pca_components) + pca_mean
        predicted_output = predicted_output * tr_availability_factor

        training_loss = criterion(predicted_output, train_output)
        training_loss_list.append(training_loss.item())

        training_loss.backward()
        optimizer.step()

        # validation set
        outputs = mod(X_validation_std)
        predicted_output = torch.matmul(outputs, pca_components) + pca_mean
        predicted_output = predicted_output * vd_availability_factor

        validation_loss = criterion(predicted_output, validation_output)
        validation_loss_list.append(validation_loss.item())

        # testing set
        outputs = mod(X_test_std)
        predicted_output = torch.matmul(outputs, pca_components) + pca_mean
        predicted_output = predicted_output * tt_availability_factor

        testing_loss = criterion(predicted_output, test_output)
        testing_loss_list.append(testing_loss.item())

        if epoch % 10 == 0:
            print(f'epoch: {epoch:4}/{num_epochs}, training loss: {training_loss.item():.4e}, validation loss: {validation_loss.item():.4e}')
        if validation_loss.item() < tol:
            print('Training Complete')
            break

    # =========== Fitting and testing mod using NN ===========
    mod.eval()
    y_train_pred = mod(X_train_std).cpu().detach().numpy()
    y_validation_pred = mod(X_validation_std).cpu().detach().numpy()
    y_test_pred = mod(X_test_std).cpu().detach().numpy()

    y_train_pred_std = pca.inverse_transform(y_train_pred)
    y_validation_pred_std = pca.inverse_transform(y_validation_pred)
    y_test_pred_std = pca.inverse_transform(y_test_pred)

    y_train_pred = descale_label(y_train_pred_std, Scalers)
    y_validation_pred = descale_label(y_validation_pred_std, Scalers)
    y_test_pred = descale_label(y_test_pred_std, Scalers)

    # y_train_pred = ysc.inverse_transform(y_train_pred_std)
    # y_validation_pred = ysc.inverse_transform(y_validation_pred_std)
    # y_test_pred = ysc.inverse_transform(y_test_pred_std)

    # =========== Plot Loss vs Epoch ===========
    if visualize_training:
        plt.figure(42)
        plt.plot(training_loss_list, 'b')
        plt.plot(validation_loss_list, 'r.-')
        plt.plot(testing_loss_list, 'k--')
        plt.legend(['training loss', 'validation loss', 'testing loss'])
        plt.xlabel('Epoch')
        plt.xlim(0, num_epochs)
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(which='both')
        plt.savefig(f'{mod_folder}/Loss_History.png')
        plt.show()

    # =========== Calculate MSE and R^2 score of NN fit ===========
    a = mean_squared_error(y_train, y_train_pred)
    b = mean_squared_error(y_validation, y_validation_pred)
    c = mean_squared_error(y_test, y_test_pred)
    e = r2_score(y_train, y_train_pred)
    f = r2_score(y_validation, y_validation_pred)
    g = r2_score(y_test, y_test_pred)
    print('Baseline (All Features)')
    print('MSE train: %.10f, validation: %.10f, test: %.10f' % (a, b, c))
    print('R^2 train: %.10f, validation: %.10f, test: %.10f' % (e, f, g))
    return mod, xsc, Scalers, pca, [train_df, validation_df, test_df]


def scale_label(df, Scalers):
    cl_std = Scalers['cl'].transform(df['cl'].to_list())
    cd_std = Scalers['cd'].transform(df['cd'].to_list())
    cm_std = Scalers['cm'].transform(df['cm'].to_list())

    y_std = np.append(cl_std, cd_std, axis=1)
    y_std = np.append(y_std, cm_std, axis=1)
    return y_std


def descale_label(y_std, Scalers):
    data_size = int(len(y_std[0]) / 3)  # if y is a list
    cl_std, cd_std, cm_std = [], [], []
    for row in y_std:
        cl_std.append(row[:data_size])
        cd_std.append(row[data_size:2 * data_size])
        cm_std.append(row[2 * data_size:3 * data_size])
        
    cl = Scalers['cl'].inverse_transform(cl_std)
    cd = Scalers['cd'].inverse_transform(cd_std)
    cm = Scalers['cm'].inverse_transform(cm_std)

    y = np.append(cl, cd, axis=1)
    y = np.append(y, cm, axis=1)
    return y


def save_files(input_size, hidden_size, n_layers, output_size, activation_func, mod, mod_fname,
               xsc, xsc_fname, ysc, ysc_fname, pca, pca_fname, dfs, mod_folder):
    torch.save({
        'architect': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'n_layers': n_layers,
            'output_size': output_size,
            'activation_func': activation_func
        },
        'state_dict': mod.state_dict()
    }, mod_fname)
    joblib.dump(xsc, xsc_fname)
    pk.dump(ysc, open(ysc_fname, 'wb'))
    pk.dump(pca, open(pca_fname, 'wb'))
    dfs[0].to_csv(f'{mod_folder}/Training.csv', index=False)
    dfs[1].to_csv(f'{mod_folder}/Validation.csv', index=False)
    dfs[2].to_csv(f'{mod_folder}/Testing.csv', index=False)
    return


def get_model(mod_folder):
    """
    get prediction model with model folder path. Said folder must contain mod.pth, xsc.save, ysc.save, and pca.pkl
    :param mod_folder:
    :return:
    """
    mod_fname, xsc_fname, ysc_fname, pca_fname = setup_model_files(mod_folder)

    # Load the saved dictionary
    checkpoint = torch.load(mod_fname)

    # Extract architect and state_dict
    architect = checkpoint['architect']
    state_dict = checkpoint['state_dict']

    # Recreate the model using the saved architect
    mod = MLP_model(
        input_size=architect['input_size'],
        hidden_size=architect['hidden_size'],
        n_layers=architect['n_layers'],
        output_size=architect['output_size'],
        activation_function_name=architect['activation_func']
    )

    # Load the weights into the model
    mod.load_state_dict(state_dict)

    # Set the model to evaluation mode (standard practice, but not needed in this case)
    mod.eval()
    xsc = joblib.load(xsc_fname)
    ysc = joblib.load(ysc_fname)
    pca = pk.load(open(pca_fname, 'rb'))
    return mod, xsc, ysc, pca


def run_case(case, model, xscaler, yscaler, pca, mod_folder, plot=False, save=False):
    X, y, = case['X'].to_list(), case['y'].to_list()
    airfoil = case['airfoil name'].iloc[0]

    output = get_output(X, model, xscaler, yscaler, pca)
    y = devector_output(np.array(y))

    # print out error
    cl_mae = np.mean(abs(y[0]-output[0]))
    cd_mape = 100 * np.mean(abs(y[1] - output[1]) / (y[1] + 10e-10))
    cm_mae = np.mean(abs(y[2]-output[2]))
    ld_mae = np.mean(abs(y[0]/y[1]-output[0]/output[1]))
    print(f'|==============================================|')
    print(f'|           Lift Mean Absolute Error : {cl_mae:1.4f}  |')
    print(f'|Drag Mean Absolute Percentage Error : {cd_mape:1.4f}% |')
    print(f'|         Moment Mean Absolute Error : {cm_mae:1.4f}  |')
    print(f'|            L/D Mean Absolute Error : {ld_mae:1.4f}  |')
    print(f'|==============================================|')

    if plot:
        print(f'Running Airfoil: {airfoil}')
        color = 'k'
        marker = '^'
        if airfoil in ['RC410_0001', 'NACA0009']:
            color = 'tab:green'
            marker = 'x'
        elif airfoil in ['NACA0012_0001', 'OA212']:
            color = 'tab:blue'
            marker = '*'
        elif airfoil in ['NACA64A015_0001', 'NACA63215']:
            color = 'tab:cyan'
            marker = '^'

        if len(output[0]) == 25:
            aoa_list = np.linspace(-4, 20, 25)
        elif len(output[0]) == 31:
            aoa_list = np.linspace(-10, 20, 31)
        else:
            raise ValueError('angle of attack unknown!')

        plot_case(aoa_list, output, y, airfoil, color, marker)

        if save:
            plt.savefig(f'{mod_folder}/{airfoil}.png')
    return output, y


def get_train_valid_test_set(mod_folder):
    """get training, validation, and testing dataset used for the model"""
    df1 = pd.read_csv(f'{mod_folder}/Training.csv')
    df1['X'] = df1['X'].apply(lambda s: list(map(float, s.strip('[]').split())))
    df1['y'] = df1['y'].apply(lambda s: list(map(float, s.strip('[]').split(' '))))

    df2 = pd.read_csv(f'{mod_folder}/Validation.csv')
    df2['X'] = df2['X'].apply(lambda s: list(map(float, s.strip('[]').split())))
    df2['y'] = df2['y'].apply(lambda s: list(map(float, s.strip('[]').split(' '))))

    df3 = pd.read_csv(f'{mod_folder}/Testing.csv')
    df3['X'] = df3['X'].apply(lambda s: list(map(float, s.strip('[]').split())))
    df3['y'] = df3['y'].apply(lambda s: list(map(float, s.strip('[]').split(' '))))
    return [df1, df2, df3]


def find_error(dfs, model, xscaler, yscaler, pca, plot=False):
    if isinstance(dfs, list):
        c = ['b', 'r', 'k']
        label = ['Training Set', 'Validation Set', 'Testing Set']
        for i in range(len(dfs)):
            df = dfs[i]

            y = devector_output(df['y'].to_list())
            y_pred = get_output(df['X'].to_list(), model, xscaler, yscaler, pca)

            f = np.array([])
            for case in df['f'].to_list():
                f = np.hstack((f, list(map(int, case.strip('[]').split()))))
            y = y*f
            y_pred = y_pred*f

            # MSE and r2
            # print('\nMSE: %.10f \nR^2: %.10f' % (mean_squared_error(y, y_pred), r2_score(y, y_pred)))

            # find r2 and correlation
            ld_true = y[0]/(y[1]+1e-10)
            ld_pred = y_pred[0]/(y_pred[1]+1e-10)

            cl_r2 = r2_score(y[0], y_pred[0])
            cd_r2 = r2_score(y[1], y_pred[1])
            cm_r2 = r2_score(y[2], y_pred[2])
            ld_r2 = r2_score(ld_true, ld_pred)

            # find MAPE
            cl_mae = np.mean(abs(y[0] - y_pred[0]))
            cd_mape = 100*np.mean(abs((y[1] - y_pred[1])/(y[1] + 10e-10)))
            cm_mae = np.mean(abs(y[2] - y_pred[2]))
            ld_mae = np.mean(abs(ld_true - ld_pred))

            print(f'Lift Correlation r2: \n{cl_r2}')
            print(f'Drag Correlation r2: \n{cd_r2}')
            print(f'Moment Correlation r2: \n{cm_r2}')
            print(f'L/D Correlation r2: \n{ld_r2}')

            print('Cl RMSE:   %.10f' % np.sqrt(mean_squared_error(y[0], y_pred[0])))
            print('Cd RMSE:   %.10f' % np.sqrt(mean_squared_error(y[1], y_pred[1])))
            print('Cm RMSE:   %.10f' % np.sqrt(mean_squared_error(y[2], y_pred[2])))
            print('LD RMSE:   %.10f' % np.sqrt(mean_squared_error(ld_true, ld_pred)))

            print(f'|==============================================| {label[i]}')
            print(f'|           Lift Mean Absolute Error : {cl_mae:1.5f}  |')
            print(f'|Drag Mean Absolute Percentage Error : {cd_mape:1.5f}% |')
            print(f'|         Moment Mean Absolute Error : {cm_mae:1.5f}  |')
            print(f'|            L/D Mean Absolute Error : {ld_mae:1.5f}  |')
            print(f'|==============================================|')

            if plot is True:
                # plot_find_error_hist(y, y_pred)
                plot_find_error_scatter(y, y_pred, c[i], label[i])
                # if mf is not None and save:
                #     plt.savefig(f'Model/{mf}/Model_Performance.png')

    else:
        y = devector_output(dfs['y'].to_list())
        y_pred = get_output(dfs['X'].to_list(), model, xscaler, yscaler, pca)

        # MSE and r2
        print('\nMSE: %.10f \nR^2: %.10f' % (mean_squared_error(y, y_pred), r2_score(y, y_pred)))

        # find r2 and correlation
        ld_true = y[0] / y[1]
        ld_pred = y_pred[0] / y_pred[1]

        cl_r2 = r2_score(y[0], y_pred[0])
        cd_r2 = r2_score(y[1], y_pred[1])
        cm_r2 = r2_score(y[2], y_pred[2])
        ld_r2 = r2_score(ld_true, ld_pred)

        # find MAPE
        cl_mae = np.mean(abs(y[0] - y_pred[0]))
        cd_mape = 100 * np.mean(abs((y[1] - y_pred[1]) / (y[1] + 10e-10)))
        cm_mae = np.mean(abs(y[2] - y_pred[2]))
        ld_mae = np.mean(abs(ld_true - ld_pred))

        # print(f'Lift Correlation (r2, correlation factor): \n{cl_r2}; {cl_corr}')
        # print(f'Drag Correlation (r2, correlation factor): \n{cd_r2}; {cd_corr}')
        # print(f'Moment Correlation (r2, correlation factor): \n{cm_r2}; {cm_corr}')
        # print(f'L/D Correlation (r2, correlation factor): \n{ld_r2}; {ld_corr}')

        print(f'|==============================================| Dataframe')
        print(f'|           Lift Mean Absolute Error : {cl_mae:1.4f}  |')
        print(f'|Drag Mean Absolute Percentage Error : {cd_mape:1.4f}% |')
        print(f'|         Moment Mean Absolute Error : {cm_mae:1.4f}  |')
        print(f'|            L/D Mean Absolute Error : {ld_mae:1.4f}  |')
        print(f'|==============================================|')

        if plot is True:
            # plot_find_error_hist(y, y_pred)
            plot_find_error_scatter(y, y_pred, 'k', 'data')
            # if mf is not None and save:
            #     plt.savefig(f'Model/{mf}/Model_Performance.png')
    return


def find_airfoil_error(df, model, xscaler, yscaler, pca, plot=False):
    max_error = [0, 0, 0]
    error_airfoil = ['', '', '']
    airfoil_names = ["GS1", "NACA0012", "NACA0012_64", "NACA0015", "NACA64A015", "RC310", "RC410", "RCSC2",
                     "SC1095", "SC2110", "SSCA09", "V23010_158", "V43015_248", "VR12"]
    for airfoil in airfoil_names:
        for k in range(1, 41):
            airfoil_name = f'{airfoil}_{k:04d}'
            trim_df = df[df['airfoil name']==airfoil_name]
            X, y = trim_df['X'].to_list(), devector_output(trim_df['y'].to_list())
            y_pred = get_output(X, model, xscaler, yscaler, pca)

            cl_error = sum(abs(y[0] - y_pred[0]))
            cd_error = sum(abs(y[1] - y_pred[1]))
            cm_error = sum(abs(y[2] - y_pred[2]))

            if cl_error > max_error[0]:
                max_error[0] = cl_error
                error_airfoil[0] = airfoil_name
                
            if cd_error > max_error[1]:
                max_error[1] = cd_error
                error_airfoil[1] = airfoil_name
            
            if cm_error > max_error[2]:
                max_error[2] = cm_error
                error_airfoil[2] = airfoil_name

    print(error_airfoil)
    return


# def find_error_stddev(df, model, xscaler, yscaler, pca, plot=False, mf=None, save=0, n=0):
#     X, y, = df['X'].to_list(), df['y'].to_list()
#     airfoil = df['airfoil name'].iloc[0]
#
#     y_pred = get_output(X, model, xscaler, yscaler, pca)
#     y = devector_output(np.array(y))
#
#     # MSE and r2
#     print('MSE: %.10f \nR^2: %.10f\n' % (mean_squared_error(y, y_pred), r2_score(y, y_pred)))
#
#     if plot:
#         X_std = xscaler.transform(X)
#         X_std = torch.tensor(X_std, dtype=torch.float32)
#         plot_find_error_stddev(y, y_pred, X_std, n)
#
#         # if mf is not None and save:
#         #     plt.savefig(f'Model/{mf}/Model_Performance.png')
#         # plt.show()
#     return


def run_unknown(file_path, model, xscaler, yscaler, pca, deg=8):
    """run and plot unknown cases given airfoil coordinate file"""
    curve_data, [corr, sv, angle, err] = fs.fit_spline(file_path, 0.5, 1.0, deg)
    if pd.isna(corr[0]) or pd.isna(corr[1]):
        print('FAILED')
        return
    if corr[0] < 2.75:
        print(f'lower surface poor fit, corr<3 | corr = {corr[0]}')
        curve_data, [corr, sv, angle, err] = fs.fit_spline(file_path, 0.5, 1.0, deg)
    X = [curve_data.to_list()]

    output = get_output(X, model, xscaler, yscaler, pca)
    plot_unknown(output)
    return output


def run_designed_airfoil(curve_data, model, xscaler, yscaler, pca, plot=False):
    """run and plot unknown cases given airfoil parameters"""
    X = [curve_data]
    output = get_output(X, model, xscaler, yscaler, pca)

    if plot:
        plot_unknown(output)
    return output


def get_output(X, model, xscaler, yscaler, pca):
    """get model output"""
    X_std = xscaler.transform(X)
    X_std = torch.tensor(X_std, dtype=torch.float32)
    y = model(X_std).detach().numpy()
    y = pca.inverse_transform(y)
    y = descale_label(y, yscaler)
    y = devector_output(y)
    return y


def devector_output(y):
    """
    :param y: [[cl cd cm]_1 [cl cd cm]_2 ... [cl cd cm]_n]
    :return y_decon: [[cl_1 cl_2 ... cl_n], [cd_1 cd_2 ... cd_n], [cm_1 cm_2 ... cm_n]
    This function rearrange the model output to workable format for analysis
    """

    # data_size = int(y.shape[1]/5)     # if y is a numpy array
    data_size = int(len(y[0])/3)        # if y is a list

    cl, cd, cm, Xbot, Xtop = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for case in y:
        cl = np.hstack((cl, case[:data_size]))
        cd = np.hstack((cd, case[data_size:2*data_size])).astype(float)
        cm = np.hstack((cm, case[2*data_size:3*data_size]))
    y_decon = np.vstack((cl, 10**cd, cm))
    return y_decon


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # default='warn'
    pd.options.display.max_columns = None  # default=NotEnough
    pd.set_option('expand_frame_repr', False)
    main()
