"""
Tak Yeung, tyeung@umd.edu
shock
"""
import os
import pickle as pk
import random
import time as t

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_auc_score, roc_curve, \
    precision_recall_curve, average_precision_score, mean_absolute_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

import fit_spline as fs
from plotting_functions import *


# Define your model with configurable number of hidden layers and activation functions
class classifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, activation_function_name):
        super(classifier, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Add hidden layers
        for _ in range(n_layers):
            if len(self.hidden_layers) == 0:
                incoming_size = input_size
            else:
                incoming_size = hidden_size
            self.hidden_layers.append(nn.Linear(incoming_size, hidden_size))
            self.dropouts.append(nn.Dropout(p=0.))  # Improving neural networks by preventing co-adaptation of feature detectors, NOT GOOD

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # normalization function to confine output between 0 and 1

        # Set activation function
        self.activation = self._get_activation_function(activation_function_name)

    def _get_activation_function(self, name):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations[name]

    def forward(self, x):
        for layer, dropout in zip(self.hidden_layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


class regressor(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, activation_function_name):
        super(regressor, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activation = self._get_activation_function(activation_function_name)

        # Add hidden layers
        for _ in range(n_layers):
            if len(self.hidden_layers) == 0:
                incoming_size = input_size
            else:
                incoming_size = hidden_size
            self.hidden_layers.append(nn.Linear(incoming_size, hidden_size))
            self.dropouts.append(nn.Dropout(p=0.0))  # its just bad

        self.output_layer = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def _get_activation_function(self, name):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }
        return activations[name]

    def forward(self, x):
        for layer, dropout in zip(self.hidden_layers, self.dropouts):
            x = self.activation(layer(x))
            x = dropout(x)

        x = self.output_layer(x)
        # x = self.sigmoid(x)
        return x


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        # stop if no improvement is made after n epochs
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():
    dataset_path = 'Datasets/ma0.7_re4666700_shock.csv'

    airfoil_name = None  # toggle this if you want to run the entire set
    airfoil_name = 'V43015_248_0001'

    # run classifier
    classifier_folder = 'Model/classifier_TSPM'
    class_mod, class_xsc = get_model(classifier_folder, 'classifier')

    mod_name = 'regressor_TSPM'
    lwr_mod_folder = f'Model/lwr_{mod_name}'
    upr_mod_folder = f'Model/upr_{mod_name}'
    lwr_reg_mod, lwr_reg_xsc = get_model(lwr_mod_folder, 'regressor')
    upr_reg_mod, upr_reg_xsc = get_model(upr_mod_folder, 'regressor')

    # loop all baseline airfoil
    airfoil_names = ["GS1", "NACA0012", "NACA0012_64", "NACA0015", "NACA64A015", "RC310", "RC410", "RCSC2",
                     "SC1095", "SC2110", "SSCA09", "V23010_158", "V43015_248", "VR12"]
    airfoil_names = ['NACA64A015', 'RC410', 'NACA0012']
    # airfoil_names = ['SC1095']

    for airfoil in airfoil_names:
        for k in range(1):  # change to 41 if looping all airfoils
            airfoil_name = f'{airfoil}_{k + 1:04d}'
            print(f'Running {airfoil_name}')

            color = 'k'
            marker = '^'
            if airfoil in ['RC410', 'NACA0009']:
                color = 'tab:green'
                marker = 'x'
            elif airfoil in ['NACA0012', 'OA212']:
                color = 'tab:blue'
                marker = '*'
            elif airfoil in ['NACA64A015', 'NACA63215']:
                color = 'tab:cyan'
                marker = '^'

            # run classifier
            label = ['lwr_flag', 'upr_flag']
            df = get_df(dataset_path, label, airfoil=airfoil_name)
            X = df['X'].to_list()
            lwr_flag, upr_flag = run_classifier(X, class_mod, class_xsc)

            # run regression model
            lwr_df = df[pd.Series(lwr_flag).astype(bool)]
            lwr_shock = run_regressor(lwr_df, lwr_reg_mod, lwr_reg_xsc)

            upr_df = df[pd.Series(upr_flag).astype(bool)]
            upr_shock = run_regressor(upr_df, upr_reg_mod, upr_reg_xsc)

            # plot predicted result
            plt.figure(27, figsize=(5.5, 5))
            lwr_aoa = [row[-1] for row in lwr_df['X']]
            pred_points = np.array([lwr_aoa, np.squeeze(lwr_shock)])
            pred_points = pred_points[:, pred_points[0].argsort()]
            plt.plot(pred_points[0], pred_points[1], zorder=3, linewidth=2.5, linestyle=':', marker=marker, color=color, label=f'{airfoil} Predicted Location')
            # plt.scatter(pred_points[0], pred_points[1], zorder=3)

            pred_points[1] = [round(num, 3) for num in pred_points[1]]
            # print(pred_points)

            upr_aoa = [row[-1] for row in upr_df['X']]
            pred_points = np.array([upr_aoa, np.squeeze(upr_shock)])
            pred_points = pred_points[:, pred_points[0].argsort()]
            plt.plot(pred_points[0], pred_points[1], zorder=3, linewidth=2.5, linestyle=':', marker=marker, color=color)
            # plt.scatter(pred_points[0], pred_points[1], zorder=3)

            pred_points[1] = [round(num, 3) for num in pred_points[1]]
            # print(pred_points)

            # get true data
            lwr_label = ['lwr_r/arc']
            upr_label = ['upr_r/arc']
            true_lwr_df = get_df(dataset_path, lwr_label, flag='lwr_flag', airfoil=airfoil_name)
            true_upr_df = get_df(dataset_path, upr_label, flag='upr_flag', airfoil=airfoil_name)

            true_lwr_aoa = [row[-1] for row in true_lwr_df['X']]
            true_points = np.array([true_lwr_aoa, true_lwr_df['y']])
            true_points = true_points[:, true_points[0].argsort()]
            # print(true_points)
            plt.plot(true_points[0], true_points[1], zorder=2, linewidth=2, marker=marker, color=color, label=f'{airfoil} CFD Location')
            # plt.scatter(true_points[0], true_points[1], zorder=2)

            true_upr_aoa = [row[-1] for row in true_upr_df['X']]
            true_points = np.array([true_upr_aoa, true_upr_df['y']])
            true_points = true_points[:, true_points[0].argsort()]
            # print(true_points)
            plt.plot(true_points[0], true_points[1], zorder=2, linewidth=2, marker=marker, color=color)
            # plt.scatter(true_points[0], true_points[1], zorder=2)

    # plotting stuff
    label_font = {'weight': 'bold', 'size': 'large'}
    grid_lines()
    plt.xlabel(r'Angle of Attack , deg', **label_font)
    plt.ylabel(r'Shock Location, Airfoil Arc-length', **label_font)
    plt.xlim([-5, 21])
    plt.ylim([0.22, 0.84])
    plt.legend(framealpha=1)
    plt.tight_layout()
    plt.show()
    return


def classifier_main():
    # ======================= Training Parameters =======================
    # dataset path
    dataset_path = 'Datasets/ma0.7_re4666700_shock.csv'

    # model directory
    mod_folder = f'Model/classifier_TSPM'

    # set up model directory and files' path
    mod_fname, xsc_fname = setup_model_files(mod_folder)

    # interested label(s)
    label = ['lwr_flag', 'upr_flag']

    # ======================= Train Classifier Model =======================
    # start timing
    st = t.time()

    # get dataframe based on provided dataset
    df = get_df(dataset_path, label)

    # read the pdata into a dataframe
    model = 'classifier'
    input_size = len(df['X'][0])
    hidden_size = 20  # hidden size
    n_layers = 2  # number of hidden layers
    output_size = len(df['y'][0])  # output size
    activation_func = 'relu'  # activation function
    loss_func = 'BCELoss'  # loss function
    save_plot = 1  # don't save=0 | save=1

    # train the model
    mod, xsc, dataframes = train_model(model, df, input_size, hidden_size, n_layers, output_size, activation_func,
                                       loss_func, visualize_training=True, mod_folder=mod_folder)
    # save model and scaler
    save_files(input_size, hidden_size, n_layers, output_size, activation_func, mod, mod_fname, xsc, xsc_fname,
               dataframes, mod_folder)

    print(f'runtime: {t.time() - st}sec!\n')

    # ==================== Evaluate Classifier ====================
    # file_path = f'Raw_Data/Airfoil/NACA0009.txt'
    # mod, xsc, ysc, pca = get_model('ham2d/DP_mod')
    # run_unknown(file_path, mod, xsc, ysc, pca)
    # plt.show()

    # load classifier
    mod, xsc = get_model(mod_folder, 'classifier')

    dataframes = get_train_valid_test_set(mod_folder)
    evaluate_classifier(dataframes, mod, xsc)

    # dataframes = get_df('Datasets/ma0.7_re04666700_SC1094_R8_shock.csv', label)
    # evaluate_classifier(dataframes, mod, xsc)
    return


def regressor_main():
    # ======================= Training Parameters =======================
    # dataset path
    dataset_path = 'Datasets/ma0.7_re4666700_shock.csv'

    # model directory
    mod_name = 'regressor_TSPM'
    lwr_mod_folder = f'Model/lwr_{mod_name}'
    upr_mod_folder = f'Model/upr_{mod_name}'

    # set up model directory and files' path
    lwr_mod_fname, lwr_xsc_fname = setup_model_files(lwr_mod_folder)
    upr_mod_fname, upr_xsc_fname = setup_model_files(upr_mod_folder)

    # interested label(s)
    lwr_label = ['lwr_r/arc']
    upr_label = ['upr_r/arc']

    # ======================= Train Regressor Model =======================
    # start timing
    st = t.time()

    # get dataframe based on provided dataset
    TE_shock = True
    lwr_df = get_df(dataset_path, lwr_label, flag='lwr_flag', TE_shock=TE_shock)
    upr_df = get_df(dataset_path, upr_label, flag='upr_flag', TE_shock=TE_shock)

    # read the pdata into a dataframe
    model = 'regressor'
    input_size = len(lwr_df['X'][0])
    hidden_size = 40  # hidden size
    n_layers = 4  # number of hidden layers
    output_size = len(lwr_df['y'][0])  # output size (n modes for output vector)
    activation_func = 'relu'  # activation function
    loss_func = 'MSELoss'  # loss function

    # # train and save lower shock model
    # lwr_mod, lwr_xsc, lwr_dataframes = train_model(model, lwr_df, input_size, hidden_size, n_layers, output_size,
    #                                                activation_func, loss_func, visualize_training=True,
    #                                                mod_folder=lwr_mod_folder)
    # save_files(input_size, hidden_size, n_layers, output_size, activation_func,
    #            lwr_mod, lwr_mod_fname,
    #            lwr_xsc, lwr_xsc_fname,
    #            lwr_dataframes, lwr_mod_folder)

    # # train and save upper shock model
    # upr_mod, upr_xsc, upr_dataframes = train_model(model, upr_df, input_size, hidden_size, n_layers, output_size,
    #                                                activation_func, loss_func, visualize_training=True,
    #                                                mod_folder=upr_mod_folder)
    # save_files(input_size, hidden_size, n_layers, output_size, activation_func,
    #            upr_mod, upr_mod_fname,
    #            upr_xsc, upr_xsc_fname,
    #            upr_dataframes, upr_mod_folder)

    print(f'runtime: {t.time() - st}sec!\n')

    # load regressors
    dataframes = get_train_valid_test_set(lwr_mod_folder)
    # dataframes = get_df('Datasets/ma0.7_re04666700_SC1094_R8_shock.csv', lwr_label, flag='lwr_flag')
    mod, xsc = get_model(lwr_mod_folder, 'regressor')
    evaluate_regressor(dataframes, mod, xsc, plot=True)
    plt.legend(framealpha=1)

    dataframes = get_train_valid_test_set(upr_mod_folder)
    # dataframes = get_df('Datasets/ma0.7_re04666700_SC1094_R8_shock.csv', upr_label, flag='upr_flag')
    mod, xsc = get_model(upr_mod_folder, 'regressor')
    evaluate_regressor(dataframes, mod, xsc, plot=True)

    plt.show()
    return


def setup_model_files(mod_folder_path):
    if not os.path.isdir(f'{mod_folder_path}'):
        os.mkdir(f'{mod_folder_path}')
    mod_fname = f'{mod_folder_path}/mod.pth'
    xsc_fname = f'{mod_folder_path}/xsc.save'
    return mod_fname, xsc_fname


def get_df(file_paths, label, flag=None, airfoil=None, TE_shock=False):
    # initialize an empty dataframe
    df = pd.DataFrame()

    # read data file(s)
    if isinstance(file_paths, str):
        df = pd.read_csv(file_paths, index_col=0)
    elif isinstance(file_paths, list):
        for path in file_paths:
            _df = pd.read_csv(path, index_col=0)
            df = pd.concat([df, _df], axis=0)
    else:
        raise TypeError('INVALID INPUT')

    # run thru a flag check before extracting data
    if flag is not None:
        df = df[df[flag] == 1]

        # filter out trailing edge shock
        if flag == 'lwr_flag' and TE_shock is False:
            df = df[df['lwr_r/arc'] > 0.05]

    # if a specific airfoil is called, filter out the dataframe
    if airfoil is not None:
        df = df[df['airfoil name'] == airfoil]
        df.reset_index(drop=True, inplace=True)

    # extract features and designated labels
    X, y = [], []

    # airfoil parameterization coefficients
    # 8th order Chebyshev | a1-a9 : lower curve | a10-a18 : upper curve | a19 : TE thickness
    features = [f'a{j}' for j in range(1, 20)]
    features.append('aoa')

    # march down dataframe to extract features and labels
    for i in range(len(df)):
        row = df.iloc[i]
        X.append(np.array(row[features]))
        y.append(np.array(row[label]))

    return pd.DataFrame({'X': X, 'y': y, 'airfoil name': df['airfoil name']})


def train_model(model, df, input_size, hidden_size, n_layers, output_size,
                activation_func, loss_func, visualize_training=False, mod_folder=None):
    
    # ############################################## HYPERPARAMETERS ##############################################
    learning_rate = 0.00075
    beta1 = 0.9
    beta2 = 0.99
    lambda_ = 1e-4  # L2 regularization
    num_epochs = 1000

    # Split dataset into training, validation, and testing set (60, 20, 20)
    train_df, _df = train_test_split(df, test_size=0.3)
    validation_df, test_df = train_test_split(_df, test_size=0.5)

    # ==================================== Standardize and scale the data sets ====================================
    xsc = StandardScaler()
    # xsc = RobustScaler()
    X_train_std = xsc.fit_transform(train_df['X'].to_list())
    X_validation_std = xsc.transform(validation_df['X'].to_list())
    X_test_std = xsc.transform(test_df['X'].to_list())

    y_train, y_validation, y_test = train_df['y'].to_list(), validation_df['y'].to_list(), test_df['y'].to_list()
    # ================================== Convert numpy arrays to PyTorch tensors ==================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'USING DEVICE = {device}.')

    X_train_std = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_validation_std = torch.tensor(X_validation_std, dtype=torch.float32, device=device)
    X_test_std = torch.tensor(X_test_std, dtype=torch.float32, device=device)
    y_train = torch.tensor(np.array(y_train, dtype=float), dtype=torch.float32, device=device)
    y_validation = torch.tensor(np.array(y_validation, dtype=float), dtype=torch.float32, device=device)
    y_test = torch.tensor(np.array(y_test, dtype=float), dtype=torch.float32, device=device)

    # ===================================== Set-up and train the PyTorch model =====================================
    # create dataset
    train_dataset = TensorDataset(X_train_std, y_train)
    val_dataset = TensorDataset(X_validation_std, y_validation)
    test_dataset = TensorDataset(X_test_std, y_test)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    if model == 'classifier':
        mod = classifier(input_size, hidden_size, n_layers, output_size, activation_func).to(device)
    elif model == 'regressor':
        mod = regressor(input_size, hidden_size, n_layers, output_size, activation_func).to(device)
    else:
        raise ValueError('Pick Either classifier or regressor!')

    loss_functions = {'HuberLoss': nn.HuberLoss(),
                      'MSELoss': nn.MSELoss(),
                      'L1Loss': nn.L1Loss(),
                      'BCELoss': nn.BCELoss()}
    criterion = loss_functions[loss_func]
    optimizer = torch.optim.Adam(mod.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=lambda_)
    early_stopper = EarlyStopper(patience=50, min_delta=0)

    # =========== Training Loop ===========
    print(f"Training \n")
    epoch, training_loss_list, validation_loss_list, testing_loss_list = 0, [], [], []

    for epoch in range(num_epochs):
        mod.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = mod(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        training_loss_list.append(train_loss / len(train_loader.dataset))

        # validation set monitoring
        mod.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = mod(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        validation_loss_list.append(val_loss / len(val_loader.dataset))

        # test set evaluation
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = mod(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        testing_loss_list.append(test_loss / len(test_loader.dataset))

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {training_loss_list[-1]:.4e} | Val Loss: {validation_loss_list[-1]:.4e}")
        if early_stopper.early_stop(val_loss):
            print('EARLY STOPPING!')
            break

    # =========== Fitting and testing mod using NN ===========
    mod.eval()
    y_train = y_train.cpu().detach().numpy()
    y_validation = y_validation.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    y_train_pred = mod(X_train_std).cpu().detach().numpy()
    y_validation_pred = mod(X_validation_std).cpu().detach().numpy()
    y_test_pred = mod(X_test_std).cpu().detach().numpy()

    # =========== Plot Loss vs Epoch ===========
    if visualize_training:
        plt.figure(42)
        plt.plot(training_loss_list, 'b')
        plt.plot(validation_loss_list, 'r.-')
        plt.plot(testing_loss_list, 'k--')
        plt.legend(['training loss', 'validation loss', 'testing loss'])
        plt.xlabel('Epoch')
        plt.xlim(0, len(training_loss_list))
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
    return mod, xsc, [train_df, validation_df, test_df]


def save_files(input_size, hidden_size, n_layers, output_size, activation_func, mod, mod_fname,
               xsc, xsc_fname, dfs, mod_folder):
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
    dfs[0].to_csv(f'{mod_folder}/Training.csv', index=False)
    dfs[1].to_csv(f'{mod_folder}/Validation.csv', index=False)
    dfs[2].to_csv(f'{mod_folder}/Testing.csv', index=False)
    return


def get_model(mod_folder, model):
    """
    get prediction model with model folder path. Said folder must contain mod.pth and xsc.save
    """
    mod_fname, xsc_fname = setup_model_files(mod_folder)

    # Load the saved dictionary
    checkpoint = torch.load(mod_fname)

    # Extract architect and state_dict
    architect = checkpoint['architect']
    state_dict = checkpoint['state_dict']

    # Recreate the model using the saved architect
    if model == 'classifier':
        mod = classifier(
            input_size=architect['input_size'],
            hidden_size=architect['hidden_size'],
            n_layers=architect['n_layers'],
            output_size=architect['output_size'],
            activation_function_name=architect['activation_func']
        )
    elif model == 'regressor':
        mod = regressor(
            input_size=architect['input_size'],
            hidden_size=architect['hidden_size'],
            n_layers=architect['n_layers'],
            output_size=architect['output_size'],
            activation_function_name=architect['activation_func']
        )
    else:
        raise ValueError('INCORRECT MODEL TYPE!')

    # Load the weights into the model
    mod.load_state_dict(state_dict)

    # Set the model to evaluation mode (standard practice, but not needed in this case)
    mod.eval()
    xsc = joblib.load(xsc_fname)
    return mod, xsc


def run_classifier(X, model, xscaler):
    X_std = xscaler.transform(X)
    X_std = torch.tensor(X_std, dtype=torch.float32)
    y_pred = model(X_std).detach().numpy()
    # print(y_pred)
    y_pred = np.round(y_pred)
    lwr_flag_pred = y_pred[:, 0]
    upr_flag_pred = y_pred[:, 1]
    return lwr_flag_pred, upr_flag_pred


def run_regressor(df, model, xscaler):
    X_std = xscaler.transform(df['X'].to_list())
    X_std = torch.tensor(X_std, dtype=torch.float32)
    y_pred = model(X_std).detach().numpy()
    return y_pred


def run_regressor_X(X, model, xscaler):
    X_std = xscaler.transform(X)
    X_std = torch.tensor(X_std, dtype=torch.float32)
    y_pred = model(X_std).detach().numpy()
    return y_pred


def get_train_valid_test_set(mod_folder):
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


def evaluate_classifier(dfs, model, xscaler):
    if isinstance(dfs, list):
        lwr_flags_pred = []
        upr_flags_pred = []
        c = ['b', 'r', 'k']
        label = ['Training Set', 'Validation Set', 'Testing Set']
        for i in range(len(dfs)):
            df = dfs[i]
            X_std = xscaler.transform(df['X'].to_list())
            X_std = torch.tensor(X_std, dtype=torch.float32)
            y_pred = model(X_std).detach().numpy()
            # y_pred = np.round(y_pred)
            lwr_flag_pred = y_pred[:, 0]
            upr_flag_pred = y_pred[:, 1]

            y = df['y'].to_list()
            lwr_flag = [int(_y[0]) for _y in y]
            upr_flag = [int(_y[1]) for _y in y]

            print(lwr_flag)
            print(lwr_flag_pred)

            lwr_flags_pred = [lwr_flags_pred, lwr_flag_pred]
            upr_flags_pred = [upr_flags_pred, upr_flag_pred]

            # AUC ROC Curve
            lwr_auc_score = roc_auc_score(lwr_flag, lwr_flag_pred)
            upr_auc_score = roc_auc_score(upr_flag, upr_flag_pred)
            print(f"lower shock AUC-ROC: {lwr_auc_score:.4f}")
            print(f"upper shock AUC-ROC: {upr_auc_score:.4f}")

            lfpr, ltpr, thresholds = roc_curve(lwr_flag, lwr_flag_pred)
            ufpr, utpr, thresholds = roc_curve(upr_flag, upr_flag_pred)

            lwr_precision, lwr_recall, thresholds = precision_recall_curve(lwr_flag, lwr_flag_pred)
            lwr_ap_score = average_precision_score(lwr_flag, lwr_flag_pred)

            upr_precision, upr_recall, thresholds = precision_recall_curve(upr_flag, upr_flag_pred)
            upr_ap_score = average_precision_score(upr_flag, upr_flag_pred)

            # plot ROC curve
            label_font = {'weight': 'bold', 'size': 14}
            tick_font = 12
            plt.figure(1, figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.minorticks_on()
            plt.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
            plt.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
            plt.plot(lfpr, ltpr, label=f"Lower Shock AUC = {lwr_auc_score:.4f}", linewidth=2)
            plt.plot(ufpr, utpr, label=f"Upper Shock AUC = {upr_auc_score:.4f}", linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
            plt.xlabel("False Positive Rate", **label_font)
            plt.ylabel("True Positive Rate", **label_font)
            # plt.title("ROC Curve")
            plt.legend(fontsize=tick_font, framealpha=1)
            plt.xticks(fontsize=tick_font)
            plt.yticks(fontsize=tick_font)
            plt.tight_layout()

            # plot Precision-Recall Curve
            plt.subplot(1, 2, 2)
            plt.minorticks_on()
            plt.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
            plt.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
            plt.plot(lwr_recall, lwr_precision, label=f'Lower Shock AP = {lwr_ap_score:.2f}')
            plt.plot(upr_recall, upr_precision, label=f'Lower Shock AP = {upr_ap_score:.2f}')
            plt.xlabel('Recall', **label_font)
            plt.ylabel('Precision', **label_font)
            # plt.title('Precision-Recall Curve')
            plt.legend(fontsize=tick_font, framealpha=1)
            plt.xticks(fontsize=tick_font)
            plt.yticks(fontsize=tick_font)
            plt.tight_layout()

            y_pred = np.round(y_pred)
            lwr_flag_pred = y_pred[:, 0]
            upr_flag_pred = y_pred[:, 1]

            # confusion matrix for lower surface
            lwr_cm = confusion_matrix(lwr_flag, lwr_flag_pred)
            plt.figure(2, figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.heatmap(lwr_cm,
                        annot=True,
                        annot_kws={"size": 14},
                        fmt='g',
                        xticklabels=['No Lower shock', 'Lower shock'],
                        yticklabels=['No Lower shock', 'Lower shock'])
            plt.ylabel('Actual', fontsize=14)
            plt.xlabel('Predicted', fontsize=14)
            plt.gca().xaxis.set_label_position('top')
            plt.gca().xaxis.tick_top()

            # confusion matrix for upper surface
            upr_cm = confusion_matrix(upr_flag, upr_flag_pred)
            plt.subplot(1, 2, 2)
            sns.heatmap(upr_cm,
                        annot=True,
                        annot_kws={"size": 14},
                        fmt='g',
                        xticklabels=['No Upper shock', 'Upper shock'],
                        yticklabels=['No Upper shock', 'Upper shock'])
            plt.ylabel('Actual', fontsize=14)
            plt.xlabel('Predicted', fontsize=14)
            plt.gca().xaxis.set_label_position('top')
            plt.gca().xaxis.tick_top()
            plt.show()
        return lwr_flags_pred, upr_flags_pred

    else:  # only one dataframe
        X_std = xscaler.transform(dfs['X'].to_list())
        X_std = torch.tensor(X_std, dtype=torch.float32)
        y_pred = model(X_std).detach().numpy()
        y_pred = np.round(y_pred)
        lwr_flag_pred = y_pred[:, 0]
        upr_flag_pred = y_pred[:, 1]

        y = dfs['y'].to_list()
        lwr_flag = [_y[0] for _y in y]
        upr_flag = [_y[1] for _y in y]

        print(lwr_flag)
        print(lwr_flag_pred)
        
        # AUC ROC Curve
        lwr_auc_score = roc_auc_score(lwr_flag, lwr_flag_pred)
        upr_auc_score = roc_auc_score(upr_flag, upr_flag_pred)
        print(f"lower shock AUC-ROC: {lwr_auc_score:.4f}")
        print(f"upper shock AUC-ROC: {upr_auc_score:.4f}")

        lfpr, ltpr, thresholds = roc_curve(lwr_flag, lwr_flag_pred)
        ufpr, utpr, thresholds = roc_curve(upr_flag, upr_flag_pred)

        plt.figure(1, figsize=(6, 5.5))
        plt.minorticks_on()
        plt.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
        plt.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
        plt.plot(lfpr, ltpr, label=f"AUC = {lwr_auc_score:.4f}")
        plt.plot(ufpr, utpr, label=f"AUC = {upr_auc_score:.4f}")
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # confusion matrix
        lwr_cm = confusion_matrix(lwr_flag, lwr_flag_pred)
        plt.figure(2, figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.heatmap(lwr_cm,
                    annot=True,
                    annot_kws={"size": 14},
                    fmt='g',
                    xticklabels=['shock', 'No shock'],
                    yticklabels=['shock', 'No shock'])
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()

        upr_cm = confusion_matrix(upr_flag, upr_flag_pred)
        plt.subplot(1, 2, 2)
        sns.heatmap(upr_cm,
                    annot=True,
                    annot_kws={"size": 14},
                    fmt='g',
                    xticklabels=['shock', 'No shock'],
                    yticklabels=['shock', 'No shock'])
        plt.ylabel('Actual', fontsize=14)
        plt.xlabel('Predicted', fontsize=14)
        plt.gca().xaxis.set_label_position('top')
        plt.gca().xaxis.tick_top()
        plt.show()
        return lwr_flag_pred, upr_flag_pred


def evaluate_regressor(df, model, scaler, airfoil=None, plot=False):
    if isinstance(df, list):
        c = ['b', 'r', 'k']
        label = ['Training Set', 'Validation Set', 'Testing Set']
        for i in range(len(df)):
            _df = df[i]
            X_std = scaler.transform(_df['X'].to_list())
            X_std = torch.tensor(X_std, dtype=torch.float32)
            y_pred = model(X_std).detach().numpy()
            y = _df['y'].to_list()

            # print(f'True: \n{y} \nPredicted: \n{y_pred}')
            # MSE and r2
            print('MAE:    %.10f' % mean_absolute_error(y, y_pred))
            print('RMSE:   %.10f' % np.sqrt(mean_squared_error(y, y_pred)))
            print('R^2:    %.10f' % r2_score(y, y_pred))
            print('stddev: %.10f' % np.std(y_pred-y))
            # find correlation
            arc_ratio_r2 = r2_score(y, y_pred)
            arc_ratio_corr = -np.log10(1 - arc_ratio_r2)

            print(f'shock location prediction (r2, correlation factor): \n{arc_ratio_r2}; {arc_ratio_corr}')

            if plot is True:
                plot_prediction(y, y_pred, c[i], label[i])

    else:
        X_std = scaler.transform(df['X'].to_list())
        X_std = torch.tensor(X_std, dtype=torch.float32)
        y_pred = model(X_std).detach().numpy()
        y = df['y'].to_list()

        y = [_y.tolist() for _y in y]
        print(f'True: \n{y} \nPredicted: \n{y_pred.T}')
        # MSE and r2
        print('\n MSE: %.10f \nR^2: %.10f' % (mean_squared_error(y, y_pred), r2_score(y, y_pred)))

        # find correlation
        arc_ratio_r2 = r2_score(y, y_pred)
        arc_ratio_corr = -np.log10(1 - arc_ratio_r2)

        print(f'shock location prediction (r2, correlation factor): \n{arc_ratio_r2}; {arc_ratio_corr} \n')

        if plot is True:
            plot_prediction(y, y_pred, 'k', 'data')
    return


# def find_error_stddev(df, model, xscaler, plot=False, mf=None, save=0, n=0):
#     X_std = xscaler.transform(df['X'].to_list())
#     X_std = torch.tensor(X_std, dtype=torch.float32)
#     y_pred = model(X_std).detach().numpy()
#
#     y = df['y'].to_list()
#
#     # MSE and r2
#     print('MSE: %.10f \nR^2: %.10f\n' % (mean_squared_error(y, y_pred), r2_score(y, y_pred)))
#
#     if plot:
#         plot_find_error_stddev(y, y_pred, X_std, n)
#
#         # if mf is not None and save:
#         #     plt.savefig(f'Model/{mf}/Model_Performance.png')
#         # plt.show()
#     return


def run_case(case, model, xscaler, mod_folder, plot=False, save=False):
    X, y, = case['X'].to_list(), case['y'].to_list()
    airfoil = case['airfoil name']

    X_std = xscaler.transform(X)
    X_std = torch.tensor(X_std, dtype=torch.float32)
    output = model(X_std).detach().numpy()

    # if plot:
    #     print(f'Running Airfoil: {airfoil}')
    #     solver = mod_folder.split('/')[0]
    #     plot_case(output, y, case['airfoil name'].iloc[0], mod_folder, solver)
    #
    #     if save:
    #         plt.savefig(f'{mod_folder}/{airfoil}.png')
    return output, y


def run_design_airfoil(curve_data, class_mod, class_xsc, lwr_reg_mod, lwr_reg_xsc, upr_reg_mod, upr_reg_xsc):
    # construct features
    X = []
    aoa_list = np.linspace(-4, 20, 25)
    for aoa in aoa_list:
        _X = curve_data.copy()
        _X.append(aoa)
        X.append(_X)

    # run classifier
    lwr_flag, upr_flag = run_classifier(X, class_mod, class_xsc)
    lwr_aoa = aoa_list[lwr_flag == 1]
    upr_aoa = aoa_list[upr_flag == 1]

    # run regression model
    lwr_X = np.array(X)[lwr_flag == 1]
    lwr_shock = run_regressor_X(lwr_X, lwr_reg_mod, lwr_reg_xsc)
    upr_X = np.array(X)[upr_flag == 1]
    upr_shock = run_regressor_X(upr_X, upr_reg_mod, upr_reg_xsc)

    lwr_pred_points = np.array([lwr_aoa, np.squeeze(lwr_shock)])
    lwr_pred_points = lwr_pred_points[:, lwr_pred_points[0].argsort()]
    upr_pred_points = np.array([upr_aoa, np.squeeze(upr_shock)])
    upr_pred_points = upr_pred_points[:, upr_pred_points[0].argsort()]

    shock_loc = np.concatenate((lwr_pred_points, upr_pred_points), axis=1)
    return shock_loc


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None  # default='warn'
    pd.options.display.max_columns = None  # default=NotEnough
    pd.set_option('expand_frame_repr', False)
    # classifier_main()
    # regressor_main()
    main()
