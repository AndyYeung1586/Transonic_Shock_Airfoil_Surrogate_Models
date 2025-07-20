import random
import matplotlib.pyplot as plt
import numpy as np


# ======================================================================================
# =                                  Predict Shock                                     =
# ======================================================================================
def plot_prediction(y, y_pred, c, label):
    label_font = {'weight': 'bold', 'size': 'large'}

    plt.figure(1, (5.5, 5))
    grid_lines()
    plt.scatter(y, y_pred, s=10, c=c, alpha=1, label=label, zorder=2)
    plt.plot([min(y), max(y)], [min(y), max(y)], c='m', linestyle='--', zorder=3)

    plt.xlabel('CFD shock location, airfoil arc-length', **label_font)
    plt.ylabel('Predicted shock location, airfoil arc-length', **label_font)
    plt.xlim([0.24, 0.82])
    plt.ylim([0.24, 0.82])
    plt.tight_layout()

    # plt.figure(37, (5.5, 5))
    # grid_lines()
    # plt.scatter(y_pred, abs(y_pred-y), s=10, c=c, alpha=1, label=label, zorder=2)
    #
    # plt.xlabel('Predicted shock Location, airfoil arc-length', **label_font)
    # plt.ylabel('Absolute error, airfoil arc-length', **label_font)
    # plt.tight_layout()
    return


# ======================================================================================
# =                        Train Aerodynamics Prediction Model                         =
# ======================================================================================
def plot_find_error_hist(y, y_pred):
    s = 10
    c = 'k'
    alpha = 0.075
    label_font = {'weight': 'bold', 'size': 'large'}
    title_font = {'weight': 'bold', 'size': 'x-large'}
    aoa = np.linspace(-4, 20, 25)

    plt.figure(10, (10, 9.5))
    plt.subplot(2, 2, 1)
    # plt.title(f'Lift Comparison, R2={cl_r2:.4f}', **title_font)
    plt.plot(aoa, abs(y[0]-y_pred[0]))
    plt.xlabel('Absolute cl Error', **label_font)
    plt.ylabel('angle of attack, deg', **label_font)
    # grid_lines()

    plt.subplot(2, 2, 2)
    # plt.title(f'Drag Comparison, R2={cd_r2:.4f}', **title_font)
    plt.plot(aoa, abs(y[1] - y_pred[1]))
    plt.xlabel('Absolute cd Error', **label_font)
    plt.ylabel('angle of attack, deg', **label_font)
    # grid_lines()

    plt.subplot(2, 2, 3)
    # plt.title(f'Moment Comparison, R2={cm_r2:.4f}', **title_font)
    plt.plot(aoa, abs(y[2] - y_pred[2]))
    plt.xlabel('Absolute cm Error', **label_font)
    plt.ylabel('angle of attack, deg', **label_font)
    # grid_lines()

    plt.subplot(2, 2, 4)
    # plt.title(f'Lift-to-Drag Comparison, R2 ={ld_r2:.4f}', **title_font)
    plt.plot(aoa, abs(y[0]/y[1] - y_pred[0]/y_pred[1]))
    plt.xlabel('Absolute L/D Error', **label_font)
    plt.ylabel('angle of attack, deg', **label_font)
    # grid_lines()

    plt.tight_layout()
    return


def plot_find_error_scatter(y, y_pred, c, label):
    s = 10
    alpha = 0.075
    alpha = 1
    label_font = {'weight': 'bold', 'size': 'large'}
    title_font = {'weight': 'bold', 'size': 'x-large'}

    plt.figure(11, (8, 7.5))
    plt.subplot(2, 2, 1)
    grid_lines()
    # plt.title(f'Lift Comparison, R2={cl_r2:.4f}', **title_font)
    plt.scatter(y[0], y_pred[0], s=s, c=c, alpha=alpha, zorder=2, label=label)
    plt.plot([np.min(y[0]), np.max(y[0])], [np.min(y[0]), np.max(y[0])], 'b--')
    plt.xlabel('CFD C$_{l}$', **label_font)
    plt.ylabel('Predicted C$_{l}$', **label_font)
    plt.legend(framealpha=1)

    plt.subplot(2, 2, 2)
    grid_lines()
    # plt.title(f'Drag Comparison, R2={cd_r2:.4f}', **title_font)
    plt.scatter(y[1], y_pred[1], s=s, c=c, alpha=alpha, zorder=2)
    plt.plot([np.min(y[1]), np.max(y[1])], [np.min(y[1]), np.max(y[1])], 'b--')
    plt.xlabel('CFD C$_{d}$', **label_font)
    plt.ylabel('Predicted C$_{d}$', **label_font)

    plt.subplot(2, 2, 3)
    grid_lines()
    # plt.title(f'Moment Comparison, R2={cm_r2:.4f}', **title_font)
    plt.scatter(y[2], y_pred[2], s=s, c=c, alpha=alpha, zorder=2)
    plt.plot([np.min(y[2]), np.max(y[2])], [np.min(y[2]), np.max(y[2])], 'b--')
    plt.xlabel('CFD C$_{m}$', **label_font)
    plt.ylabel('Predicted C$_{m}$', **label_font)

    plt.subplot(2, 2, 4)
    ld_true = y[0] / (y[1] + 1e-10)
    ld_pred = y_pred[0] / (y_pred[1] + 1e-10)
    grid_lines()
    # plt.title(f'Lift-to-Drag Comparison, R2 ={ld_r2:.4f}', **title_font)
    plt.scatter(ld_true, ld_pred, s=s, c=c, alpha=alpha, zorder=2)
    plt.plot([np.min(ld_true), np.max(ld_true)], [np.min(ld_true), np.max(ld_true)], 'b--')
    plt.xlabel('CFD L/D', **label_font)
    plt.ylabel('Predicted L/D', **label_font)

    plt.tight_layout()
    return


def plot_find_error_stddev(y, y_pred, X_std, n):
    s = 10
    c = 'k'
    alpha = 0.075
    label_font = {'weight': 'bold', 'size': 'large'}
    title_font = {'weight': 'bold', 'size': 'x-large'}

    # Calculate the repeat count for each element in `a`
    new_Xstd = [val for val in X_std.detach().numpy()[:, n] for _ in range(31)]

    plt.figure(n, (10, 9.5))
    plt.subplot(2, 2, 1)
    # plt.title(f'Lift Error Distribution, R2={cl_r2:.4f}', **title_font)
    plt.scatter(new_Xstd, abs(y[0] - y_pred[0]), s=s, c=c, alpha=alpha, zorder=2)
    plt.xlabel(f'Scaled Input a{n}, stdev', **label_font)
    plt.ylabel('cl absolute error', **label_font)

    plt.subplot(2, 2, 2)
    # plt.title(f'Drag Error Distribution, R2={cd_r2:.4f}', **title_font)
    plt.scatter(new_Xstd, abs(y[1] - y_pred[1]), s=s, c=c, alpha=alpha, zorder=2)
    plt.xlabel(f'Scaled Input a{n}, stdev', **label_font)
    plt.ylabel('cd absolute error', **label_font)

    plt.subplot(2, 2, 3)
    # plt.title(f'Moment Error Distribution, R2={cm_r2:.4f}', **title_font)
    plt.scatter(new_Xstd, abs(y[2] - y_pred[2]), s=s, c=c, alpha=alpha, zorder=2)
    plt.xlabel(f'Scaled Input a{n}, stdev', **label_font)
    plt.ylabel('cm absolute error', **label_font)

    plt.subplot(2, 2, 4)
    # plt.title(f'Lift-to-Drag Error Distribution, R2 ={ld_r2:.4f}', **title_font)
    plt.scatter(new_Xstd, abs(y[0] / y[1] - y_pred[0] / y_pred[1]), s=s, c=c, alpha=alpha, zorder=2)
    plt.xlabel(f'Scaled Input a{n}, stdev', **label_font)
    plt.ylabel('L/D absolute error', **label_font)

    plt.tight_layout()
    return


def plot_case(aoa_list, output, y, airfoil, color, marker):
    airfoil = airfoil.split('_')[0]
    # plotting
    label_font = {'weight': 'bold', 'size': 14}
    title_font = {'weight': 'bold', 'size': 16}
    xlabel_name = r'$\alpha$, deg'
    tick_font = 12
    linewidth = 2
    label1 = f'{airfoil} CFD'
    label2 = f'{airfoil} predicted'

    plt.figure(2, figsize=(11, 10))

    # lift curve
    plt.subplot(2, 2, 1)
    grid_lines()
    # plt.figure(1)
    # plt.title('Lift Curve')
    plt.plot(aoa_list, y[0], marker=marker, color=color, markerfacecolor='none', label=label1, ms=10, linewidth=linewidth)
    plt.plot(aoa_list, output[0], linestyle=':', marker=marker, color=color, label=label2, ms=10, linewidth=linewidth)
    plt.legend(fontsize=tick_font, framealpha=1)
    plt.xlim([-5, 21])
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('lift coefficient', **label_font)

    # moment curve
    plt.subplot(2, 2, 2)
    grid_lines()
    # plt.title('Moment Curve')
    # plt.figure(2)
    plt.plot(aoa_list, y[2], marker=marker, color=color, markerfacecolor='none', label=label1, ms=10, linewidth=linewidth)
    plt.plot(aoa_list, output[2], linestyle=':', marker=marker, color=color, label=label2, ms=10, linewidth=linewidth)
    # plt.legend(fontsize=tick_font)
    plt.xlim([-5, 21])
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('moment coefficient', **label_font)

    # drag curve
    plt.subplot(2, 2, 3)
    grid_lines()
    # plt.title('Drag Curve')
    # plt.figure(3)
    plt.plot(aoa_list, y[1], marker=marker, color=color, markerfacecolor='none', label=label1, ms=10, linewidth=linewidth)
    plt.plot(aoa_list, output[1], linestyle=':', marker=marker, color=color, label=label2, ms=10, linewidth=linewidth)
    # plt.legend(fontsize=tick_font)
    plt.xlim([-5, 21])
    # plt.ylim([0, 0.36])
    # plt.xlim([-5, 5])
    # plt.ylim([0.008, 0.05])
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('drag coefficient', **label_font)

    # # drag polar
    # plt.subplot(2, 2, 4)
    # plt.title('Drag Polar')
    # plt.plot(y.T[0], y.T[1], marker, markerfacecolor='none', label=label1, ms=10, linewidth=linewidth)
    # plt.plot(output.T[0], output.T[1], marker, label=label2, ms=10, linewidth=linewidth)
    # plt.legend(fontsize=tick_font)
    # plt.xlabel('lift coefficient')
    # plt.ylabel('drag coefficient')
    # plt.xticks(fontsize=tick_font)
    # plt.yticks(fontsize=tick_font)

    # L/D curve
    plt.subplot(2, 2, 4)
    grid_lines()
    # plt.title('Lift-to-Drag')
    # plt.figure(4)
    plt.plot(aoa_list, y[0]/y[1], marker=marker, color=color, markerfacecolor='none', label=label1, ms=10, linewidth=linewidth)
    plt.plot(aoa_list, output[0]/output[1], linestyle=':', marker=marker, color=color, label=label2, ms=10, linewidth=linewidth)
    # plt.legend(fontsize=tick_font)
    plt.xlim([-5, 21])
    # plt.ylim([-30, 34])
    # plt.xlim([0, 5])
    # plt.ylim([-2, 32])
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('L/D', **label_font)

    # plt.suptitle(f'Airfoil: {airfoil}',  **title_font)
    plt.tight_layout()
    return


def plot_unknown(output):
    # plotting
    aoa_list = np.linspace(-4, 20, 25)
    label_font = {'weight': 'bold', 'size': 14}
    xlabel_name = r'$\alpha$, deg'

    linewidth = 3
    label2 = 'ML Fit'
    dd = 'b:'
    plt.figure(2, figsize=(11, 10))

    # lift curve
    plt.subplot(2, 2, 1)
    grid_lines()
    plt.title('Lift Curve')
    plt.plot(aoa_list, output[0], dd, label=label2, ms=10, linewidth=linewidth)
    plt.legend(fontsize=tick_font)
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('lift coefficient', **label_font)

    # moment curve
    plt.subplot(2, 2, 2)
    grid_lines()
    plt.title('Moment Curve')
    plt.plot(aoa_list, output[2], dd, label=label2, ms=10, linewidth=linewidth)
    plt.legend(fontsize=tick_font)
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('moment coefficient', **label_font)

    # drag curve
    plt.subplot(2, 2, 3)
    grid_lines()
    plt.title('Drag Curve')
    plt.plot(aoa_list, output[1], dd, label=label2, ms=10, linewidth=linewidth)
    plt.legend(fontsize=tick_font)
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('drag coefficient', **label_font)

    # # drag polar
    # plt.subplot(2, 2, 4)
    # plt.title('Drag Polar')
    # plt.plot(output.T[0], output.T[1], dd, label=label2, ms=10, linewidth=linewidth)
    # plt.legend(fontsize=tick_font)
    # plt.xlabel('lift coefficient')
    # plt.ylabel('drag coefficient')
    # plt.xticks(fontsize=tick_font)
    # plt.yticks(fontsize=tick_font)

    # L/D curve
    plt.subplot(2, 2, 4)
    grid_lines()
    plt.title('Lift-to-Drag')
    plt.plot(aoa_list, output[0]/output[1], dd, label=label2, ms=10, linewidth=linewidth)
    plt.legend(fontsize=tick_font)
    plt.xlabel(xlabel_name, **label_font)
    plt.ylabel('L/D', **label_font)

    plt.tight_layout()
    return


def grid_lines():
    tick_font = 12
    plt.minorticks_on()
    plt.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
    plt.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
    plt.xticks(fontsize=tick_font)
    plt.yticks(fontsize=tick_font)
    return
