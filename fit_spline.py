import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # initializing
    deg = 8
    curve_df = df_init(deg)
    correlation_factors = []
    lwr_sv = []
    upr_sv = []
    angles = []
    err_list = []

    # # read airfoil coord
    # airfoil_names = ["GS1", "NACA0012", "NACA0012_64", "NACA0015", "NACA64A015", "RC310", "RC410", "RCSC2",
    #                  "SC1095", "SC2110", "SSCA09", "V23010_158", "V43015_248", "VR12"]
    # # airfoil_names = ["GS1"]
    # for airfoil in airfoil_names:
    #     for k in range(1, 41):
    #         # creating input file name, may change depending on the cases you plan to run
    #         airfoil_name = f'{airfoil}_{k:04d}'
    #         file_path = f'../Transition_MFSM/Raw_Data/Airfoil/{airfoil_name}.txt'
    #
    #         # running fit Spline function
    #         print(f'Running {airfoil_name}')
    #         curve_data, [corr, sv, angle, err] = fit_spline(file_path, deg)
    #
    #         if pd.isna(corr[0]) or pd.isna(corr[1]):
    #             print('FAILED')
    #             return
    #         if corr[0] < 2.75:
    #             print('lower surface poor fit, corr<3')
    #             curve_data, [corr, sv, angle, err] = fit_spline(file_path, deg)
    #         curve_data['airfoil name'] = airfoil_name
    #
    #         # add curve data to the data frame
    #         curve_df.loc[-1] = curve_data
    #         curve_df.index = curve_df.index + 1
    #         curve_df = curve_df.sort_index()
    #
    #         # correlation factor and singular value
    #         correlation_factors.append(corr)
    #         lwr_sv.append(sv[0])
    #         upr_sv.append(sv[1])
    #         angles.append(angle)
    #         err_list.append(err)
    #
    # print(min(correlation_factors))

    curve_df = pd.read_excel('Datasets/M0_7-Re4_67e+06-clcdcm.xlsx')
    rename_dict = {}
    for i in range(1, 10):
        rename_dict[f'a{i}'] = f'lwr_c{i - 1}'
    for i in range(10, 19):
        rename_dict[f'a{i}'] = f'upr_c{i - 10}'
    curve_df = curve_df.rename(columns=rename_dict)

    # # write all curve data into
    # output_path = 'Model/curve.csv'
    # if os.path.isfile(output_path):
    #     os.remove(output_path)
    # curve_df.to_csv(output_path)

    # Explore other airfoil parameterization
    file_path = f'Raw_Data/NACA0009.txt'
    file_path = f'Raw_Data/OA212.txt'
    # file_path = f'Raw_Data/NACA63215.txt'
    # file_path = '../Transition_MFSM/Raw_Data/Airfoil/clarky_flipped.txt'
    curve_data, [corr, sv, angle, err] = fit_spline(file_path, 0.5, 1.0, deg)
    if corr[0] < 2.75:
        print('lower surface poor fit, corr<3')
        curve_data, [corr, sv, angle, err] = fit_spline(file_path, 0.5, 1.0, deg)

    print(curve_data)
    print([curve_data.tolist()])

    # plot cst shape coefficients parameter space
    plt.figure(1, figsize=(25, 14))
    for i in range(deg+1):
        plt.subplot(3, 6, i+1)
        cnt, _, __ = plt.hist(curve_df[f'lwr_c{i}'], bins=20)
        plt.vlines(curve_data[f'lwr_c{i}'], 0, np.max(cnt)+5, 'r', linewidth=3)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.title(f'Lower C{i}')

        plt.subplot(3, 6, 10+i)
        cnt, _, __ = plt.hist(curve_df[f'upr_c{i}'], bins=20)
        plt.vlines(curve_data[f'upr_c{i}'], 0, np.max(cnt)+5, 'r', linewidth=3)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.title(f'Upper C{i}')
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.hist(np.array(correlation_factors), bins=50, stacked=True)
    plt.xlim([0, np.array(correlation_factors).max()])
    plt.legend(['lower surface', 'upper surface'])
    plt.xlabel('correlation factor, -log(1-r2)')
    plt.ylabel('n cases')
    plt.show()

    err_list = np.array(err_list)
    plt.hist(err_list, bins=50, stacked=True)
    plt.axvline(x=0.0004, linestyle='--')
    plt.xlim([0, err_list.max()])
    plt.legend(['Typical Wind Tunnel Model Tolerance', 'lower surface', 'upper surface'])
    plt.title('Airfoil Fit')
    plt.xlabel('Mean Absolute Error, Chord')
    plt.ylabel('n cases')
    plt.grid()
    plt.show()

    # lwr_sv = np.array(lwr_sv)
    # upr_sv = np.array(upr_sv)
    # label = [f'c{i}' for i in range(lwr_sv.shape[1])]
    # plt.subplot(1, 2, 1)
    # plt.boxplot(lwr_sv, labels=label)
    # plt.title('lower surface polynomial singular values')
    #
    # plt.subplot(1, 2, 2)
    # plt.boxplot(upr_sv, labels=label)
    # plt.title('upper surface polynomial singular values')
    # plt.show()
    return


def df_init(deg):
    curve_df = pd.DataFrame(columns=['airfoil name'])
    for i in range(deg+1):
        curve_df[f'lwr_c{i}'] = pd.Series(dtype=float)
    # curve_df['lwr_zeta'] = pd.Series(dtype=float)
    for i in range(deg+1):
        curve_df[f'upr_c{i}'] = pd.Series(dtype=float)
    # curve_df['upr_zeta'] = pd.Series(dtype=float)
    curve_df['zeta'] = pd.Series(dtype=float)
    return curve_df


def one_off(file_path='Datasets/Airfoil/RAE2822.txt'):
    print(f'Running {file_path}')
    curve_data, fit_data = fit_spline(file_path, 0.5, 1.0, 8, True, False)
    [[lwr_corr_factor, upr_corr_factor], [lwr_sv, upr_sv], angle, [lwr_error, upr_error]] = fit_data

    print(curve_data)
    print(lwr_corr_factor, upr_corr_factor)
    print(lwr_error, upr_error)
    return


def fit_spline(file_path, N1=0.5, N2=1.0, deg=8, plot=False, overwrite=False):
    angle = 0
    raw_df = pd.read_csv(file_path, header=None, sep='\s+', names=['x/c', 'y/c'])
    df = raw_df.copy()

    # split df into upper and lower surface
    split = df['x/c'].idxmin()
    if df.iloc[split]['y/c'] != 0:
        # translate the leading edge to (0,0) and rotate the airfoil such that LE&TE are on y/c axis
        df, rot = __derotate_airfoil(df, split)
        angle = rot[2]*180/np.pi
        if angle > 180:
            angle -= 360
        # overwrite airfoil coordinate file
        if overwrite:
            print(f'write de-rotated airfoil')
            df.to_csv(file_path, index=False, header=False, sep=' ')

    lwr_curve = df.iloc[:split+1].iloc[::-1]
    lwr_curve.reset_index(drop=True, inplace=True)
    upr_curve = df.iloc[split:]
    upr_curve.reset_index(drop=True, inplace=True)

    # shift and scale to fit in range [0,1]
    lwr_curve, xmin_l, xmax_l = __scale_curve(lwr_curve)
    upr_curve, xmin_u, xmax_u = __scale_curve(upr_curve)

    # convert from airfoil curve to a shape function (assume a class function with N1:0.5 N2:1)
    lwr_shape, lwr_zeta = __get_shape(lwr_curve, N1, N2)
    upr_shape, upr_zeta = __get_shape(upr_curve, N1, N2)

    # get chebyshev polynomials
    lwr_c_poly, lwr_fit = __fit_chebyshev(lwr_shape, deg)
    upr_c_poly, upr_fit = __fit_chebyshev(upr_shape, deg)

    # # shift and descale to fit back to the original curve
    # lwr_curve = __descale_curve(lwr_curve, dx_l, dc_l)
    # upr_curve = __descale_curve(upr_curve, dx_l, dc_l)

    # evaluate the fit of chebyshev curve using correlation factor
    lwr_cheby_curve = get_spline(lwr_c_poly, lwr_zeta, lwr_curve['x/c'], N1, N2)
    upr_cheby_curve = get_spline(upr_c_poly, upr_zeta, upr_curve['x/c'], N1, N2)
    # lwr_cheby_curve = __descale_curve(lwr_cheby_curve, dx_l, dc_l)
    # upr_cheby_curve = __descale_curve(upr_cheby_curve, dx_l, dc_l)

    lwr_corr_factor = eval_spline(lwr_curve, lwr_cheby_curve)
    upr_corr_factor = eval_spline(upr_curve, upr_cheby_curve)
    # print(f'lower curve correlation factor: {lwr_corr_factor}')
    # print(f'upper curve correlation factor: {upr_corr_factor}')

    # evaluate the chebyshev polynomial using singular value  % wrong???
    lwr_sv = lwr_fit[2]
    upr_sv = upr_fit[2]

    # find mean square error of the fit
    lwr_error = np.mean(np.abs(lwr_curve['y/c']-lwr_cheby_curve['y/c']))
    upr_error = np.mean(np.abs(upr_curve['y/c']-upr_cheby_curve['y/c']))

    if plot:
        font = {'weight': 'bold', 'size': 'large'}

        # plot it
        plt.figure(figsize=(12, 3))
        plt.plot(lwr_cheby_curve['x/c'], lwr_cheby_curve['y/c'], 'ro', markersize=4, label='lower fitted curve')
        plt.plot(upr_cheby_curve['x/c'], upr_cheby_curve['y/c'], 'go', markersize=4, label='upper fitted curve')
        plt.plot(df['x/c'], df['y/c'], 'k', linewidth=2, label='original curve', zorder=3)
        plt.minorticks_on()
        plt.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
        plt.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
        plt.xlabel('x/c', **font)
        plt.ylabel('y/c', **font)
        # plt.xlim([-0.01, 0.05])
        # plt.ylim([-0.05, 0.05])
        # plt.axis('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 3))
        plt.minorticks_on()
        plt.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
        plt.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
        plt.plot(lwr_curve['x/c'][1::], (lwr_curve['y/c'][1::]-lwr_cheby_curve['y/c'][1::]), 'r', linewidth=2, label='lower surface fit')
        plt.plot(upr_curve['x/c'][1::], (upr_curve['y/c'][1::]-upr_cheby_curve['y/c'][1::]), 'g', linewidth=2, label='upper surface fit')
        # plt.title(f'Airfoil Fit')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
        plt.xlabel('x/c', **font)
        plt.ylabel('Residual, chord length', **font)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # construct the output series
    curve_data = pd.Series()
    for i in range(deg + 1):
        curve_data[f'lwr_c{i}'] = lwr_c_poly[i]
    # curve_data['lwr_zeta'] = lwr_zeta
    for i in range(deg+1):
        curve_data[f'upr_c{i}'] = upr_c_poly[i]
    # curve_data['upr_zeta'] = upr_zeta
    curve_data['zeta'] = (upr_zeta - lwr_zeta)/2

    return curve_data, [[lwr_corr_factor, upr_corr_factor], [lwr_sv, upr_sv], angle, [lwr_error, upr_error]]


# ============================ CHEBYSHEV POLYNOMIAL ============================
def __derotate_airfoil(df, split):
    # find leading edge and trailing edge coords
    end_i = len(df)-1
    le_x, le_y = df.loc[split, 'x/c'], df.loc[split, 'y/c']
    te_x = (df.loc[end_i, 'x/c']+df.loc[0, 'x/c'])/2
    te_y = (df.loc[end_i, 'y/c']+df.loc[0, 'y/c'])/2

    # translate airfoil coord s.t. LE is at (0,0)
    df.loc[:, 'x/c'] -= le_x
    df.loc[:, 'y/c'] -= le_y

    # rotate airfoil coord s.t. TE(y/c) is 0
    angle = np.pi-np.arctan2(le_y-te_y, le_x-te_x)
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
    rotated_df = np.dot(df[['x/c', 'y/c']], rotation_matrix)
    rotated_df = pd.DataFrame(rotated_df, columns=['x/c', 'y/c'])
    return rotated_df, [le_x, le_y, angle]


def __scale_curve(df):
    # scale x/c s.t. it ranges (0,1)
    x_min = df['x/c'].min()
    x_max = df['x/c'].max()
    df.loc[:, 'x/c'] -= x_min
    df.loc[:, 'x/c'] /= (x_max-x_min)
    return df, x_min, x_max


def __get_shape(df, N1=0.5, N2=1.0):
    # preprocess the point curve to fit shape function
    zeta = df['y/c'].iloc[-1]
    class_func = df['x/c']**N1 * (1 - df['x/c'])**N2
    df['S'] = (df['y/c'] - df['x/c'] * zeta) / class_func
    return df, zeta


def __fit_chebyshev(df, degree):
    # trim to eliminate the edge points
    df = df[1:-2]
    df = df[df['x/c'] >= 0.01]
    df = df[df['x/c'] <= 0.99]

    # stretch x from (0,1) to (-1,1) for chebyshev fit
    x_fit = 2 * df['x/c'] - 1
    y_fit = df['S']

    # find chebyshev polynomials using the least square fit
    chebyshev_polynomials, fit_data = np.polynomial.chebyshev.chebfit(x_fit, y_fit, degree, full=True)
    # fit_data = [residuals, rank, singular_values, rcond]
    return chebyshev_polynomials, fit_data


# def __fit_bernstein(point_curve, degree):
#     return bernstein_polynomials, zeta, point_curve, fit_data


def __descale_curve(curve, dc):
    curve.loc[:, 'x/c'] *= dc
    return curve


def get_spline(c_poly, zeta, n_points, N1=0.5, N2=1.0):
    if isinstance(n_points, int):
        # generate output curve with cosine distribution
        xc = (-np.cos(np.linspace(0, np.pi, n_points)) + 1)/2
        xc = xc.tolist()
    else:
        xc = n_points
    cheby_shape = []
    yc = []

    # find y/c for each x/c
    for x in xc:
        class_func = x**N1 * (1 - x)**N2
        shape_func = 0
        # adding up n degrees of polynomials https://en.wikipedia.org/wiki/Chebyshev_polynomials
        # note: there are better way to get Tn, this is just easier for me
        for i, coeff in enumerate(c_poly):
            shape_func += coeff * np.cos(i * np.arccos(2*x-1))

        cheby_shape.append(shape_func)
        yc.append(shape_func * class_func + x * zeta)

    return pd.Series({'x/c': xc, 'y/c': yc, 'S': cheby_shape})


def get_fast_spline(c_poly, zeta, n_points, N1=0.5, N2=1.0):
    if isinstance(n_points, int):
        # Cosine-spaced x/c
        xc = (-np.cos(np.linspace(0, np.pi, n_points)) + 1) / 2
    else:
        xc = np.asarray(n_points)

    # Evaluate Chebyshev polynomials at 2x-1
    tx = 2 * xc - 1
    T = np.polynomial.chebyshev.chebvander(tx, len(c_poly) - 1)  # (N_points, degree)

    # Shape function at all points
    shape_func = T @ c_poly

    # Class function
    class_func = xc**N1 * (1 - xc)**N2

    # yc
    yc = shape_func * class_func + xc * zeta

    return xc, yc


def eval_spline(original_curve, fitted_curve):
    true, pred = np.array(original_curve['y/c']), np.array(fitted_curve['y/c'])
    num = ((true - pred) ** 2).sum()
    den = ((true - np.average(true)) ** 2).sum()
    correlation_factor = -np.log10(num/den)  # correlation factor = -np.log(1-r2); r2 = 1-num/den
    return correlation_factor


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None  # default='warn'
    path = '../Transition_MFSM/Raw_Data/Airfoil/NACA64A015_0004.txt'
    one_off(path)
    # main()
