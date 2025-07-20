import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from fit_spline import *
import train_aerodynamics_prediction_model as apm
import predict_shock as ps


# ---------------------
# Airfoil generator (example)
# ---------------------
def generate_airfoil(lower_coef, upper_coef, te_thickness):
    x, y_lower = get_fast_spline(lower_coef, -te_thickness, 200)
    x, y_upper = get_fast_spline(upper_coef, te_thickness, 200)

    return x, y_upper, y_lower


# ---------------------
# GUI
# ---------------------
root = tk.Tk()
root.title("Airfoil Characteristics Surrogate Model at Transonic Speed")
# root.tk.call('source', '../TK_Azure/azure.tcl')
# root.tk.call('set_theme', 'light')

# Get surrogate models
apm_mod_folder = 'Model/APM'
classifier_folder = 'Model/classifier_TSPM'
mod_name = 'regressor_TSPM'
lwr_mod_folder = f'Model/lwr_{mod_name}'
upr_mod_folder = f'Model/upr_{mod_name}'

model, xscaler, yscaler, pca = apm.get_model(apm_mod_folder)
class_mod, class_xsc = ps.get_model(classifier_folder, 'classifier')
lwr_reg_mod, lwr_reg_xsc = ps.get_model(lwr_mod_folder, 'regressor')
upr_reg_mod, upr_reg_xsc = ps.get_model(upr_mod_folder, 'regressor')

# Initialize data
lower_coeffs = [0.0] * 9
upper_coeffs = [0.0] * 9
te_thickness = [0.0]

all_vars = lower_coeffs + upper_coeffs + te_thickness
slider_vars = []
entry_vars = []

# Matplotlib figure
# fig, axs = plt.subplots(2, 3, figsize=(17, 11))
fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
label_font = {'weight': 'bold', 'size': 14}
title_font = {'weight': 'bold', 'size': 16}
tick_font = 12
linewidth = 3

title_list = ['Airfoil Shape', 'Lift Coefficient', 'Moment Coefficient', 'Shock Locations', 'Drag Coefficient', 'Lift-to-Drag']
xlabel_list = ['x/c', r'$\alpha$, deg', r'$\alpha$, deg', r'$\alpha$, deg', r'$\alpha$, deg', r'$\alpha$, deg']
ylabel_list = ['y/c', 'lift coefficient', 'moment coefficient', r'shock location, $hat{r}$', 'drag coefficient', 'lift-to-drag']
for i, ax in enumerate(axs.flatten()):
    ax.set_title(title_list[i], **title_font)
    ax.set_xlabel(xlabel_list[i], **label_font)
    ax.set_ylabel(ylabel_list[i], **label_font)
    ax.minorticks_on()
    ax.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
    ax.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
    ax.tick_params(labelsize=tick_font)

plt.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
# canvas_widget.grid(row=0, column=3, rowspan=10)
canvas_widget.grid(row=0, column=3)


def update_plot():
    # Running aerodynamics surrogate model
    output = apm.run_designed_airfoil(all_vars, model, xscaler, yscaler, pca)

    # Running transonic shock surrogate model
    shock_loc = ps.run_design_airfoil(all_vars, class_mod, class_xsc, lwr_reg_mod, lwr_reg_xsc, upr_reg_mod, upr_reg_xsc)

    # airfoil spline reconstruction
    x, y_u, y_l = generate_airfoil(all_vars[:9], all_vars[9:18], all_vars[18])

    aoa_list = np.linspace(-4, 20, 25)
    x_list = [[], aoa_list, aoa_list, shock_loc[0], aoa_list, aoa_list]
    y_list = [[], output[0], output[2], shock_loc[1], output[1], output[0]/output[1]]

    for i, ax in enumerate(axs.flatten()):
        ax.clear()
        if i == 3:
            ax.scatter(x_list[i], y_list[i], c='b', zorder=4)
        else:
            ax.plot(x_list[i], y_list[i], linewidth=linewidth, zorder=4)

        ax.set_title(title_list[i], **title_font)
        ax.set_xlabel(xlabel_list[i], **label_font)
        ax.set_ylabel(ylabel_list[i], **label_font)
        ax.minorticks_on()
        ax.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
        ax.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
        ax.tick_params(labelsize=tick_font)

    axs[0, 0].plot(x, y_l, 'r', linewidth=linewidth, label='Lower Surface')
    axs[0, 0].plot(x, y_u, 'g', linewidth=linewidth, label='Upper Surface')
    axs[0, 0].legend()
    plt.tight_layout()
    canvas.draw()


# ---------------------
# Helper to link slider and textbox
# ---------------------
def make_slider_and_entry(parent, label, row, var_index, start, end):
    padx = 5
    pady = 5

    lbl = tk.Label(parent, text=label)
    lbl.grid(row=row, column=0, sticky="e", padx=padx, pady=pady)

    entry_var = tk.StringVar(value=f"{all_vars[var_index]:.5f}")

    # Create the slider first
    slider = ttk.Scale(parent, from_=start, to=end, orient="horizontal", length=300)
    slider.set(all_vars[var_index])
    slider.grid(row=row, column=1, sticky="we", padx=padx, pady=pady)

    # Create the entry
    entry_font = ("Arial", 14)
    entry = tk.Entry(parent, textvariable=entry_var, width=8, font=entry_font)
    entry.grid(row=row, column=2, padx=padx, pady=pady)

    # Append slider and entry input to a global list
    slider_vars.append(slider)
    entry_vars.append(entry_var)

    # Define callbacks after both exist
    def on_slider_move(val):
        try:
            val_float = float(val)
            all_vars[var_index] = val_float
            entry_var.set(f"{val_float:.5f}")
        except ValueError:
            pass

    def on_slider_release(event):
        update_plot()

    def on_entry_change(*args):
        try:
            val = float(entry_var.get())
            all_vars[var_index] = val
            slider.set(val)

            # update airfoil shape continuously, a little bit laggy but manageable
            x, y_u, y_l = generate_airfoil(all_vars[:9], all_vars[9:18], all_vars[18])
            ax = axs[0, 0]
            ax.clear()
            ax.plot(x, y_l, 'r', linewidth=linewidth, label='Lower Surface')
            ax.plot(x, y_u, 'g', linewidth=linewidth, label='Upper Surface')
            ax.legend()
            ax.set_title(title_list[0], **title_font)
            ax.set_xlabel(xlabel_list[0], **label_font)
            ax.set_ylabel(ylabel_list[0], **label_font)
            ax.minorticks_on()
            ax.grid(which='major', linewidth=0.8, color='dimgray', zorder=1)
            ax.grid(which='minor', linewidth=0.6, color='darkgray', zorder=1)
            ax.tick_params(labelsize=tick_font)
            canvas.draw()
        except ValueError:
            pass

    def on_entry_enter(event):
        update_plot()

    # Attach the callbacks
    slider.configure(command=on_slider_move)
    slider.bind("<ButtonRelease-1>", on_slider_release)
    entry_var.trace_add("write", on_entry_change)
    entry.bind("<Return>", on_entry_enter)


def update_sliders_and_entries(new_values):
    if len(new_values) != 19:
        print("Error: Expected 19 coefficients!")
        return
    for i in range(19):
        all_vars[i] = new_values[i]
        slider_vars[i].set(new_values[i])
        entry_vars[i].set(f"{new_values[i]:.4f}")


def load_file():
    filepath = filedialog.askopenfilename(
        title="Select Airfoil Coordinate File",
        filetypes=(("Text Files", "*.txt *.dat"), ("All Files", "*.*"))
    )
    if not filepath:
        return

    try:
        # Fit airfoil coordinates
        print(f"Loaded file: {filepath}")
        curve_data, _ = fit_spline(filepath)

        # Update GUI
        update_sliders_and_entries(curve_data.to_list())
        update_plot()

        # Plot in existing figure
        ax = axs[0, 0]
        data = np.loadtxt(filepath)
        ax.plot(data[:, 0], data[:, 1], 'k--', label='Loaded Airfoil', zorder=4)
        ax.legend()
        canvas.draw()

    except Exception as e:
        print(f"Error loading file: {e}")


# ---------------------
# Frame for control panel
# ---------------------
slider_frame = tk.Frame(root)
slider_frame.grid(row=0, column=0)

# Upper Coefficients
tk.Label(slider_frame, text="Lower Surface Coefficients", font=("Arial", 10, "bold")).grid(row=0, column=1, pady=10)
for i in range(9):
    make_slider_and_entry(slider_frame, f"Upper c{i}", i + 1, i, -0.5, 0.5)

# Lower Coefficients
start_row = 10
tk.Label(slider_frame).grid(row=start_row, column=1)
tk.Label(slider_frame, text="Upper Surface Coefficients", font=("Arial", 10, "bold")).grid(row=start_row+1, column=1, pady=10)
for i in range(9):
    make_slider_and_entry(slider_frame, f"Lower c{i}", start_row + 2 + i, 9 + i, -0.5, 0.5)

# TE Thickness
tk.Label(slider_frame).grid(row=2*start_row + 1, column=1)
make_slider_and_entry(slider_frame, "TE Thickness", 2*start_row + 2, 18, 0, 0.01)
tk.Label(slider_frame).grid(row=2*start_row + 3, column=1)

# Load airfoil coordinates
load_button = tk.Button(slider_frame, text="Load Airfoil Coordinates", command=load_file)
load_button.grid(row=2*start_row + 4, column=0, columnspan=3, pady=10)

root.mainloop()
