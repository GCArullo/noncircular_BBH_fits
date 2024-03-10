import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt, pandas as pd, os, seaborn as sns

from Utils import template

if not os.path.exists('Plots'): os.mkdir('Plots')

def init_plotting():

    """
    
    Function to set the default plotting parameters.

    Parameters
    ----------
    None

    Returns
    -------
    Nothing, but sets the default plotting parameters.
    
    """
    
    plt.rcParams['figure.max_open_warning'] = 0
    
    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    plt.rcParams['font.size']         = 14
    plt.rcParams['axes.linewidth']    = 1
    plt.rcParams['axes.labelsize']    = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize']    = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize']   = plt.rcParams['font.size']
    plt.rcParams['xtick.major.size']  = 3
    plt.rcParams['xtick.minor.size']  = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size']  = 3
    plt.rcParams['ytick.minor.size']  = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    
    plt.rcParams['legend.frameon']             = False
    plt.rcParams['legend.loc']                 = 'center left'
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    return

def label_mapping(param, qc_norm=0):

    if(  param=='A_peak22'    ): 
        if(qc_norm): label = r'$A_{\mathrm{mrg}} \, / \, A_{\mathrm{mrg}}^{qc}$'
        else       : label = r'$A_{\mathrm{mrg}}$'
    elif(param=='omega_peak22'): 
        if(qc_norm): label = r'$\omega_{\mathrm{mrg}} \, / \, \omega_{\mathrm{mrg}}^{qc}$'
        else       : label = r'$\omega_{\mathrm{mrg}}$'
    elif(param=='Mf'          ): 
        if(qc_norm): label = r'$M_f \, / \, M_f^{qc}$'
        else       : label = r'$M_f$'
    elif(param=='af'          ): 
        if(qc_norm): label = r'$a_f \, / \, a_f^{qc}$'
        else       : label = r'$a_f$'
    elif(param=='b_massless'     ): label = r'$b^{E}_{\mathrm{mrg}}$'
    elif(param=='b_massless_Heff'): label = r'$b_{\mathrm{mrg}}$'
    elif(param=='b_massless_EOB' ): label = r'$\hat{b}_{\mathrm{mrg}}$'
    elif(param=='Emrg_til'       ): label = r'$\tilde{E}_{\rm mrg}$'
    elif(param=='Heff_til'       ): label = r'$\hat{E}_{\mathrm{eff}}^{\rm mrg}$'
    elif(param=='Jmrg_til'       ): label = r'$j_{\rm mrg}$'
    elif(param=='nu'             ): label = r'$\nu$'
    elif(param=='ecc'            ): label = r'$e_0$'  
    elif(param=='chieff'         ): label = r'$\chi_{\rm eff}$'     
    else                          : label = param

    return label

def plot_residuals_histogram(fitting_quantities_dict, quantity_to_fit, data, template_model, fit_dim, coeffs, dataset_type, catalogs_string, fitting_quantities_string):

    # Plot a single histogram with the fit residuals

    init_plotting()

    fontsize_labels = 22
    label_pad_size  = 12

    fitting_model_data = template(coeffs, fitting_quantities_dict, template_model)
    residuals          = 100*(fitting_model_data - data[quantity_to_fit])/data[quantity_to_fit]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.histplot(residuals, alpha=0.5, ax=ax)
    ax.set_xlabel(r'Residuals (' + label_mapping(quantity_to_fit, qc_norm=1) + ') [%]', fontsize=fontsize_labels, labelpad=label_pad_size)
    ax.set_ylabel(r'$ Counts $' , fontsize=fontsize_labels, labelpad=label_pad_size)
    ax.grid(True, linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'Plots/Residuals_{fitting_quantities_string}_{quantity_to_fit}_{dataset_type}_{catalogs_string}_{template_model}_{fit_dim}.pdf', bbox_inches='tight')

    return

def plot_3D_fit(fitting_quantities_dict, quantity_to_fit, dataframe, template_model, fit_dim, coeffs, dataset_type, catalogs_string):

    init_plotting()

    markers_size    = 14
    cmap            = mpl.cm.inferno
    fontsize_labels = 22
    label_pad_size  = 12

    fitting_quantity_name_1 = list(fitting_quantities_dict.keys())[0]
    fitting_quantity_name_2 = list(fitting_quantities_dict.keys())[1]

    fitting_quantity1_grid                         = np.linspace(np.min(dataframe[fitting_quantity_name_1]), np.max(dataframe[fitting_quantity_name_1]), 100)
    fitting_quantity2_grid                         = np.linspace(np.min(dataframe[fitting_quantity_name_2]), np.max(dataframe[fitting_quantity_name_2]), 100)
    fitting_quantity1_grid, fitting_quantity2_grid = np.meshgrid(fitting_quantity1_grid, fitting_quantity2_grid)
    fitting_quantities_grid_dict                   = {fitting_quantity_name_1: fitting_quantity1_grid, fitting_quantity_name_2: fitting_quantity2_grid}
    
    fitting_model_grid      = template(coeffs, fitting_quantities_grid_dict, template_model)
    fitting_model_data      = template(coeffs, fitting_quantities_dict     , template_model)
    dataframe['residuals']  = 100*(fitting_model_data - dataframe[quantity_to_fit])/dataframe[quantity_to_fit]

    size_increase = 1.1

    fig_tmp = plt.figure(figsize=(9*size_increase,7*size_increase))
    ax_tmp  = fig_tmp.add_subplot(111, projection='3d')
    p = ax_tmp.scatter(dataframe[fitting_quantity_name_1], dataframe[fitting_quantity_name_2], dataframe[quantity_to_fit], c=dataframe['residuals'], s=markers_size*2.0, marker = '.', cmap=cmap, zorder=-1)
    plt.close()

    fig = plt.figure(figsize=(9*size_increase,7*size_increase))
    ax  = fig.add_subplot(111, projection='3d')

    # White box
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    stride = 15

    ax.plot_surface(fitting_quantity1_grid, fitting_quantity2_grid, fitting_model_grid, 
                    alpha=0.3, 
                    zorder=0, 
                    color='dodgerblue',
                    linewidth=0.8, 
                    antialiased=False, 
                    rstride=stride, cstride=stride)

    # Loop over catalogs and corresponding markers
    for catalog, marker in zip(['RIT', 'SXS', 'ET', 'RWZ'], ['.', 'x', 's', '^']):

        dataframe_x = dataframe[dataframe['catalog']==catalog]
        if(marker=='.'): size_increase_marker = 7.0
        else           : size_increase_marker = 2.0
        ax.scatter(dataframe_x[fitting_quantity_name_1], dataframe_x[fitting_quantity_name_2], dataframe_x[quantity_to_fit], c=dataframe_x['residuals'], s=markers_size*size_increase_marker, marker = marker, cmap=cmap, zorder=-1)

    ax.set_xlabel(label_mapping(fitting_quantity_name_1   ), fontsize=fontsize_labels, labelpad=label_pad_size)
    ax.set_ylabel(label_mapping(fitting_quantity_name_2   ), fontsize=fontsize_labels, labelpad=label_pad_size)
    ax.set_zlabel(label_mapping(quantity_to_fit, qc_norm=1), fontsize=fontsize_labels, labelpad=label_pad_size, rotation=90)

    plt.rcParams['mathtext.fontset']  = 'stix'
    plt.rcParams['font.family']       = 'STIXGeneral'

    cb1 = fig.colorbar(p, orientation='horizontal',fraction=0.038, pad=0.0)
    cb1.set_label(label = r'$\mathrm{Residuals} \,( \% )$', fontsize=int(fontsize_labels*0.8))

    if(dataset_type=='aligned-spins-equal-mass'): 

        if(  quantity_to_fit=='A_peak22'    ): ax.view_init(azim=-111, elev=17)
        elif(quantity_to_fit=='omega_peak22'): ax.view_init(azim=-115, elev=17)
        elif(quantity_to_fit=='Mf'          ): ax.view_init(azim= -62, elev=20)
        elif(quantity_to_fit=='af'          ): ax.view_init(azim= -71, elev=19)

    else:
        if(fitting_quantity_name_1=='nu' or fitting_quantity_name_2=='nu'): 

            if(  quantity_to_fit=='A_peak22'    ): ax.view_init(azim=-139, elev=19)
            elif(quantity_to_fit=='omega_peak22'): ax.view_init(azim=-146, elev=22)
            elif(quantity_to_fit=='Mf'          ): ax.view_init(azim=-145, elev=19)
            elif(quantity_to_fit=='af'          ): ax.view_init(azim=-145, elev=19)

        else:

            if(  quantity_to_fit=='A_peak22'    ): ax.view_init(azim=-36,  elev=20)
            elif(quantity_to_fit=='omega_peak22'): ax.view_init(azim=-40,  elev=22)
            elif(quantity_to_fit=='Mf'          ): ax.view_init(azim=-161, elev=15)
            elif(quantity_to_fit=='af'          ): ax.view_init(azim=-158, elev=15)

    plt.savefig(f'Plots/{fitting_quantity_name_1}_{fitting_quantity_name_2}_{quantity_to_fit}_{dataset_type}_{catalogs_string}_{template_model}_{fit_dim}.pdf', bbox_inches='tight')

    return

def plot_2D_fit(fitting_quantities_dict, quantity_to_fit, dataframe, template_model, fit_dim, coeffs, dataset_type, catalogs_string):

    # Plot parameters
    lw_fit      = 1.6
    size_labels = 24
    ls_fit      = 'dashed'
    color_fit   = 'darkred'
    palette_fig = 'rocket_r' #'mako_r'

    # Build a finer grid of data to plot the fit
    fitting_quantity_name      = list(fitting_quantities_dict.keys())[0]
    fitting_quantity_grid_dict = {fitting_quantity_name: np.linspace(np.min(dataframe[fitting_quantity_name]), np.max(dataframe[fitting_quantity_name]), 100)}
    fitting_model_grid         = template(coeffs, fitting_quantity_grid_dict, template_model)
    fitting_model_data         = template(coeffs, fitting_quantities_dict   , template_model)
    dataframe['residuals']     = 100*(fitting_model_data - dataframe[quantity_to_fit])/dataframe[quantity_to_fit]

    size_small = 80
    size_big   = 100

    # Initialise the figure 
    init_plotting()
    myfig = plt.figure(figsize=(9*1.1,7*1.1))
    ax1   = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax2   = plt.subplot2grid((4, 1), (3, 0), rowspan=1)

    if(quantity_to_fit=='A_peak22'): legend_flag = True
    else:                            legend_flag = False

    # Plot the data and the fit
    sns.scatterplot(data=dataframe, x=fitting_quantity_name, y=quantity_to_fit, hue="ecc", style="catalog", style_order=['RIT', 'SXS', 'ET'], size="catalog", sizes = {"RIT": size_small, "SXS": size_small, "ET": size_big}, palette=palette_fig, legend=legend_flag, ax=ax1)
    ax1.plot(fitting_quantity_grid_dict[fitting_quantity_name], fitting_model_grid, c=color_fit, ls=ls_fit, lw=lw_fit, zorder=-1)
    ax1.grid(alpha=0.2)

    ax1.set_ylabel(label_mapping(quantity_to_fit, qc_norm=1), fontsize=size_labels)
    # Plot residuals at the bottom of the plot
    sns.scatterplot(data=dataframe, x=fitting_quantity_name, y='residuals',     hue="ecc", style="catalog", style_order=['RIT', 'SXS', 'ET'], size="catalog", sizes = {"RIT": size_small, "SXS": size_small, "ET": size_big}, palette=palette_fig, legend=False, ax=ax2)
    ax2.grid(alpha=0.2)
    ax2.set_ylabel(r'Res. [%]', fontsize=size_labels)
    ax2.set_xlabel(label_mapping(fitting_quantity_name), fontsize=size_labels)
    # Finalise the plot
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax1.set_xticklabels([])

    if not('aligned-spins' in dataset_type):

        if('b_massless' in fitting_quantity_name):
            # Paper plot lims for right side of Fig.1
            ax1.set_xlim([1.78,3.47])
            if(  quantity_to_fit=='A_peak22'    ): 
                ax1.set_ylim([0.47,1.28])
                ax2.set_ylim([-5, 5])
            elif(quantity_to_fit=='omega_peak22'): 
                ax1.set_ylim([0.5,1.2 ])
                ax2.set_ylim([-14, 14])

        elif(fitting_quantity_name=='Heff_til'):
            ax1.set_xlim([0.86,0.973])
            if(quantity_to_fit=='Mf'): 
                ax1.set_ylim([0.984,1.05])
                ax2.set_ylim([-1.5, 1.5])
        elif(fitting_quantity_name=='Jmrg_til'):
            ax1.set_xlim([1.72,3.25])
            if(quantity_to_fit=='af'): 
                ax1.set_ylim([0.6,1.15])
                ax2.set_yticks(ax2.get_yticks()[:-1])
                ax2.set_ylim([-5, 5])     

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'Plots/{fitting_quantity_name}_{quantity_to_fit}_{dataset_type}_{catalogs_string}_{template_model}_{fit_dim}.pdf', bbox_inches='tight')

    return

def plot_results(fitting_quantities_dict, quantity_to_fit, data, template_model, fit_dim, coeffs, dataset_type, catalogs_string, fitting_quantities_string):
    
    if(  len(fitting_quantities_dict.keys())==1): plot_2D_fit(fitting_quantities_dict, quantity_to_fit, data, template_model, fit_dim, coeffs, dataset_type, catalogs_string)
    elif(len(fitting_quantities_dict.keys())==2): plot_3D_fit(fitting_quantities_dict, quantity_to_fit, data, template_model, fit_dim, coeffs, dataset_type, catalogs_string)
    plot_residuals_histogram(fitting_quantities_dict, quantity_to_fit, data, template_model, fit_dim, coeffs, dataset_type, catalogs_string, fitting_quantities_string)

    return