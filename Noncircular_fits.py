import matplotlib.pyplot as plt

# Package-specific imports
from Utils import create_catalog_string, load_data, read_fit_coefficients, select_catalogs, select_fitting_quantities, select_template_model
from Plots import plot_results

########################
# Start of user inputs #
########################

# Dataset and quantities to fit 
quantities_to_fit    = ['A_peak22', 'omega_peak22', 'Mf', 'af']    
dataset_types        = ['non-spinning-equal-mass'] # Available options: ['non-spinning-equal-mass', 'non-spinning']

# Additional flags
single_catalog       = None # Available options: ['RIT', 'SXS', 'ET']
add_RWZ_data_flag    = 1
show_plots           = 1

######################
# End of user inputs #
######################

# ================================================================================================================================================

#######
# Fit #
#######

fit_dim = 4             # Number of free coefficient for each fitting variable

# Repeat the fit for each dataset
for dataset_type in dataset_types:

    template_model = select_template_model(dataset_type)

    # Repeat the fit for each of the quantities to fit
    for quantity_to_fit in quantities_to_fit:

        # Select the fitting quantities, based on the dataset type and template model.
        fitting_quantities_strings_list = select_fitting_quantities(dataset_type, quantity_to_fit)

        # Plot the fit for each of the fitting quantities combinations
        for fitting_quantities_string in fitting_quantities_strings_list:

            # Load the data. 
            # IMPROVEME: This is done inside the loop because the data is different for each quantity to fit (i.e. Mf-af do not have RWZ data). Should be improved and done only once.
            data = load_data(dataset_type)

            # Use the subset of catalogs selected by the user, and add perturbation theory data if requested
            catalogs, data = select_catalogs(single_catalog, data, add_RWZ_data_flag, dataset_type, quantity_to_fit)

            # Rescale by quasi-circular data. This avoids rescaling by nu and Heff, since both scale together.
            data[quantity_to_fit] = data[quantity_to_fit]/(data[quantity_to_fit+'_qc'])

            # Convert the catalogs list to a string
            catalogs_string = create_catalog_string(catalogs)

            # Unpack fitting quantities
            fitting_quantities_dict = {}
            for fitting_quantity_x in fitting_quantities_string.split('-'): fitting_quantities_dict[fitting_quantity_x] = data[fitting_quantity_x]

            # Read the coefficients from a previously performed fit
            coeffs = read_fit_coefficients(quantity_to_fit, fitting_quantities_dict, fitting_quantities_string, dataset_type, catalogs_string, template_model, fit_dim)

            # Plot the results
            plot_results(fitting_quantities_dict, quantity_to_fit, data, template_model, fit_dim, coeffs, dataset_type, catalogs_string, fitting_quantities_string)

if(show_plots): plt.show()