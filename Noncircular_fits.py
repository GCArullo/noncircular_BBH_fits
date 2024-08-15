import matplotlib.pyplot as plt

# Package-specific imports
from Utils import create_catalog_string, load_data, read_fit_coefficients, select_catalogs, select_fitting_quantities, select_template_model, set_actual_quantity_to_fit, add_manual_unwrap
from Plots import plot_results

########################
# Start of user inputs #
########################

# Select either ringdown or merger-remnant data
ringdown_data        = 1

# Dataset and quantities to fit 

if(ringdown_data): quantities_to_fit = ['A_220_from_22_220_median_ta', 'A_330_from_33_330_median_ta', 'A_210_from_21_210_median_ta', 'A_440_from_44_440_median_ta', 'A_320_from_32_320_220_median_ta', 'phi_330_from_33_330_median_ta', 'phi_210_from_21_210_median_ta', 'phi_440_from_44_440_median_ta', 'phi_320_from_32_320_220_median_ta']  
else             : quantities_to_fit = ['A_peak22', 'omega_peak22', 'Mf', 'af']  

dataset_types        = ['non-spinning-equal-mass', 'non-spinning'] # Available options: ['non-spinning-equal-mass', 'non-spinning', 'aligned-spins-equal-mass']

# Additional flags
single_catalog       = None # Available options: ['RIT', 'SXS', 'ET']
add_RWZ_data_flag    = 1
show_plots           = 1

######################
# End of user inputs #
######################

# ================================================================================================================================================

if(ringdown_data and ('aligned-spins' in dataset_types)): raise ValueError('Ringdown data is not yet available for aligned spins.')
if(ringdown_data): single_catalog  = 'RIT'

#######
# Fit #
#######

fit_dim = 4             # Number of free coefficient for each fitting variable

# Repeat the fit for each dataset
for dataset_type in dataset_types:

    template_model = select_template_model(dataset_type, ringdown_data)

    # Repeat the fit for each of the quantities to fit
    for quantity_to_fit in quantities_to_fit:

        if(dataset_type=='non-spinning-equal-mass' and ('210' in quantity_to_fit or '330' in quantity_to_fit)): continue

        # Select the fitting quantities, based on the dataset type and template model.
        fitting_quantities_strings_list = select_fitting_quantities(dataset_type, quantity_to_fit)

        # Plot the fit for each of the fitting quantities combinations
        for fitting_quantities_string in fitting_quantities_strings_list:

            # Load the data. 
            # IMPROVEME: This is done inside the loop because the data is different for each quantity to fit (i.e. Mf-af do not have RWZ data). Should be improved and done only once.
            data = load_data(dataset_type, ringdown_data)

            # Use the subset of catalogs selected by the user, and add perturbation theory data if requested
            catalogs, data = select_catalogs(single_catalog, data, add_RWZ_data_flag, dataset_type, quantity_to_fit)

            # Rescale by quasi-circular data and unwrap if needed. This avoid rescaling by nu and Heff, since both scale together.
            data = set_actual_quantity_to_fit(data, quantity_to_fit, dataset_type)

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