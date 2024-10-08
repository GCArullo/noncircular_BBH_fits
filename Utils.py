# Standard python imports
import numpy as np, pandas as pd, os

twopi = 2.0*np.pi

def create_catalog_string(catalogs):

    catalogs_string = ''
    for catalog_y in catalogs: catalogs_string += catalog_y + '-'
    catalogs_string = catalogs_string[:-1]

    return catalogs_string

def select_catalogs(single_catalog, data, add_RWZ_data_flag, dataset_type, quantity_to_fit):

    if(single_catalog is not None): 
        data = data[data['catalog']==single_catalog]
        catalogs = [single_catalog]
    else:
        catalogs = ['RIT', 'SXS', 'ET']

    # Add perturbation theory data if requested, but only once
    if(add_RWZ_data_flag): data, catalogs = add_RWZ_data(dataset_type, quantity_to_fit, data, catalogs)

    return catalogs, data

def load_data(dataset_type, ringdown_data):

    if(ringdown_data): tag_append = '_plus_ringdown'
    else             : tag_append = ''

    data_dir = 'Parameters_to_fit'

    # Load the data
    if(ringdown_data): 
        data = pd.read_csv(os.path.join(data_dir, f'Parameters_non-spinning{tag_append}.csv'))
    else:
        if('non-spinning' in dataset_type): data = pd.read_csv(os.path.join(data_dir, f'Parameters_non-spinning.csv'))
        else                              : data = pd.read_csv(os.path.join(data_dir, f'Parameters_aligned-spins-equal-mass.csv'))

    # Filter the data, accounting for roundoffs
    if('equal-mass'   in dataset_type): data = data[data['q']>0.98]

    return data

def merge_with_RWZ_data(data):

    file_RW = np.genfromtxt('Data/testmass_data.txt', names=True)

    dict_RW = {
            # Metadata
            'ID'             : np.arange(1, len(file_RW['b_pk'])+1, dtype=int)                ,
            'catalog'        : np.full(     len(file_RW['b_pk']), 'RWZ')                      ,

            # Initial quantities
            'q'              : np.ones(     len(file_RW['b_pk'])) * 1e-3                      ,
            'nu'             : np.ones(     len(file_RW['b_pk'])) * 1e-3                      ,
            'ecc'            : np.arange(0.0, 1.0, 0.05)                                      ,
            'Heff_til'       : file_RW['E_pk']                                                ,
            'Jmrg_til'       : file_RW['J_pk']                                                ,
            'b_massless_Heff': file_RW['b_pk']                                                , 
            'b_massless_EOB' : file_RW['b_pk']                                                , 
            'a_f'            : np.zeros(    len(file_RW['b_pk']))                             , # This is doubled on purpose, to trick pandas and use it both as quantity to fit and fitting variable

            # Final quantities
            'Mf'             : np.ones(     len(file_RW['b_pk']))                             , 
            'af'             : np.zeros(    len(file_RW['b_pk']))                             , 
            'Mf_qc'          : np.ones(     len(file_RW['b_pk']))                             , 
            'af_qc'          : np.zeros(    len(file_RW['b_pk']))                             ,
            'A_peak22'       : file_RW['A22_pk']                                              ,
            'omega_peak22'   : file_RW['omg22_pk']                                            ,
            'A_peak22_qc'    : np.ones(     len(file_RW['A22_pk'])  ) * file_RW['A22_pk'][0]  ,
            'omega_peak22_qc': np.ones(     len(file_RW['omg22_pk'])) * file_RW['omg22_pk'][0],
            }

    data_RW = pd.DataFrame(dict_RW)
    
    data = pd.concat([data, data_RW], ignore_index=True)

    return data

def add_RWZ_data(dataset_type, quantity_to_fit, data, catalogs):

    if((dataset_type=='non-spinning') and ((quantity_to_fit=='A_peak22') or (quantity_to_fit=='omega_peak22'))): 
        data = merge_with_RWZ_data(data)
        catalogs.append('RWZ')

    return data, catalogs

def select_fitting_quantities(dataset_type, quantity_to_fit):

    if  (quantity_to_fit=='A_peak22'                     ): fitting_quantities_base_list = ['b_massless_EOB', 'Heff_til-b_massless_EOB']
    elif(quantity_to_fit=='omega_peak22'                 ): fitting_quantities_base_list = ['b_massless_EOB', 'Heff_til-b_massless_EOB']
    elif(quantity_to_fit=='Mf'                           ): fitting_quantities_base_list = ['Heff_til'      , 'Heff_til-Jmrg_til'      ]
    elif(quantity_to_fit=='af'                           ): fitting_quantities_base_list = ['Jmrg_til'      , 'Heff_til-Jmrg_til'      ]

    # KerrBinary
    elif(quantity_to_fit==  'A_220_from_22_220_median_ta'     or \
         quantity_to_fit==  'A_330_from_33_330_median_ta'     or \
         quantity_to_fit==  'A_210_from_21_210_median_ta'     or \
         quantity_to_fit==  'A_440_from_44_440_median_ta'     or \
         quantity_to_fit==  'A_320_from_32_320_220_median_ta' or \
         quantity_to_fit==  'A_200_from_20_200_median_ta'     or \
         quantity_to_fit=='phi_220_from_22_220_median_ta'     or \
         quantity_to_fit=='phi_330_from_33_330_median_ta'     or \
         quantity_to_fit=='phi_210_from_21_210_median_ta'     or \
         quantity_to_fit=='phi_440_from_44_440_median_ta'     or \
         quantity_to_fit=='phi_320_from_32_320_220_median_ta' or \
         quantity_to_fit=='phi_200_from_20_200_median_ta'        ): fitting_quantities_base_list = ['b_massless_EOB', 'Heff_til-b_massless_EOB']

    fitting_quantities_strings_list = []
    # In case we are fitting non-equal mass data, add nu-dependence to each fitting quantity, if using non-factorised
    if not('equal-mass' in dataset_type):
        for fx in fitting_quantities_base_list: fitting_quantities_strings_list.append('nu-'+fx)
    else: fitting_quantities_strings_list = fitting_quantities_base_list
    # In case we are fitting aligned-spin data, add chi_eff-dependence to each fitting quantity
    if('aligned-spins' in dataset_type):
        fitting_quantities_strings_list_tmp = []
        for fx in fitting_quantities_strings_list: 
            fitting_quantities_strings_list_tmp.append('chieff-'+fx)
        fitting_quantities_strings_list = fitting_quantities_strings_list_tmp

    return fitting_quantities_strings_list

def select_template_model(dataset_type, ringdown_data):

    if(ringdown_data)                   : template_model = 'rational'
    else                                :
        if('equal-mass' in dataset_type): template_model = 'rational'
        else                            : template_model = 'factorised-nu'

    return template_model

def template(coeffs, fitting_quantities_dict, template_model='rational'):

    any_quantity_name            = list(fitting_quantities_dict.keys())[0]
    len_data                     = len(fitting_quantities_dict[any_quantity_name])
    ones_len_data                = [1.0] * len_data 
    result                       = [coeffs[0]] * len_data
    len_coeffs                   = len(coeffs)-1
    fitting_quantities_dict_loop = list(fitting_quantities_dict.keys())
    if('factorised-nu' in template_model): 
        fitting_quantities_dict_loop.remove('nu')
        single_var_coeffs_len    = int(int(len_coeffs/len(fitting_quantities_dict_loop))/2)
    else:
        single_var_coeffs_len    = int(len_coeffs/len(fitting_quantities_dict_loop))
    first_half_vec_len           = int((single_var_coeffs_len)/2)

    # Loop on the fitting quantities
    for (i, key_x) in enumerate(fitting_quantities_dict_loop):

        # Construct a rational function per fitting quantity
        if(template_model == 'rational'):

            # With this structure, the first block of length single_var_coeffs_len are the coefficients of the first variable, the second block are the ones of the second, and so on.

            result_i_num     = ones_len_data + np.sum([coeffs[j] * fitting_quantities_dict[key_x]**(j-(i*single_var_coeffs_len)                   )                          for j in range(1                     +i*single_var_coeffs_len, (first_half_vec_len+1) + ( i    * single_var_coeffs_len)   )], axis=0)
            result_i_den     = ones_len_data + np.sum([coeffs[j] * fitting_quantities_dict[key_x]**(j-(i*single_var_coeffs_len)-first_half_vec_len)                          for j in range((first_half_vec_len+1)+i*single_var_coeffs_len,                          ((i+1) * single_var_coeffs_len)+1 )], axis=0)

        # Construct a rational function per fitting quantity, except for the mass ratio, which is folded-in through X
        # With this structure, the first half of the coefficients are the X=0, while the second half are the X=1 in reverse order

        elif('factorised-nu' in template_model):

            if(key_x == 'nu'): continue

            X = 1 - 4.*fitting_quantities_dict['nu']

            if(  template_model == 'factorised-nu'  ):

                result_i_num = ones_len_data + np.sum([coeffs[j] * fitting_quantities_dict[key_x]**(j-(i*single_var_coeffs_len)                   ) * (1.0 + coeffs[-j] * X) for j in range(1                     +i*single_var_coeffs_len, (first_half_vec_len+1) + ( i    * single_var_coeffs_len)   )], axis=0)
                result_i_den = ones_len_data + np.sum([coeffs[j] * fitting_quantities_dict[key_x]**(j-(i*single_var_coeffs_len)-first_half_vec_len) * (1.0 + coeffs[-j] * X) for j in range((first_half_vec_len+1)+i*single_var_coeffs_len,                          ((i+1) * single_var_coeffs_len)+1 )], axis=0)

        result *= result_i_num/result_i_den

    return result

def read_fit_coefficients(quantity_to_fit, fitting_quantities_dict, fitting_quantities_string, dataset_type, catalogs_string, template_model, fit_dim):

    print(f'* Post-processing `{quantity_to_fit}` fit in terms of {list(fitting_quantities_dict.keys())}, trained on {dataset_type} simulations with the catalogs: {catalogs_string}.\n')

    coeffs_dir = 'Fitting_coefficients'
    coeffs = pd.read_csv(os.path.join(coeffs_dir, f'Fitting_coefficients_{dataset_type}_{catalogs_string}_{template_model}_{fit_dim}_{fitting_quantities_string}_{quantity_to_fit}.csv'))['coeffs']

    coeffs = np.array(coeffs)

    return coeffs

def set_qc_name(quantity_to_fit):

    if('median_ta' in quantity_to_fit): name_qc = quantity_to_fit.replace('median_ta', 'qc')
    else                              : name_qc = quantity_to_fit+'_qc'

    return name_qc

def set_actual_quantity_to_fit(data, quantity_to_fit, dataset_type):

    # Define the qc name
    name_qc = set_qc_name(quantity_to_fit)

    # Odd m modes have zero amplitude in the equal mass limit. Remove them from the fit.
    if('210' in quantity_to_fit or '330' in quantity_to_fit): data = data[data['q']<0.98]

    # Normalise the amplitude wrt the qc value
    if not('phi' in quantity_to_fit): data[quantity_to_fit] = data[quantity_to_fit]/data[name_qc]
        
    # Define the phase wrt the 220 arbitrary phase, wrt the qc value and unwrap it
    else                            : 
        m_int = int(quantity_to_fit.split('_')[1][1])
        data[quantity_to_fit] = np.unwrap(2.* data[quantity_to_fit] - m_int * data['phi_220_from_22_220_median_ta'] - data[name_qc])

        # Manually add unwrapping needed for the phase.
        # Alternative to this manual shift, phase wrapper in 3D: https://github.com/geggo/phase-unwrap?tab=readme-ov-file
        data = add_manual_unwrap(data, quantity_to_fit, dataset_type)

    return data

def add_manual_unwrap(data, quantity_to_fit, dataset_type):

    if(  quantity_to_fit=='phi_330_from_33_330_median_ta'):                        
        data[quantity_to_fit] += (2.*np.pi)
    elif(quantity_to_fit=='phi_210_from_21_210_median_ta'):                        
        data[quantity_to_fit][data[quantity_to_fit]<-1.8] += 2.*np.pi
    elif(quantity_to_fit=='phi_440_from_44_440_median_ta'): 
        if(dataset_type=='non-spinning-equal-mass'       ): data[quantity_to_fit] += (2.*np.pi)
        elif(dataset_type=='non-spinning'                ): 
            data[quantity_to_fit][data[quantity_to_fit]<-10] += (2.*np.pi)*2
            data[quantity_to_fit][data[quantity_to_fit]>10]  -= (2.*np.pi)*2
            data[quantity_to_fit][data[quantity_to_fit]<0]   += (2.*np.pi)*2
            data[quantity_to_fit][data[quantity_to_fit]>3]   -= (2.*np.pi)*2
            data[quantity_to_fit] += (2.*np.pi)*2
            data[quantity_to_fit][data[quantity_to_fit]>9]   -= (2.*np.pi)
            data[quantity_to_fit][data[quantity_to_fit]>9]   -= (2.*np.pi)
            data[quantity_to_fit]                            -= (2.*np.pi)
    elif(quantity_to_fit=='phi_320_from_32_320_220_median_ta'): 
            data[quantity_to_fit]                             += (2.*np.pi)*2
            data[quantity_to_fit][data[quantity_to_fit]>11.0] -= (2.*np.pi)
            data[quantity_to_fit][data[quantity_to_fit]< 4  ] += (2.*np.pi)
            data[quantity_to_fit][data[quantity_to_fit]< 5.6] += (2.*np.pi)
            data[quantity_to_fit][data[quantity_to_fit]>11.0] -= (2.*np.pi)
            data[quantity_to_fit][data[quantity_to_fit]< 6.0] += (2.*np.pi)
            data[quantity_to_fit][data[quantity_to_fit]>11.0] -= (2.*np.pi)
            data[quantity_to_fit][(data['nu']<0.21) & (data[quantity_to_fit]>9.0)] -= (2.*np.pi)
            data[quantity_to_fit][(data['b_massless_EOB']>3.2) & (data[quantity_to_fit]>10.0)] -= (2.*np.pi)
    
    return data