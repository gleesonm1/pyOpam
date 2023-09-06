import numpy as np
import pandas as pd
import Thermobar
import scipy
from pyOpam.Pcalc import *
from pyOpam.Tcalc import *
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

## OPAM calculations

def calc_liq_press(*, liq_comps = None, equationP = None, fo2 = None, fo2_offset = None, Fe3Fet_Liq = None, equationT = None, T_K = None):
    '''
    Calculate multi-phase saturation pressure and 'probability' of three phase saturation for basaltic melts. Pressure sensitive equations for the cation fraction of Al, Mg, and Ca in the melt phase are used to determine the pressure of storage by identifying the location of the minimum misfit between the observed and calculated cation fractions.

    Parameters:
    -----------
    liq_comps: pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'.

    equationP: string
        Specifies the equations used to calculate the theorectical cation fractions of Al, Mg, and Ca at different pressures. Choice of "Yang1996" or "Voigt2017". "Yang1996" is used as default.

    fo2: string
        fO2 buffer to use if the equations of Voigt et al. 2017 are chosen and no Fe3Fet_Liq is specified in the liq_comps dataframe. Here the buffer will be used to calculate the Fe3Fet_Liq value for each composition in the dataframe using Thermobar and a Temperature specified by "equationT"

    equationT: string
        If fo2 is not None, a liquid-only thermometer from Thermobar can be specified here to calculate the temperature of the different liquids in liq_comps prior to calculation of the Fe3Fet_Liq value for each sample. As default, "T_Helz1987_MgO" is used.

    Returns:
    ----------
    Results: pandas DataFrame
        copy of the initial DataFrame with two columns added: calculated pressures and 'probability' of three-phase saturation
    '''

    # check all required parameters are present
    if liq_comps is None:
        raise Exception("No composition specified")

    if equationP is None:
        equationP = "P_Yang1996"

    if fo2 is None and Fe3Fet_Liq is None and "Fe3Fet_Liq" not in list(liq_comps.keys()) and equationP == "P_Voigt2017":
        raise Warning(f'{equationP} requires you to specify the redox state of the magma. Please ensure the Fe3+/Fe_total ratio of the melt phase is specified in the input dataframe.')

    # create a copy of the input dataframe
    liq = liq_comps.copy()
    liq = liq.reset_index(drop = True)

    # calculation is determined based on equation chosen
    if equationP == "P_Herzberg2004":
        Results = P_Herzberg2004(liq)
    elif equationP == "P_Villiger2007":
        Results = P_Villiger2007(liq)
    else:
        if equationP == "P_Voigt2017":
            # determine Fe3 from fO2 if set
            if fo2 is not None:
                if equationT is None:
                    equationT="T_Helz1987_MgO"

                # estimate Liq T to convert fO2 into Fe3Fet ratio
                if T_K is None:
                    T_K = Thermobar.calculate_liq_only_temp(liq_comps=liq,  equationT=equationT)
                liq_comp_Fe3 = Thermobar.convert_fo2_to_fe_partition(liq_comps=liq, T_K=T_K, P_kbar=3, fo2=fo2, fo2_offset = fo2_offset, model="Kress1991", renorm=False)
                liq['Fe3Fet_Liq'] = liq_comp_Fe3['Fe3Fet_Liq']
            elif Fe3Fet_Liq is not None:
                liq['Fe3Fet_Liq'] = Fe3Fet_Liq

            #liq_comps['Fe3Fet_Liq'] = liq['Fe3Fet_Liq'].copy()

        # As OPAM works using cation fractions, ensure minor elements not included in the OPAM parameterisation are not influencing the cation fraction calculations.
        # if "P2O5_Liq" in list(liq.keys()):
        #     liq['P2O5_Liq'] = np.zeros(len(liq['SiO2_Liq']))
        # if "MnO_Liq" in list(liq.keys()):
        #     liq['MnO_Liq'] = np.zeros(len(liq['MnO_Liq']))
        # if "H2O_Liq" in list(liq.keys()):
        #     liq['H2O_Liq'] = np.zeros(len(liq['H2O_Liq']))
        # if "Cr2O3_Liq" in list(liq.keys()) and equationP != "P_Voigt2017":
        #     liq['Cr2O3_Liq'] = np.zeros(len(liq['SiO2_Liq']))

        #calculate cation fractions and create np.ndarrays to store outputs
        liq_cats=Thermobar.calculate_anhydrous_cat_fractions_liquid(liq)
        P = np.zeros(len(liq_cats['SiO2_Liq']))
        Prob = np.zeros(len(liq_cats['SiO2_Liq']))

        # minimise P for each sample and calculate Pf using scipy.stats.chi2
        for t in range(len(liq_cats['SiO2_Liq'])):
            liq_cat = liq_cats.loc[t].to_dict().copy()

            def optimize_function(P_GPa):
                return findMin(P_GPa, liq_cat, equationP)

            res = minimize_scalar(optimize_function, method = 'brent')
            #res = minimize_scalar(findMin, method = 'brent', args = (liq_cat, equationP))
            P[t] = res.x
            Stats = res.fun
            Prob[t] = 1 - chi2.cdf(Stats, 2)

        # save outputs into the dataframe, converting P into kbar
        liq['P_kbar_calc'] = P*10
        liq['Pf'] = Prob
        if "P2O5_Liq" in list(liq_comps.keys()):
            liq['P2O5_Liq'] = liq_comps['P2O5_Liq']
        if "MnO_Liq" in list(liq_comps.keys()):
            liq['MnO_Liq'] = liq_comps['MnO_Liq']
        if "H2O_Liq" in list(liq_comps.keys()):
            liq['H2O_Liq'] = liq_comps['H2O_Liq']
        if "Cr2O3_Liq" in list(liq_comps.keys()) and equationP != "P_Voigt2017":
            liq['Cr2O3_Liq'] = liq_comps['Cr2O3_Liq']

        Results = liq.copy()

    return Results

def calc_liq_temp(*, liq_comps = None, equationT = None, P = None, fo2 = None, Fe3Fet_Liq = None):
    '''
    Calculate multi-phase saturation pressure and 'probability' of three phase saturation for basaltic melts. Pressure sensitive equations for the cation fraction of Al, Mg, and Ca in the melt phase are used to determine the pressure of storage by identifying the location of the minimum misfit between the observed and calculated cation fractions.

    Parameters:
    -----------
    liq_comps: pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'.

    equationT: string
        Specifies the equations used to calculate the liquid temperature. Choice of "T_Yang1996" or "T_Voigt2017". "T_Yang1996" is used as default.

    P: float or np.ndarray
        either single value or array equal to the length of the number of samples. Previously calculated pressures

    equationP: string
        Specifies the equations used to calculate the theorectical cation fractions of Al, Mg, and Ca at different pressures. Choice of "Yang1996" or "Voigt2017". "Yang1996" is used as default.

    fo2: string
        fO2 buffer to use if the equations of Voigt et al. 2017 are chosen and no Fe3Fet_Liq is specified in the liq_comps dataframe. Here the buffer will be used to calculate the Fe3Fet_Liq value for each composition in the dataframe using Thermobar and a Temperature specified by "equationT"

    Returns:
    ----------
    Results: pandas DataFrame
        copy of the initial DataFrame with two columns added: calculated pressures and 'probability' of three-phase saturation
    '''

    # check all required parameters are present
    if liq_comps is None:
        raise Exception("No composition specified")

    if equationT is None:
        equationT = "T_Yang1996"

    if fo2 is None and Fe3Fet_Liq is None and "Fe3Fet_Liq" not in list(liq_comps.keys()) and equationT == "T_Voigt2017":
        raise Warning(f'{equationT} requires you to specify the redox state of the magma. Please ensure the Fe3+/Fe_total ratio of the melt phase is specified in the input dataframe.')

    # create a copy of the input dataframe
    liq = liq_comps.copy()
    liq = liq.reset_index(drop = True)

    if equationT == "T_Voigt2017":
        # determine Fe3 from fO2 if set
        e = equationT
        if fo2 is not None:
            # estimate Liq T to convert fO2 into Fe3Fet ratio
            T = Thermobar.calculate_liq_only_temp(liq_comps=liq,  equationT="T_Helz1987_MgO")-273.15
            liq_comp_Fe3 = Thermobar.convert_fo2_to_fe_partition(liq_comps=liq, T_K=T+273.15, P_kbar=P/10, fo2=fo2, model="Kress1991", renorm=False)
            liq['Fe3Fet_Liq'] = liq_comp_Fe3['Fe3Fet_Liq']
        elif Fe3Fet_Liq is not None:
            liq['Fe3Fet_Liq'] = Fe3Fet_Liq

        liq_comps['Fe3Fet_Liq'] = liq['Fe3Fet_Liq'].copy()

        equationT = e

    # As OPAM works using cation fractions, ensure minor elements not included in the OPAM parameterisation are not influencing the cation fraction calculations.
    if "P2O5_Liq" in list(liq.keys()):
        liq['P2O5_Liq'] = np.zeros(len(liq['SiO2_Liq']))
    if "MnO_Liq" in list(liq.keys()):
        liq['MnO_Liq'] = np.zeros(len(liq['MnO_Liq']))
    if "H2O_Liq" in list(liq.keys()):
        liq['H2O_Liq'] = np.zeros(len(liq['H2O_Liq']))
    if "Cr2O3_Liq" in list(liq.keys()) and equationT != "T_Voigt2017":
        liq['Cr2O3_Liq'] = np.zeros(len(liq['SiO2_Liq']))

    #calculate cation fractions and create np.ndarrays to store outputs
    liq_cats=Thermobar.calculate_anhydrous_cat_fractions_liquid(liq)

    if equationT == "T_Yang1996":
        T_C = T_Yang1996(liq_cats, P)
    else:
        T_C = T_Voigt2017(liq_cats, P)

    Results = liq_comps.copy()
    Results['T_C_calc'] = T_C

    return Results

def calc_liq_press_temp(*, liq_comps = None, equationP = None, equationT = None, fo2 = None, Fe3Fet_Liq = None):

    P_results = calc_liq_press(liq_comps = liq_comps, equationP = equationP, fo2 = fo2, Fe3Fet_Liq = Fe3Fet_Liq)
    Results = calc_liq_temp(liq_comps = liq_comps, equationT = equationT, P = P_results['P_kbar_calc'].values, fo2 = fo2, Fe3Fet_Liq = Fe3Fet_Liq)

    Results['P_kbar_calc'] = P_results['P_kbar_calc']
    Results['Pf'] = P_results['Pf']

    return Results

## minimisation calculation

def findMin(P_Gpa, liq_cat, equationP):
    '''
    Funcion used to identify the location of the minimum value of X2 (modified Chi-squared), which is a function of pressure. Individually, the function calculates the modified Chi-squared value for a given pressure and melt composition, but when combined with the scipy.optimize.minimize_scalar function this can be used to determine the pressure at which the minimum value of X2 is located.

    Parameters:
    -----------
    P_GPa: float
        Pressure (in GPa).

    liq_cat: dict
        Dictionary containing the melt cation fractions.

    equationP: string
        Specifies the equations used to calculate the theorectical cation fractions of Al, Mg, and Ca at different pressures. Choice of "Yang1996" or "Voigt2017". "Yang1996" is used as default.

    Returns:
    ----------
    X2: float
        Value of the modified Chi-Squared expression from Hartley et al. (2018) at the specified pressure.
    '''

    if equationP == "P_Yang1996":
        XAl, XMg, XCa = P_Yang1996(liq_cat, P_Gpa)
    else:
        XAl, XMg, XCa = P_Voigt2017(liq_cat, P_Gpa)

    X2 = ((liq_cat['Al_Liq_cat_frac']-XAl)/(0.05*liq_cat['Al_Liq_cat_frac']))**2. + ((liq_cat['Ca_Liq_cat_frac']-XCa)/(0.05*liq_cat['Ca_Liq_cat_frac']))**2. + ((liq_cat['Mg_Liq_cat_frac']-XMg)/(0.05*liq_cat['Mg_Liq_cat_frac']))**2.

    return X2

