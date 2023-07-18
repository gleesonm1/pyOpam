import numpy as np
import pandas as pd

def T_Voigt2017(Liq_Comps, P):
    '''
    Equations for liquid temperature in basaltic melts from Voigt et al. (2017).

    Parameters
    -------

    Liq_comps: DataFrame
        melt cation fraction (normalised to 1) with column headings Si_Liq_cat_frac, Mg_Liq_cat_frac etc.

    P: float
        Pressure of the calculation in GPa.

    Returns
    -------
    T_calc_C: float or np.ndarray
       Predicted liquid temperature

    '''
    T_calc_C = 679.1+46.8*P-671.9*Liq_Comps['Na_Liq_cat_frac']-699.1*Liq_Comps['K_Liq_cat_frac']+3022*Liq_Comps['Ti_Liq_cat_frac']-627.6*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-283.5*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']+2689*Liq_Comps['Si_Liq_cat_frac']-4799*Liq_Comps['Cr_Liq_cat_frac']-3056*(Liq_Comps['Si_Liq_cat_frac']**2)-8228*(Liq_Comps['Si_Liq_cat_frac']*Liq_Comps['Ti_Liq_cat_frac'])

    return T_calc_C

def T_Yang1996(Liq_Comps, P):
    '''
    Equations for the liquid temperature in basaltic melts from Yang et al. (1996).

    Parameters
    -------

    Liq_comps: DataFrame
        melt cation fraction (normalised to 1) with column headings Si_Liq_cat_frac, Mg_Liq_cat_frac etc.

    P: float
        Pressure of the calculation in GPa.

    Returns
    -------
    T_calc_C: float or np.ndarray
       Predicted liquid temperature

    '''
    # coefficients for Si and Fe switched following the recommondation of Voigt et al. (2017)
    T_calc_C = 581.7 + 5.858*(P*10) - 691.0*Liq_Comps['Na_Liq_cat_frac'] - 848.9*Liq_Comps['K_Liq_cat_frac'] + 11492*Liq_Comps['Ti_Liq_cat_frac'] - 574.3*Liq_Comps['Fet_Liq_cat_frac'] + 3114*Liq_Comps['Si_Liq_cat_frac'] - 3529*(Liq_Comps['Si_Liq_cat_frac']**2) - 25679*(Liq_Comps['Si_Liq_cat_frac']*Liq_Comps['Ti_Liq_cat_frac'])

    return T_calc_C