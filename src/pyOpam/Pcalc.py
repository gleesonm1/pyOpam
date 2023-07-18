import numpy as np
import pandas as pd
import Thermobar

## Equations for XAl, XMg, and XCa

def P_Voigt2017(Liq_Comps, P):
    '''
    Equations for the cation fraction of Al, Mg, and Ca in basaltic melts from Voigt et al. (2017).

    Parameters
    -------

    Liq_comps: DataFrame
        melt cation fraction (normalised to 1) with column headings Si_Liq_cat_frac, Mg_Liq_cat_frac etc.

    P: float
        Pressure of the calculation in GPa.

    Returns
    -------
    XAl, XMg, XCa: numpy array
       Predicted cation fractions of Al, Mg, and Ca

    '''
    XAl=0.239+0.01801*P+0.162*Liq_Comps['Na_Liq_cat_frac']+0.485*Liq_Comps['K_Liq_cat_frac']-0.304*Liq_Comps['Ti_Liq_cat_frac']-0.32*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-0.353*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']-0.13*Liq_Comps['Si_Liq_cat_frac']+5.652*Liq_Comps['Cr_Liq_cat_frac']
    XCa=1.07-0.02707*P-0.634*Liq_Comps['Na_Liq_cat_frac']-0.618*Liq_Comps['K_Liq_cat_frac']-0.515*Liq_Comps['Ti_Liq_cat_frac']-0.188*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-0.597*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']-3.044*Liq_Comps['Si_Liq_cat_frac']-9.367*Liq_Comps['Cr_Liq_cat_frac']+2.477*(Liq_Comps['Si_Liq_cat_frac']**2)
    XMg=-0.173+0.00625*P-0.541*Liq_Comps['Na_Liq_cat_frac']-1.05*Liq_Comps['K_Liq_cat_frac']-0.182*Liq_Comps['Ti_Liq_cat_frac']-0.493*(1-Liq_Comps['Fe3Fet_Liq'])*Liq_Comps['Fet_Liq_cat_frac']-0.028*Liq_Comps['Fe3Fet_Liq']*Liq_Comps['Fet_Liq_cat_frac']+1.599*Liq_Comps['Si_Liq_cat_frac']+3.246*Liq_Comps['Cr_Liq_cat_frac']-1.873*(Liq_Comps['Si_Liq_cat_frac']**2)

    return XAl, XMg, XCa

def P_Yang1996(Liq_Comps, P):
    '''
    Equations for the cation fraction of Al, Mg, and Ca in basaltic melts from Yang et al. (1996).

    Parameters
    -------

    Liq_comps: DataFrame
        melt cation fraction (normalised to 1) with column headings Si_Liq_cat_frac, Mg_Liq_cat_frac etc.

    P: float
        Pressure of the calculation in GPa.

    Returns
    -------
    XAl, XMg, XCa: numpy array
       Predicted cation fractions of Al, Mg, and Ca
    '''
    XAl=0.236+0.00218*(P*10)+0.109*Liq_Comps['Na_Liq_cat_frac']+0.593*Liq_Comps['K_Liq_cat_frac']-0.350*Liq_Comps['Ti_Liq_cat_frac']-0.299*Liq_Comps['Fet_Liq_cat_frac']-0.13*Liq_Comps['Si_Liq_cat_frac']
    XCa=1.133-0.00339*(P*10)-0.569*Liq_Comps['Na_Liq_cat_frac']-0.776*Liq_Comps['K_Liq_cat_frac']-0.672*Liq_Comps['Ti_Liq_cat_frac']-0.214*Liq_Comps['Fet_Liq_cat_frac']-3.355*Liq_Comps['Si_Liq_cat_frac']+2.830*(Liq_Comps['Si_Liq_cat_frac']**2)
    XMg=-0.277+0.00114*(P*10)-0.543*Liq_Comps['Na_Liq_cat_frac']-0.947*Liq_Comps['K_Liq_cat_frac']-0.117*Liq_Comps['Ti_Liq_cat_frac']-0.490*Liq_Comps['Fet_Liq_cat_frac']+2.086*Liq_Comps['Si_Liq_cat_frac']-2.4*(Liq_Comps['Si_Liq_cat_frac']**2)

    return XAl, XMg, XCa

## Alternative OPAM formulations

def P_Herzberg2004(Liq_Comps):
    '''
    Method described in Herzberg et al. (2004) to calculate the pressure of OPAM-saturated liquids.

    Parameters:
    ----------
    Liq_Comps: Pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'

    Returns:
    ----------
    liq_Comps: Pandas DataFrame
        Input DataFrame with a column added displaying the calculated pressure in kbar.

    '''
    mol_perc = 100*Thermobar.calculate_anhydrous_mol_fractions_liquid(Liq_Comps)

    # calculate projections
    An = mol_perc['Al2O3_Liq_mol_frac'] + mol_perc['Cr2O3_Liq_mol_frac'] + mol_perc['TiO2_Liq_mol_frac']
    Di = mol_perc['CaO_Liq_mol_frac'] + mol_perc['Na2O_Liq_mol_frac'] + 3*mol_perc['K2O_Liq_mol_frac'] - mol_perc['Al2O3_Liq_mol_frac'] - mol_perc['Cr2O3_Liq_mol_frac']
    En = mol_perc['SiO2_Liq_mol_frac'] - 0.5*mol_perc['Al2O3_Liq_mol_frac'] - 0.5*mol_perc['Cr2O3_Liq_mol_frac'] - 0.5*mol_perc['FeOt_Liq_mol_frac'] - 0.5*mol_perc['MnO_Liq_mol_frac'] - 0.5*mol_perc['MgO_Liq_mol_frac'] - 1.5*mol_perc['CaO_Liq_mol_frac'] - 3*mol_perc['Na2O_Liq_mol_frac'] - 3*mol_perc['K2O_Liq_mol_frac']

    # ensure projections sum to 100
    Bottom = An + Di + En
    An = 100*An/Bottom
    Di = 100*Di/Bottom
    En = 100*En/Bottom

    Liq_Comps['P_kbar_calc'] =10 * (-9.1168 + 0.2446*(0.4645*En + An) - 0.001368*((0.4645*En + An)**2))

    return Liq_Comps

def P_Villiger2007(Liq_Comps):
    '''
    Method described in Villiger et al. (2007) to calculate the pressure of OPAM-saturated liquids.

    Parameters:
    ----------
    Liq_Comps: Pandas DataFrame
        Dataframe of compositions. All columns headings should contain the suffix '_Liq'

    Returns:
    ----------
    liq_Comps: Pandas DataFrame
        Input DataFrame with a column added displaying the calculated pressure in kbar.

    '''
    MgNo = (Liq_Comps['MgO_Liq']/40.3044)/(Liq_Comps['MgO_Liq']/40.3044 + Liq_Comps['FeOt_Liq']/71.844)
    P_kbar = (Liq_Comps['CaO_Liq'] - 3.98 - 14.96*MgNo)/(-0.260)

    Liq_Comps['P_kbar_calc'] = P_kbar

    return Liq_Comps

