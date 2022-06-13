# -*- coding: utf-8 -*-
"""
Generates shear normal curves and plots for the inputs in the specified CSV.

Transition curve calculations modified from Rose et al. (2018).
Created on Wed Jul 31 16:47:19 2019

@author: JDanielson
"""


import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib_inline

warnings.filterwarnings("ignore", message="invalid value encountered in")

# filepaths
INPUT_PATH = r'C:\Users\jdanielson\OneDrive - BGC Engineering Inc\Local Work\030 SS2022 paper\example_inputs.csv'
OUTPUT_PATH = r'C:\Users\jdanielson\OneDrive - BGC Engineering Inc\Local Work\030 SS2022 paper\Outputs\\'
MODE = "ROSE" # options are "ROSE" "LINEAR" "GALLANT"

# plotting parameters
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=0.9)


def plot_hb_curve(params, disturb, axis, label=None, lstyle='solid'):
    """
    Plot HB curve for the given GSI, Sigci, mi on the stated axis.

    Parameters
    ----------
    params : dataframe
        With the columns 'Sigci', 'GSI', 'mi', 'UnitWeight'.
    disturb : number or array-like with a size of len(sigma_3)
        Hoek-Brown disturbance within the range 0-1.
    axis : matplotlib axis object
        axis upon which to plot curve.
    linestyle : matplotlib linestyle, optional
    label : string, optional
        axis legend label. The default is None.

    Returns
    -------
    intercept : number
        y-intercept of shear normal curve.
    hb_shear : numpy array
        shear values.
    hb_normal : numpy array
        normal values.

    """
    sig_3 = np.arange(-1, 10, 0.01)
    hb_shear, hb_normal = hb_to_shear_normal(params, sig_3, disturb)
    axis.plot(hb_normal, hb_shear, params.color, linestyle=lstyle,
              label=label, linewidth=1.0)

    sn_df = pd.DataFrame(data=np.vstack((hb_normal, hb_shear)).T, columns=[
        'NormalStress', 'ShearStress'])
    sn_df.dropna(inplace=True)
    intercept = sn_df.iloc[(sn_df['NormalStress']
                            ).abs().argsort()].ShearStress.iloc[0]
    return intercept, hb_shear, hb_normal


def hb_to_shear_normal(params, s_3, disturb):
    """
    Calculate HB shear normal pairs for the given parameters and sig 3 values.

    Parameters
    ----------
    params : dataframe
        With the columns 'Sigci', 'GSI', 'mi', 'UnitWeight'.
    s_3 : array-like
        sigma 3 values for which to calculate shear/normal pairs.
    disturb : number or array-like
        Hoek-Brown disturbance within the range 0-1.

    Returns
    -------
    hb_shear : numpy array
        shear values for given inputs.
    hb_normal : numpy array
        normal values for given inputs.

    """
    _mb = params.mi*np.exp((params.GSI-100)/(28-14*disturb))
    _s = np.exp((params.GSI-100)/(9-3*disturb))
    _a = 1/2+1/6*(np.e**(-params.GSI/15)-np.e**(-20/3))

    hb_normal = s_3+params.Sigci*((_mb*s_3/params.Sigci+_s)**_a) / \
        (2+_a*_mb*((_mb*s_3/params.Sigci+_s)**(_a-1)))
    hb_shear = params.Sigci*((_mb*s_3/params.Sigci+_s)**_a)*np.sqrt(1+_a*_mb*(
        _mb*s_3/params.Sigci+_s)**(_a-1))/(2+_a*_mb*(_mb*s_3/params.Sigci+_s)**(_a-1))
    return hb_shear, hb_normal


def est_sig3_at_depth(params, depth, disturb, sig_3):
    """Estimate sigma 3 for a given depth.

    Parameters
    ----------
    params : dataframe
        With the columns 'Sigci', 'GSI', 'mi', 'UnitWeight',
        'OverallSlopeAngle_deg', 'SaturationRatio'.
    depth : number
        Depth at which to estimate sig_3.
    disturb : number
        Hoek-Brown disturbance within the range 0-1.
    s_3 : array-like
        sigma 3 range.

    Returns
    -------
    s3_at_depth
        the sigma_3 value associated with the normal stress at a given depth.

    """
    unit_weight = params['UnitWeight']
    cos_alpha = np.cos(np.radians(params.OverallSlopeAngle_deg))

    sat_constant = 1-params.SaturationRatio * \
        0.009807/(unit_weight*cos_alpha**2)

    _, hb_normal = hb_to_shear_normal(params, sig_3, disturb)
    stress_df = pd.DataFrame(data=hb_normal, columns=['normal'])
    stress_df = stress_df.assign(sigma_3=sig_3)

    norm_at_limit = depth*unit_weight*cos_alpha**2*sat_constant

    # interpolate between sigma 3 values assoc. with the normal stress depth
    stress_df = stress_df.dropna(subset=['normal'])
    s3_at_depth = np.interp(norm_at_limit,stress_df['normal'],stress_df['sigma_3'])

    return s3_at_depth


def trans_curve(unit_params, sig_3):
    """Develop the transition disturbance shear normal curves.

    Has fully disturbed conditions above a given fully disturbed limit,
    undisturbed conditions below a given undisturbed limit, and logarithmic
    decaying disturbance between the limits.

    Parameters
    ----------
    unit_params : dataframe
        Has the columns 'Sigci', 'GSI', 'mi', 'UnitWeight',
        'OverallSlopeAngle_deg'.
    sig_3 : array-like
        sigma 3 values for which to calculate shear/normal pairs.

    Returns
    -------
    trans_shear : numpy array
        shear values for given inputs.
    trans_normal : numpy array
        shear values for given inputs.
    hb_df : dataframe
        with shear, normal, sig3, disturbance values.

    """
    unit_weight = unit_params['UnitWeight']
    fdl = unit_params.FullyDisturbedLimit_m
    udl = unit_params.UndisturbedLimit_m
    full_d = unit_params.FullyDistD

    c2alpha = np.cos(np.radians(unit_params.OverallSlopeAngle_deg))**2

    hb_df = pd.DataFrame(data=sig_3, columns=['sigma_3'])

    sigma3_at_fdl = est_sig3_at_depth(unit_params, fdl, full_d, sig_3)
    sigma3_at_udl = est_sig3_at_depth(unit_params, udl, 0, sig_3)

    hb_df.loc[sig_3 <= sigma3_at_fdl, 'D'] = full_d
    hb_df.loc[sig_3 > sigma3_at_udl, 'D'] = 0

    # interpolate depth between fdl and udl
    depth = fdl+((sig_3-sigma3_at_fdl)/(sigma3_at_udl-sigma3_at_fdl))*(udl-fdl)
    hb_df = hb_df.assign(depth=depth)

    # use depth to calc decay and D. Otherwise, at some low depths
    # sigma3 at fdl is negative, and log(negative) is undefined, so the Rose
    # method won't work.

    if MODE == "ROSE":
        decay = -np.log(fdl/udl)
        hb_df.loc[(sig_3 <= sigma3_at_udl) &
                  (sig_3 > sigma3_at_fdl),
                  'D'] = (np.log(udl/hb_df.depth))/decay*full_d

    if MODE == "GALLANT":
        hb_df.loc[depth > 0, 'D'] = 2*full_d/(1+np.exp(5*hb_df.depth/udl))

    if MODE == "LINEAR":
        hb_df.loc[(sig_3 <= sigma3_at_udl) &
                  (sig_3 > sigma3_at_fdl),
                  'D'] = full_d+(hb_df.depth-fdl)*(0-full_d)/(udl-fdl)

    trans_shear, trans_normal = hb_to_shear_normal(unit_params, sig_3, hb_df.D)
    hb_df = hb_df.assign(shear=trans_shear)
    hb_df = hb_df.assign(normal=trans_normal)

    hb_df = hb_df.assign(depth=hb_df.normal/unit_weight/(c2alpha))
    return trans_shear, trans_normal, hb_df


def plot_disturbance_limits(unit_params, axis):
    unit_weight = unit_params['UnitWeight']
    fdl = unit_params.FullyDisturbedLimit_m
    udl = unit_params.UndisturbedLimit_m
    cos_alpha = np.cos(np.radians(unit_params.OverallSlopeAngle_deg))

    sat_constant = 1-unit_params.SaturationRatio * \
        0.009807/(unit_weight*cos_alpha**2)

    norm_at_limit = udl*unit_weight*cos_alpha**2*sat_constant
    axis.axvline(norm_at_limit, color="k",
                 alpha=0.5, ls="--", lw=1, label=None)

    norm_text = f"UDL: \n{udl:.0f} m"
    axis.text(norm_at_limit+0.05, 0.01, norm_text, fontsize=8, va="bottom",
              rotation=0, color="k", weight="bold", alpha=0.5, zorder=9)

    if MODE in ("ROSE", "LINEAR"):
        norm_at_limit = fdl*unit_weight*cos_alpha**2*sat_constant
        axis.axvline(norm_at_limit, color="k",
                     alpha=0.5, ls="--", lw=1, label=None)

        norm_text = f"FDL: {fdl:.0f} m"
        ax.text(norm_at_limit+0.05, 0.01, norm_text, fontsize=8, va="bottom",
                rotation=0, color="k", weight="bold", alpha=0.5, zorder=9)


# main
df = pd.read_csv(INPUT_PATH)
df = pd.DataFrame(df)
sigma_3 = np.arange(-1, 5, 0.0025)
fig = plt.figure(figsize=(15, 10))
FIG_COLS = 6
fig_rows = int(np.ceil(len(df)/FIG_COLS))+1  # add enough rows to accom. units

for unit in df.index[0:3]:
    unit_df = df.iloc[unit]

    # plot fully disturbed and undisturbed curves
    ax = fig.add_subplot(fig_rows, FIG_COLS, unit+1)
    plot_hb_curve(unit_df, 0, ax, 'D = 0')
    plot_hb_curve(unit_df, unit_df.FullyDistD,
                  ax, 'D = {}'.format(unit_df.FullyDistD), lstyle='dashed')

    # plot transition curve
    shear, normal, trans_df = trans_curve(unit_df, sigma_3)
    ax.plot(normal, shear, 'k', linestyle='dotted',
            label='Transitional D', linewidth=2)

    # plot disturbance limit normal stresses
    plot_disturbance_limits(unit_df, ax)

    # axis layout
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 6)
    ax.set_title(df.GeotechnicalUnit[unit].upper())
    ax.set_xlabel('Normal Stress (MPa)')
    ax.set_ylabel('Shear Stress (MPa)')
    ax.set_xticks(np.arange(0, 5.1, 1))
    ax.set_yticks(np.arange(0, 6.1, 1))
    ax.legend(loc='upper left')
    ax.set(aspect='equal')

    # export transition shear normal .csv for use in other programs (in kPa)
    shear_normal_pairs_kpa = trans_df[['normal', 'shear']][::5]*1000
    shear_normal_pairs_kpa = shear_normal_pairs_kpa.dropna()
    shear_normal_pairs_kpa.to_csv(
        OUTPUT_PATH+df.GeotechnicalUnit[unit]+'_sn.csv', index=False)

plt.tight_layout()
fig.savefig(OUTPUT_PATH + r'Summary.svg', bbox_inches='tight')
plt.show()
plt.close('all')
