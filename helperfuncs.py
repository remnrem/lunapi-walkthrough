
# ------------------------------------------------------------                                                                                           # some simple helper functions for the lunapi-walkthrough

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------------------------------------------------------
# convert all columns that can be converted to numeric

def safe_to_numeric(df):
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='coerce')            
            if not converted.isna().all():
                df[col] = converted
        except Exception:
            pass
    return df


# ------------------------------------------------------------
# replace outliers with NaN

def outliers(x, t=3):
    m = np.nanmean(x)
    sdev = np.nanstd(x)
    lwr = m - t * sdev
    upr = m + t * sdev
    x = x.copy()
    x[(x < lwr) | (x > upr)] = np.nan
    return x


# ------------------------------------------------------------
# regress Y ~ age + sex
# requires those variables present in passed df

def fit_lm(v,df):
    df = df.copy()
    df.loc[:, 'DV'] = outliers(df[v])
    model = smf.ols('DV ~ age + male', data=df).fit()
    return model

# ------------------------------------------------------------
# fit lm for multiple DVs
# returns a table of betas & p-values for the sex effect

def fit_lms(vars,df):
    vars = vars if isinstance(vars, list) else [vars]
    rows = []
    for v in vars:
        model = fit_lm(v,df)
        beta = model.params['male']
        pval = model.pvalues['male']
        rows.append( [v,beta,pval] )
    return pd.DataFrame(rows, columns=['Var', 'Beta', 'p-value'])


# ------------------------------------------------------------
# heatmap


def heatmap( x, y, z, col=None, mt="", f=None, zero=None,
             zlim=None, legend=None , useRaster=True ):

    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if col is None: col = plt.cm.turbo
    if zero is not None: z[zero] = 0
    if f is None: f = np.ones_like(z, dtype=bool)
    x = x[f]
    y = y[f]
    z = z[f]

    # Validate square data
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    nz = len(z)
    if nz == 0: raise ValueError("No data to plot")
    if nz != nx * ny: raise ValueError("Requires square data")

    # Sort data by y then x
    idx = np.lexsort((x, y))
    x = x[idx]
    y = y[idx]
    z = z[idx]

    # Compute z limits
    if zlim is None: zlim = [np.nanmin(z), np.nanmax(z)]

    # Convert to 2D matrix
    m = np.array(z).reshape((ny, nx))

    # Plot heatmap
    plt.figure(figsize=(14, 3))  # (width, height) in inches
    plt.imshow(
        m, origin='lower', cmap=col,
        aspect='auto', interpolation='nearest' if useRaster else 'none',
        extent=[min(x), max(x), min(y), max(y)],
        norm=Normalize(vmin=zlim[0], vmax=zlim[1])
    )
    plt.title(mt)
    plt.xticks([])
    plt.yticks([])

    # Return legend quantiles if requested
    if legend is not None:
        if not isinstance(legend, int):
            raise ValueError("legend should be an integer")
        q = np.quantile(z, np.linspace(0, 1, legend))
        return q

    plt.colorbar(label='z')
    plt.tight_layout()
    plt.show()

    
# ------------------------------------------------------------
# t-tests for each 'F'
# assumes : F, 'male' and some DV value 'dv'
def ttests_by_F( dv , df ):
    import pandas as pd
    import numpy as np
    from scipy.stats import ttest_ind

    results = []

    for f_val in df['F'].unique():
        subset = df[df['F'] == f_val]
        psd_male = subset[subset['male'] == 1][dv]
        psd_female = subset[subset['male'] == 0][dv]
    
        # Perform independent t-test
        tstat, pval = ttest_ind(psd_female, psd_male, equal_var=False)
    
        # Signed -log10(p)
        sign = np.sign(tstat)  
        signed_logp = -np.log10(pval) * sign
        results.append({'F': f_val, 'signed_log10_p': signed_logp})

    # Create result DataFrame
    t_table = pd.DataFrame(results).sort_values(by='F')

    return t_table

        
