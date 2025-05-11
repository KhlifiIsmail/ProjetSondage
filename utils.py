import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def simple_random_sample(df, sample_size):
    """
    Generate a simple random sample without replacement from the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The sampling frame
    sample_size : int
        The desired sample size
        
    Returns:
    --------
    pandas.DataFrame
        The sampled dataframe
    """
    if sample_size > len(df):
        return df.copy()
    
    sample_indices = np.random.choice(
        df.index, 
        size=sample_size, 
        replace=False
    )
    
    return df.loc[sample_indices].reset_index(drop=True)

def stratified_sample(df, strat_var, sample_size, aux_var=None):
    """
    Generate a stratified sample with proportional allocation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The sampling frame
    strat_var : str
        The stratification variable name
    sample_size : int
        The desired sample size
    aux_var : str, optional
        Auxiliary variable for post-stratification
        
    Returns:
    --------
    tuple
        (sample_df, allocation_df, strata_stats_df)
    """
    # Calculate stratum sizes and proportions
    strata_counts = df[strat_var].value_counts()
    total_size = len(df)
    
    # Calculate proportional allocation
    allocation = pd.DataFrame({
        'Strate': strata_counts.index,
        'Taille de la strate': strata_counts.values,
        'Proportion (%)': (strata_counts.values / total_size * 100).round(2),
        'Allocation (nh)': np.round(sample_size * strata_counts.values / total_size).astype(int)
    })
    
    # Ensure the sum of allocated samples equals the requested sample size
    if allocation['Allocation (nh)'].sum() != sample_size:
        diff = sample_size - allocation['Allocation (nh)'].sum()
        # Add or subtract the difference to/from the largest stratum
        idx = allocation['Taille de la strate'].idxmax()
        allocation.loc[idx, 'Allocation (nh)'] += diff
    
    # Draw the sample from each stratum
    sample_data = pd.DataFrame()
    strata_stats = []
    
    for stratum, nh in zip(allocation['Strate'], allocation['Allocation (nh)']):
        # Get the stratum data
        stratum_data = df[df[strat_var] == stratum]
        
        # Skip if the stratum is empty or allocation is zero
        if len(stratum_data) == 0 or nh == 0:
            continue
        
        # Sample from the stratum
        if nh <= len(stratum_data):
            stratum_sample = stratum_data.sample(n=nh, replace=False)
        else:
            # If requested sample size is larger than stratum size, take all
            stratum_sample = stratum_data
        
        # Append to the final sample
        sample_data = pd.concat([sample_data, stratum_sample])
        
        # Calculate stratum statistics
        if aux_var and aux_var in stratum_data.columns:
            aux_mean_frame = stratum_data[aux_var].mean() if pd.api.types.is_numeric_dtype(stratum_data[aux_var]) else None
            aux_mean_sample = stratum_sample[aux_var].mean() if pd.api.types.is_numeric_dtype(stratum_sample[aux_var]) else None
            
            strata_stats.append({
                'Strate': stratum,
                'Taille': len(stratum_data),
                'Échantillon': len(stratum_sample),
                'Sampling Fraction': len(stratum_sample) / len(stratum_data),
                'Auxiliary Mean (Frame)': aux_mean_frame,
                'Auxiliary Mean (Sample)': aux_mean_sample,
                'Difference (%)': ((aux_mean_sample - aux_mean_frame) / aux_mean_frame * 100).round(2) if aux_mean_frame else None
            })
    
    # Reset index
    sample_data = sample_data.reset_index(drop=True)
    
    # Create stratum statistics dataframe
    strata_stats_df = pd.DataFrame(strata_stats) if strata_stats else None
    
    return sample_data, allocation, strata_stats_df

def calculate_descriptive_stats(df):
    """
    Calculate descriptive statistics for numeric columns in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset
        
    Returns:
    --------
    pandas.DataFrame
        Descriptive statistics
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 0:
        stats_df = df[numeric_cols].describe().T
        
        # Add additional statistics
        stats_df['cv'] = stats_df['std'] / stats_df['mean']  # Coefficient of variation
        stats_df['skewness'] = df[numeric_cols].skew()
        stats_df['kurtosis'] = df[numeric_cols].kurtosis()
        stats_df['sum'] = df[numeric_cols].sum()
        
        return stats_df.round(4)
    else:
        return pd.DataFrame()

def compare_categorical_distributions(frame_df, sample_df, var_name):
    """
    Compare the distribution of a categorical variable between the frame and sample.
    
    Parameters:
    -----------
    frame_df : pandas.DataFrame
        The sampling frame
    sample_df : pandas.DataFrame
        The sample
    var_name : str
        The variable name to compare
        
    Returns:
    --------
    tuple
        (comparison_df, matplotlib.figure.Figure)
    """
    # Calculate proportions
    frame_counts = frame_df[var_name].value_counts(normalize=True)
    sample_counts = sample_df[var_name].value_counts(normalize=True)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Modalité': frame_counts.index,
        'Proportion dans le cadre (%)': (frame_counts * 100).values,
        'Proportion dans l\'échantillon (%)': [
            (sample_counts[val] * 100) if val in sample_counts else 0 
            for val in frame_counts.index
        ]
    })
    
    # Calculate absolute and relative differences
    comparison_df['Différence (pts)'] = (
        comparison_df['Proportion dans l\'échantillon (%)'] - 
        comparison_df['Proportion dans le cadre (%)']
    ).round(2)
    
    comparison_df['Différence relative (%)'] = (
        comparison_df['Différence (pts)'] / 
        comparison_df['Proportion dans le cadre (%)'] * 100
    ).round(2)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the comparison
    x = range(len(comparison_df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], comparison_df['Proportion dans le cadre (%)'], 
           width=width, label='Cadre', color='#1E88E5')
    ax.bar([i + width/2 for i in x], comparison_df['Proportion dans l\'échantillon (%)'], 
           width=width, label='Échantillon', color='#FFC107')
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Modalité'], rotation=45, ha='right')
    ax.set_ylabel('Proportion (%)')
    ax.set_title(f'Comparaison des proportions pour {var_name}')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return comparison_df, fig

def plot_strata_allocation(allocation_df):
    """
    Create plots showing the stratum sizes and allocations.
    
    Parameters:
    -----------
    allocation_df : pandas.DataFrame
        The allocation dataframe
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure with stratum allocation visualization
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Taille des strates dans le cadre",
            "Allocation de l'échantillon"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add traces for stratum sizes
    fig.add_trace(
        go.Bar(
            x=allocation_df['Strate'],
            y=allocation_df['Taille de la strate'],
            name='Taille des strates',
            marker_color='rgba(30, 136, 229, 0.8)'
        ),
        row=1, col=1
    )
    
    # Add traces for allocations
    fig.add_trace(
        go.Bar(
            x=allocation_df['Strate'],
            y=allocation_df['Allocation (nh)'],
            name='Allocation (nh)',
            marker_color='rgba(255, 193, 7, 0.8)'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text="Allocation proportionnelle par strate"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Strate")
    fig.update_yaxes(title_text="Nombre d'unités", row=1, col=1)
    fig.update_yaxes(title_text="Taille de l'échantillon", row=1, col=2)
    
    return fig

def calculate_confidence_intervals(sample_df, numeric_vars, confidence_level=95):
    """
    Calculate confidence intervals for the means of numeric variables.
    
    Parameters:
    -----------
    sample_df : pandas.DataFrame
        The sample dataframe
    numeric_vars : list
        List of numeric variable names
    confidence_level : int
        Confidence level (in percent)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with confidence intervals
    """
    # Initialize results
    results = []
    
    # Critical value for the confidence level
    alpha = 1 - confidence_level / 100
    z_critical = stats.norm.ppf(1 - alpha / 2)
    
    for var in numeric_vars:
        # Get variable data
        data = sample_df[var].dropna()
        
        # Calculate statistics
        n = len(data)
        mean = data.mean()
        std = data.std()
        se = std / np.sqrt(n)
        
        # Calculate confidence interval
        margin_error = z_critical * se
        lower_ci = mean - margin_error
        upper_ci = mean + margin_error
        
        # Add to results
        results.append({
            'Variable': var,
            'Moyenne': mean,
            'Écart-type': std,
            'Erreur-type': se,
            'Marge d\'erreur': margin_error,
            'IC Inf': lower_ci,
            'IC Sup': upper_ci,
            'Précision (%)': (margin_error / mean * 100).round(2) if mean != 0 else np.nan
        })
    
    # Create dataframe
    ci_df = pd.DataFrame(results).set_index('Variable')
    
    return ci_df.round(4)

def generate_sampling_diagnostics(frame_df, sample_df, strat_var=None):
    """
    Generate sampling diagnostics comparing the sample to the frame.
    
    Parameters:
    -----------
    frame_df : pandas.DataFrame
        The sampling frame
    sample_df : pandas.DataFrame
        The sample
    strat_var : str, optional
        Stratification variable if using stratified sampling
        
    Returns:
    --------
    dict
        Dictionary with diagnostic metrics and visualizations
    """
    # Initialize results
    diagnostics = {}
    
    # Sample size and sampling rate
    diagnostics['sample_size'] = len(sample_df)
    diagnostics['frame_size'] = len(frame_df)
    diagnostics['sampling_rate'] = len(sample_df) / len(frame_df)
    
    # Calculate design effect if stratified
    if strat_var:
        # Get stratum proportions
        strata_props = frame_df[strat_var].value_counts(normalize=True)
        sample_props = sample_df[strat_var].value_counts(normalize=True)
        
        # Calculate design effect (approximation)
        design_effect = sum((sample_props / strata_props) ** 2 * strata_props)
        diagnostics['design_effect'] = design_effect
        
        # Effective sample size
        diagnostics['effective_sample_size'] = len(sample_df) / design_effect
    
    # Representativeness tests for categorical variables
    cat_vars = [
        col for col in frame_df.columns 
        if pd.api.types.is_categorical_dtype(frame_df[col]) or 
           pd.api.types.is_object_dtype(frame_df[col]) or
           len(frame_df[col].unique()) <= 15
    ]
    
    chi2_tests = {}
    for var in cat_vars:
        # Calculate proportions
        frame_props = frame_df[var].value_counts(normalize=True)
        sample_props = sample_df[var].value_counts(normalize=True)
        
        # Align indices
        common_categories = list(set(frame_props.index) & set(sample_props.index))
        
        if len(common_categories) > 1:  # Need at least 2 categories for chi-square test
            aligned_frame_props = frame_props.loc[common_categories]
            aligned_sample_props = sample_props.loc[common_categories]
            
            # Chi-square test
            try:
                chi2, p_value = stats.chisquare(
                    aligned_sample_props * 100,
                    aligned_frame_props * 100
                )
                
                chi2_tests[var] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'representative': p_value > 0.05
                }
            except:
                chi2_tests[var] = {
                    'chi2': np.nan,
                    'p_value': np.nan,
                    'representative': None,
                    'error': 'Could not perform chi-square test'
                }
    
    diagnostics['representativeness_tests'] = chi2_tests
    
    # Numeric variable comparisons
    num_vars = frame_df.select_dtypes(include=['int64', 'float64']).columns
    
    numeric_comparisons = {}
    for var in num_vars:
        frame_mean = frame_df[var].mean()
        sample_mean = sample_df[var].mean()
        
        # Relative difference
        rel_diff = (sample_mean - frame_mean) / frame_mean * 100 if frame_mean != 0 else np.nan
        
        # T-test
        try:
            t_stat, p_value = stats.ttest_1samp(sample_df[var], frame_mean)
            
            numeric_comparisons[var] = {
                'frame_mean': frame_mean,
                'sample_mean': sample_mean,
                'abs_diff': sample_mean - frame_mean,
                'rel_diff_pct': rel_diff,
                't_stat': t_stat,
                'p_value': p_value,
                'representative': p_value > 0.05
            }
        except:
            numeric_comparisons[var] = {
                'frame_mean': frame_mean,
                'sample_mean': sample_mean,
                'abs_diff': sample_mean - frame_mean,
                'rel_diff_pct': rel_diff,
                't_stat': np.nan,
                'p_value': np.nan,
                'representative': None,
                'error': 'Could not perform t-test'
            }
    
    diagnostics['numeric_comparisons'] = numeric_comparisons
    
    # Overall representativeness score (percentage of variables that are representative)
    cat_rep_count = sum(1 for test in chi2_tests.values() if test.get('representative', False))
    num_rep_count = sum(1 for test in numeric_comparisons.values() if test.get('representative', False))
    
    total_tests = len(chi2_tests) + len(numeric_comparisons)
    rep_score = (cat_rep_count + num_rep_count) / total_tests * 100 if total_tests > 0 else np.nan
    
    diagnostics['representativeness_score'] = rep_score
    
    return diagnostics

def create_distribution_plot(frame_df, sample_df, var_name):
    """
    Create a distribution comparison plot for a numeric variable.
    
    Parameters:
    -----------
    frame_df : pandas.DataFrame
        The sampling frame
    sample_df : pandas.DataFrame
        The sample
    var_name : str
        The variable name to plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure with distribution comparison
    """
    # Check if variable is numeric
    if not pd.api.types.is_numeric_dtype(frame_df[var_name]):
        raise ValueError(f"Variable {var_name} is not numeric")
    
    # Create figure
    fig = go.Figure()
    
    # Add distributions
    fig.add_trace(go.Histogram(
        x=frame_df[var_name],
        name='Cadre',
        opacity=0.6,
        marker_color='rgba(30, 136, 229, 0.8)',
        histnorm='probability',
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=sample_df[var_name],
        name='Échantillon',
        opacity=0.6,
        marker_color='rgba(255, 193, 7, 0.8)',
        histnorm='probability',
        nbinsx=30
    ))
    
    # Add vertical lines for means
    frame_mean = frame_df[var_name].mean()
    sample_mean = sample_df[var_name].mean()
    
    fig.add_vline(
        x=frame_mean,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Moyenne cadre: {frame_mean:.2f}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=sample_mean,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Moyenne échantillon: {sample_mean:.2f}",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title=f"Distribution comparative de {var_name}",
        xaxis_title=var_name,
        yaxis_title="Densité de probabilité",
        barmode='overlay',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def generate_bivariate_analysis(sample_df, var1, var2, categorical=False):
    """
    Generate bivariate analysis between two variables.
    
    Parameters:
    -----------
    sample_df : pandas.DataFrame
        The sample dataframe
    var1 : str
        First variable name
    var2 : str
        Second variable name
    categorical : bool
        Whether to treat variables as categorical
        
    Returns:
    --------
    tuple
        (fig, stats_dict) with visualization and statistics
    """
    results = {}
    
    # Check if both variables are categorical or should be treated as categorical
    if categorical or (
        (pd.api.types.is_categorical_dtype(sample_df[var1]) or pd.api.types.is_object_dtype(sample_df[var1]) or len(sample_df[var1].unique()) <= 15) and
        (pd.api.types.is_categorical_dtype(sample_df[var2]) or pd.api.types.is_object_dtype(sample_df[var2]) or len(sample_df[var2].unique()) <= 15)
    ):
        # Create contingency table
        contingency = pd.crosstab(
            sample_df[var1], 
            sample_df[var2], 
            normalize='all'
        ) * 100
        
        # Chi-square test of independence
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Cramer's V (measure of association)
        n = len(sample_df)
        phi2 = chi2 / n
        r, k = contingency.shape
        cramers_v = np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else np.nan
        
        # Create heatmap visualization
        fig = px.imshow(
            contingency,
            text_auto='.1f',
            labels=dict(x=var2, y=var1, color="Pourcentage (%)"),
            title=f"Relation entre {var1} et {var2}",
            color_continuous_scale="Greens"
        )
        
        fig.update_layout(height=500)
        
        results = {
            'type': 'categorical',
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': cramers_v,
            'independent': p_value > 0.05,
            'association_strength': 'Aucune' if cramers_v < 0.1 else 'Faible' if cramers_v < 0.3 else 'Modérée' if cramers_v < 0.5 else 'Forte'
        }
        
    # If both variables are numeric
    elif pd.api.types.is_numeric_dtype(sample_df[var1]) and pd.api.types.is_numeric_dtype(sample_df[var2]):
        # Calculate correlation
        pearson_corr, p_value = stats.pearsonr(sample_df[var1], sample_df[var2])
        spearman_corr, spearman_p = stats.spearmanr(sample_df[var1], sample_df[var2])
        
        # Create scatter plot with regression line
        fig = px.scatter(
            sample_df,
            x=var1,
            y=var2,
            trendline="ols",
            labels={var1: var1, var2: var2},
            title=f"Relation entre {var1} et {var2} (r = {pearson_corr:.2f})"
        )
        
        # Add confidence bands
        try:
            # Fit OLS model for prediction interval
            X = sm.add_constant(sample_df[var1])
            model = sm.OLS(sample_df[var2], X).fit()
            
            # Generate predictions with confidence intervals
            x_range = np.linspace(sample_df[var1].min(), sample_df[var1].max(), 100)
            X_pred = sm.add_constant(x_range)
            
            # Get prediction and confidence intervals
            preds = model.get_prediction(X_pred)
            pred_mean = preds.predicted_mean
            pred_ci = preds.conf_int(alpha=0.05)  # 95% confidence interval
            
            # Add confidence bands
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate([pred_ci[:, 0], pred_ci[:, 1][::-1]]),
                fill='toself',
                fillcolor='rgba(30, 136, 229, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='95% Confidence Interval'
            ))
        except:
            pass  # Skip confidence bands if there's an error
        
        fig.update_layout(height=500)
        
        results = {
            'type': 'numeric',
            'pearson_corr': pearson_corr,
            'pearson_p': p_value,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'significant': p_value < 0.05,
            'correlation_strength': 'Aucune' if abs(pearson_corr) < 0.1 else 'Faible' if abs(pearson_corr) < 0.3 else 'Modérée' if abs(pearson_corr) < 0.7 else 'Forte'
        }
    
    # If one variable is categorical and one is numeric
    else:
        # Identify which variable is categorical and which is numeric
        if pd.api.types.is_categorical_dtype(sample_df[var1]) or pd.api.types.is_object_dtype(sample_df[var1]) or len(sample_df[var1].unique()) <= 15:
            cat_var = var1
            num_var = var2
        else:
            cat_var = var2
            num_var = var1
        
        # Perform one-way ANOVA
        categories = sample_df[cat_var].unique()
        data_by_category = [sample_df[sample_df[cat_var] == cat][num_var].dropna() for cat in categories]
        
        # Only perform ANOVA if there are at least 2 categories with data
        valid_categories = [data for data in data_by_category if len(data) > 0]
        
        if len(valid_categories) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*valid_categories)
                eta_squared = float('nan')  # Placeholder, calculating eta squared properly requires more work
                
                # Calculate eta squared (effect size)
                ss_between = sum(len(data) * ((data.mean() - sample_df[num_var].mean()) ** 2) for data in valid_categories)
                ss_total = sum((sample_df[num_var] - sample_df[num_var].mean()) ** 2)
                
                if ss_total > 0:
                    eta_squared = ss_between / ss_total
            except:
                f_stat, p_value, eta_squared = float('nan'), float('nan'), float('nan')
        else:
            f_stat, p_value, eta_squared = float('nan'), float('nan'), float('nan')
        
        # Create boxplot
        fig = px.box(
            sample_df,
            x=cat_var,
            y=num_var,
            points="all",
            title=f"Distribution de {num_var} par {cat_var}"
        )
        
        # Add mean lines
        for cat in categories:
            cat_data = sample_df[sample_df[cat_var] == cat][num_var]
            if len(cat_data) > 0:
                cat_mean = cat_data.mean()
                
                fig.add_shape(
                    type="line",
                    x0=cat,
                    y0=cat_mean,
                    x1=cat,
                    y1=cat_mean,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    )
                )
        
        fig.update_layout(height=500)
        
        results = {
            'type': 'mixed',
            'categorical_var': cat_var,
            'numeric_var': num_var,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05 if not pd.isna(p_value) else None,
            'effect_size': 'Aucun' if eta_squared < 0.01 else 'Faible' if eta_squared < 0.06 else 'Modéré' if eta_squared < 0.14 else 'Fort'
        }
    
    return fig, results