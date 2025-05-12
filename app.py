import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import sampling utilities
from utils import (
    simple_random_sample, 
    stratified_sample,
    calculate_descriptive_stats,
    compare_categorical_distributions,
    plot_strata_allocation,
    generate_sampling_diagnostics,
    calculate_confidence_intervals
)

# Function to check if a column is categorical-like
def is_categorical_like(series):
    """Check if a series is categorical-like (categorical, object, or few unique values)"""
    return (isinstance(series.dtype, pd.CategoricalDtype) or 
            pd.api.types.is_object_dtype(series) or 
            len(series.unique()) <= 15)

# Page configuration and theme
st.set_page_config(
    page_title="Théorie de Sondage | Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    /* Main elements */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Container styling */
    .css-1vq4p4l {
        padding: 1rem 1rem 10rem;
    }
    
    /* Custom header */
    .custom-header {
        background-color: #4CAF50;
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* App title */
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle */
    .app-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Method cards */
    .method-card {
        background-color: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #4CAF50;
    }
    
    /* Method title */
    .method-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #1B5E20;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E0E0E0;
    }
    
    /* Footer */
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #2E7D32;
        color: white;
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        padding-left: 450px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 0.35rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #2E7D32;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E8F5E9;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: #2E7D32;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Results area */
    .results-area {
        background-color: #F1F8E9;
        border-radius: 0.5rem;
        padding: 1rem;
        border: 1px solid #C8E6C9;
    }
    
    /* Data metrics */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #757575;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        margin: 1rem 0;
    }
    
    /* Tables */
    div[data-testid="stTable"] {
        border: 1px solid #E0E0E0;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Sidebar */
    .css-1aumxhk {
        background-color: #E8F5E9;
    }
    
    /* Tooltip styling */
    div[data-baseweb="tooltip"] {
        background-color: #424242;
        color: white;
        border-radius: 0.25rem;
        padding: 0.5rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a download link for dataframes
def get_excel_download_link(df, filename, button_text):
    """Generate a download link for a dataframe as Excel file"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">{button_text}</a>'
    return href

# Function to add color to dataframe
def style_dataframe(df):
    """Add styling to dataframe"""
    # Add a style with bar charts for numerical columns
    if df.select_dtypes(include=['float64', 'int64']).columns.any():
        styled_df = df.style.background_gradient(cmap='Greens', low=0.5, high=1.0, subset=df.select_dtypes(include=['float64', 'int64']).columns)
        return styled_df
    return df

# Application Header
def display_header():
    """Display the application header"""
    st.markdown("""
    <div class="custom-header">
        <div class="app-title">Application de Théorie de Sondage</div>
        <div class="app-subtitle">École supérieure de la statistique et l'Analyse de l'information | 2024-2025</div>
    </div>
    """, unsafe_allow_html=True)

# Application Sidebar
def display_sidebar():
    """Display the application sidebar with file upload and settings"""
    st.sidebar.image("Logo_ESSAIT.png", width=200)
    st.sidebar.markdown("### Paramètres de l'Application")
    
    # File upload
    st.sidebar.markdown("#### Cadre d'échantillonnage")
    uploaded_file = st.sidebar.file_uploader("📂 Télécharger le fichier Excel", type=['xlsx', 'xls'])
    
    # Settings after file upload
    sampling_method = None
    sampling_frame = None
    
    if uploaded_file is not None:
        try:
            sampling_frame = pd.read_excel(uploaded_file)
            st.sidebar.success(f"✅ Fichier chargé avec succès! ({sampling_frame.shape[0]} blocs)")
            
            # Display basic info about the frame
            with st.sidebar.expander("📊 Informations sur le cadre"):
                st.write(f"**Nombre d'observations:** {sampling_frame.shape[0]}")
                st.write(f"**Nombre de variables:** {sampling_frame.shape[1]}")
                st.write(f"**Variables disponibles:** {', '.join(sampling_frame.columns)}")
            
            # Method selection
            st.sidebar.markdown("#### Méthode d'échantillonnage")
            sampling_method = st.sidebar.radio(
                "Sélectionner une méthode:",
                ["Aléatoire Simple sans Remise (SAS)", 
                 "Stratification à Allocation Proportionnelle"]
            )
            
            # Settings section
            st.sidebar.markdown("#### Paramètres généraux")
            seed = st.sidebar.number_input("Graine aléatoire (seed)", min_value=1, max_value=9999, value=42)
            np.random.seed(seed)
            
            # Color scheme selection
            st.sidebar.markdown("#### Personnalisation")
            color_scheme = st.sidebar.selectbox(
                "Palette de couleurs",
                ["Vert (défaut)", "Bleu", "Violet", "Orange"],
            )
            
            # Map color selections to actual color schemes
            color_map = {
                "Vert (défaut)": ["#4CAF50", "#E8F5E9", "#2E7D32"],
                "Bleu": ["#2196F3", "#E3F2FD", "#0D47A1"],
                "Violet": ["#9C27B0", "#F3E5F5", "#6A1B9A"],
                "Orange": ["#FF9800", "#FFF3E0", "#E65100"]
            }
            
            # Update colors based on selection
            if color_scheme in color_map:
                primary_color, bg_color, dark_color = color_map[color_scheme]
                set_color_scheme(primary_color, bg_color, dark_color)
            
        except Exception as e:
            st.sidebar.error(f"Erreur lors du chargement du fichier: {e}")
    
    # About the app
    with st.sidebar.expander("ℹ️ À propos de l'application"):
        st.write("""
        Cette application permet de tirer des échantillons selon deux méthodes différentes et d'analyser leur représentativité par rapport au cadre d'échantillonnage.
        
        Pour tout problème ou suggestion, veuillez contacter l'équipe de développement.
        """)
    
    return uploaded_file, sampling_method, sampling_frame

def set_color_scheme(primary_color, bg_color, dark_color):
    """Set color scheme using CSS variables"""
    st.markdown(f"""
    <style>
        :root {{
            --primary-color: {primary_color};
            --background-color: {bg_color};
            --dark-color: {dark_color};
        }}
        
        .custom-header {{
            background-color: var(--primary-color);
        }}
        
        .method-card {{
            border-left: 4px solid var(--primary-color);
        }}
        
        .method-title {{
            color: var(--dark-color);
        }}
        
        .section-header {{
            color: var(--dark-color);
        }}
        
        .app-footer {{
            background-color: var(--dark-color);
        }}
        
        .stButton>button {{
            background-color: var(--primary-color);
        }}
        
        .stButton>button:hover {{
            background-color: var(--dark-color);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background-color: var(--background-color);
            color: var(--dark-color);
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: var(--primary-color) !important;
        }}
        
        .metric-value {{
            color: var(--dark-color);
        }}
        
        .results-area {{
            background-color: var(--background-color);
            border: 1px solid var(--primary-color);
        }}
    </style>
    """, unsafe_allow_html=True)

# Display overview page when no file is uploaded
def display_welcome_page():
    """Display welcome page with instructions and overview"""
    st.markdown("""
    <div class="method-card">
        <div class="method-title">Bienvenue dans l'Application de Théorie de Sondage</div>
        <p>Cette application vous permet de tirer automatiquement des échantillons selon différentes méthodes statistiques et d'analyser leur représentativité.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview of methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="method-card">
            <div class="method-title">Aléatoire Simple sans Remise (SAS)</div>
            <p>Cette méthode permet de tirer un échantillon où chaque unité du cadre a la même probabilité d'être sélectionnée. Elle est simple mais efficace pour les populations homogènes.</p>
            <div class="section-header">Fonctionnalités</div>
            <ul>
                <li>Tirage aléatoire sans remise</li>
                <li>Statistiques descriptives</li>
                <li>Tableau comparatif échantillon-cadre</li>
                <li>Visualisations interactives</li>
                <li>Analyse de représentativité</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="method-card">
            <div class="method-title">Stratification à Allocation Proportionnelle</div>
            <p>Cette méthode divise la population en strates homogènes et tire un échantillon proportionnel à la taille de chaque strate, améliorant ainsi la précision des estimations.</p>
            <div class="section-header">Fonctionnalités</div>
            <ul>
                <li>Allocation proportionnelle par strate</li>
                <li>Tableau des allocations</li>
                <li>Statistiques descriptives par strate</li>
                <li>Visualisations interactives</li>
                <li>Analyse comparative multi-variables</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started instructions
    st.markdown("""
    <div class="method-card">
        <div class="method-title">Pour commencer</div>
        <ol>
            <li>Téléchargez le cadre d'échantillonnage (fichier Excel) en utilisant le menu latéral</li>
            <li>Sélectionnez une méthode d'échantillonnage</li>
            <li>Configurez les paramètres selon vos besoins</li>
            <li>Générez l'échantillon et analysez les résultats</li>
            <li>Téléchargez les données et visualisations pour votre rapport</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Project information
    st.markdown("""
    <div class="info-box">
        <p><strong>Projet Théorie de sondage | 2ème année</strong><br>
        Cadre d'échantillonnage : Blocs conçus à partir du dernier recensement général de la population effectué par l'institut national de la statistique (INS).</p>
    </div>
    """, unsafe_allow_html=True)

# Simple Random Sampling module
def srs_sampling_module(sampling_frame):
    """Display the simple random sampling module"""
    st.markdown("""
    <div class="method-card">
        <div class="method-title">Méthode Aléatoire Simple sans Remise (SAS)</div>
        <p>Cette méthode donne à chaque unité du cadre la même probabilité d'être sélectionnée dans l'échantillon, sans qu'une unité puisse être sélectionnée deux fois.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout with two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample size input
        sample_size = st.number_input(
            "Taille de l'échantillon", 
            min_value=1, 
            max_value=sampling_frame.shape[0],
            value=min(int(sampling_frame.shape[0] * 0.1), 100),
            step=1
        )
        
        # Additional parameters
        confidence_level = st.slider(
            "Niveau de confiance (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
    
    with col2:
        # Comparison variable selection
        categorical_vars = [col for col in sampling_frame.columns 
                          if is_categorical_like(sampling_frame[col])]
        
        comparison_var = st.selectbox(
            "Variable comparative échantillon-cadre",
            options=categorical_vars if categorical_vars else sampling_frame.columns,
            help="Variable utilisée pour comparer la distribution entre l'échantillon et le cadre"
        )
        
        # Secondary comparison variable
        secondary_vars = [col for col in sampling_frame.columns if col != comparison_var]
        
        if secondary_vars:
            secondary_var = st.selectbox(
                "Variable comparative secondaire (optionnelle)",
                options=["Aucune"] + secondary_vars,
                help="Une seconde variable pour analyser la représentativité de l'échantillon"
            )
        else:
            secondary_var = "Aucune"
    
    # Execute SRS button
    if st.button("Générer l'échantillon aléatoire", key="generate_srs"):
        with st.spinner("Génération de l'échantillon en cours..."):
            # Draw the sample
            sample_data = simple_random_sample(sampling_frame, sample_size)
            
            # Display the sample in tabs
            tabs = st.tabs(["📋 Échantillon", "📊 Statistiques", "📈 Visualisations", "🔍 Analyse avancée"])
            
            # Tab 1: Sample Data
            with tabs[0]:
                st.markdown('<div class="section-header">Données de l\'échantillon</div>', unsafe_allow_html=True)
                
                # Sample metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Taille de l'échantillon", f"{sample_size:,}")
                col2.metric("Taux d'échantillonnage", f"{sample_size/sampling_frame.shape[0]:.2%}")
                col3.metric("Variables", f"{sample_data.shape[1]}")
                
                # Display sample data with download option
                st.dataframe(style_dataframe(sample_data))
                
                st.markdown(get_excel_download_link(sample_data, "echantillon_sas", "💾 Télécharger l'échantillon (Excel)"), unsafe_allow_html=True)
            
            # Tab 2: Statistics
            with tabs[1]:
                st.markdown('<div class="section-header">Statistiques descriptives</div>', unsafe_allow_html=True)
                
                # Numeric statistics
                numeric_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_cols) > 0:
                    # Calculate stats
                    stats_df = sample_data[numeric_cols].describe().T
                    stats_df['cv'] = stats_df['std'] / stats_df['mean']  # Coefficient of variation
                    stats_df = stats_df.round(2)
                    
                    # Add comparison with frame
                    frame_stats = sampling_frame[numeric_cols].describe().T
                    frame_stats = frame_stats.round(2)
                    
                    # Calculate differences
                    diff_df = pd.DataFrame({
                        'Variable': stats_df.index,
                        'Moyenne (Échantillon)': stats_df['mean'],
                        'Moyenne (Cadre)': frame_stats['mean'],
                        'Écart (%)': ((stats_df['mean'] - frame_stats['mean']) / frame_stats['mean'] * 100).round(2),
                        'Écart-type (Échantillon)': stats_df['std'],
                        'Écart-type (Cadre)': frame_stats['std'],
                        'Coef. Variation (Échantillon)': stats_df['cv']
                    })
                    
                    # Style and display
                    st.dataframe(style_dataframe(diff_df.set_index('Variable')))
                    
                    # Download link
                    st.markdown(get_excel_download_link(diff_df, "statistiques_sas", "💾 Télécharger les statistiques (Excel)"), unsafe_allow_html=True)
                else:
                    st.info("Aucune variable numérique trouvée pour les statistiques descriptives.")
                
                # Categorical statistics
                st.markdown('<div class="section-header">Distribution des variables catégorielles</div>', unsafe_allow_html=True)
                
                # Distribution of primary comparison variable
                comparison_df, _ = compare_categorical_distributions(sampling_frame, sample_data, comparison_var)
                
                # Display comparison dataframe
                st.dataframe(style_dataframe(comparison_df))
                
                # Chi-square test for goodness of fit
                from scipy import stats as scipy_stats  # Import here to avoid name conflict
                chi2, p_value = scipy_stats.chisquare(
                    comparison_df['Proportion dans l\'échantillon (%)'].values,
                    comparison_df['Proportion dans le cadre (%)'].values
                )
                
                col1, col2 = st.columns(2)
                col1.metric("Test Chi² de représentativité", f"{chi2:.2f}")
                col2.metric("Valeur p", f"{p_value:.4f}")
                
                if p_value > 0.05:
                    st.success("✅ La distribution de l'échantillon est représentative du cadre (p > 0.05)")
                else:
                    st.warning("⚠️ La distribution de l'échantillon n'est pas représentative du cadre (p < 0.05)")
            
            # Tab 3: Visualizations
            with tabs[2]:
                st.markdown('<div class="section-header">Visualisations interactives</div>', unsafe_allow_html=True)
                
                # Primary comparison variable visualization
                fig = make_subplots(rows=1, cols=2, 
                                   subplot_titles=[
                                       f'Distribution de {comparison_var}',
                                       'Comparaison échantillon-cadre'
                                   ],
                                   specs=[[{"type": "pie"}, {"type": "bar"}]])
                
                # Pie chart for sample distribution
                fig.add_trace(
                    go.Pie(
                        labels=comparison_df['Modalité'],
                        values=comparison_df['Proportion dans l\'échantillon (%)'],
                        name="Échantillon",
                        marker_colors=px.colors.qualitative.G10,
                        hole=0.3
                    ),
                    row=1, col=1
                )
                
                # Bar chart for comparison
                fig.add_trace(
                    go.Bar(
                        x=comparison_df['Modalité'],
                        y=comparison_df['Proportion dans le cadre (%)'],
                        name='Cadre',
                        marker_color='rgba(58, 71, 80, 0.6)'
                    ),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Bar(
                        x=comparison_df['Modalité'],
                        y=comparison_df['Proportion dans l\'échantillon (%)'],
                        name='Échantillon',
                        marker_color='rgba(246, 78, 139, 0.6)'
                    ),
                    row=1, col=2
                )
                
                # Update layout
                fig.update_layout(
                    height=500,
                    title_text=f"Distribution comparative de {comparison_var}",
                    barmode='group',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # If secondary variable is selected
                if secondary_var != "Aucune":
                    # Check if both variables are categorical
                    if secondary_var in categorical_vars and comparison_var in categorical_vars:
                        st.markdown(f'<div class="section-header">Analyse bivariée: {comparison_var} vs {secondary_var}</div>', unsafe_allow_html=True)
                        
                        # Create heatmap of the relationship
                        heatmap_df = pd.crosstab(
                            sample_data[comparison_var], 
                            sample_data[secondary_var], 
                            normalize='all'
                        ) * 100
                        
                        fig = px.imshow(
                            heatmap_df,
                            text_auto='.1f',
                            labels=dict(x=secondary_var, y=comparison_var, color="Pourcentage (%)"),
                            title=f"Heatmap de la relation entre {comparison_var} et {secondary_var}",
                            color_continuous_scale="Greens"
                        )
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Numeric variables distribution
                if len(numeric_cols) > 0:
                    st.markdown('<div class="section-header">Distribution des variables numériques</div>', unsafe_allow_html=True)
                    
                    selected_var = st.selectbox(
                        "Sélectionner une variable numérique",
                        options=numeric_cols
                    )
                    
                    # Distribution comparison
                    fig = go.Figure()
                    
                    # Histogram for frame
                    fig.add_trace(go.Histogram(
                        x=sampling_frame[selected_var],
                        name='Cadre',
                        opacity=0.7,
                        marker_color='rgba(58, 71, 80, 0.7)',
                        nbinsx=30
                    ))
                    
                    # Histogram for sample
                    fig.add_trace(go.Histogram(
                        x=sample_data[selected_var],
                        name='Échantillon',
                        opacity=0.7,
                        marker_color='rgba(246, 78, 139, 0.7)',
                        nbinsx=30
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Distribution comparative de {selected_var}",
                        xaxis_title=selected_var,
                        yaxis_title="Fréquence",
                        barmode='overlay',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Advanced Analysis
            with tabs[3]:
                st.markdown('<div class="section-header">Analyse avancée de l\'échantillon</div>', unsafe_allow_html=True)
                
                # Confidence intervals for key metrics
                if len(numeric_cols) > 0:
                    st.markdown("#### Intervalles de confiance")
                    
                    # Calculate confidence intervals
                    ci_df = calculate_confidence_intervals(sample_data, numeric_cols, confidence_level)
                    
                    # Display the results
                    st.dataframe(style_dataframe(ci_df))
                    
                    # Plot confidence intervals for selected variable
                    selected_var_ci = st.selectbox(
                        "Afficher l'intervalle de confiance pour",
                        options=numeric_cols,
                        key="ci_var_selector"
                    )
                    
                    # Get CI values for selected variable
                    var_row = ci_df.loc[selected_var_ci]
                    mean_val = var_row['Moyenne']
                    lower_ci = var_row['IC Inf']
                    upper_ci = var_row['IC Sup']
                    
                    # Create confidence interval visualization
                    fig = go.Figure()
                    
                    # Add mean point
                    fig.add_trace(go.Scatter(
                        x=[selected_var_ci],
                        y=[mean_val],
                        mode='markers',
                        marker=dict(size=12, color='green'),
                        name='Moyenne estimée'
                    ))
                    
                    # Add error bars for CI
                    fig.add_trace(go.Scatter(
                        x=[selected_var_ci, selected_var_ci],
                        y=[lower_ci, upper_ci],
                        mode='lines',
                        line=dict(width=3, color='green'),
                        name=f'Intervalle de confiance ({confidence_level}%)'
                    ))
                    
                    # Add population mean reference
                    pop_mean = sampling_frame[selected_var_ci].mean()
                    fig.add_trace(go.Scatter(
                        x=[selected_var_ci],
                        y=[pop_mean],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='diamond'),
                        name='Moyenne du cadre'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Intervalle de confiance ({confidence_level}%) pour {selected_var_ci}",
                        xaxis_title="",
                        yaxis_title="Valeur",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sampling diagnostics
                st.markdown("#### Diagnostics d'échantillonnage")
                
                # Calculate sampling diagnostics
                diagnostics = generate_sampling_diagnostics(sampling_frame, sample_data)
                
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Score de représentativité", 
                        f"{diagnostics['representativeness_score']:.1f}%",
                        delta="Bon" if diagnostics['representativeness_score'] > 80 else "Moyen" if diagnostics['representativeness_score'] > 60 else "Faible"
                    )
                
                with col2:
                    st.metric(
                        "Variables représentatives", 
                        f"{sum(1 for test in diagnostics['representativeness_tests'].values() if test.get('representative', False))}/{len(diagnostics['representativeness_tests'])}"
                    )
                
                with col3:
                    st.metric(
                        "Taux d'échantillonnage", 
                        f"{diagnostics['sampling_rate']:.2%}"
                    )
                
                # Representativeness tests for categorical variables
                st.markdown("##### Tests de représentativité (Variables catégorielles)")
                
                # Create dataframe from representativeness tests
                rep_tests = []
                for var, test in diagnostics['representativeness_tests'].items():
                    rep_tests.append({
                        'Variable': var,
                        'Statistique Chi²': f"{test.get('chi2', float('nan')):.2f}",
                        'Valeur p': f"{test.get('p_value', float('nan')):.4f}",
                        'Représentativité': "✅ Oui" if test.get('representative', False) else "❌ Non"
                    })
                
                rep_tests_df = pd.DataFrame(rep_tests)
                st.dataframe(rep_tests_df)
                
                # Numeric variable comparisons
                st.markdown("##### Comparaison des moyennes (Variables numériques)")
                
                # Create dataframe from numeric comparisons
                num_comps = []
                for var, comp in diagnostics['numeric_comparisons'].items():
                    num_comps.append({
                        'Variable': var,
                        'Moyenne (Cadre)': f"{comp.get('frame_mean', float('nan')):.2f}",
                        'Moyenne (Échantillon)': f"{comp.get('sample_mean', float('nan')):.2f}",
                        'Différence (%)': f"{comp.get('rel_diff_pct', float('nan')):.2f}%",
                        'Valeur p': f"{comp.get('p_value', float('nan')):.4f}",
                        'Représentativité': "✅ Oui" if comp.get('representative', False) else "❌ Non"
                    })
                
                num_comps_df = pd.DataFrame(num_comps)
                st.dataframe(num_comps_df)
                
                # Bivariate analysis section
                if len(categorical_vars) >= 2 or len(numeric_cols) >= 2:
                    st.markdown("#### Analyse Bivariée")
                    
                    # Create two columns for variable selection
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        var1 = st.selectbox(
                            "Première variable",
                            options=sampling_frame.columns,
                            key="bivar_var1"
                        )
                    
                    with col2:
                        # Filter second variable options to avoid selecting the same variable
                        var2_options = [col for col in sampling_frame.columns if col != var1]
                        var2 = st.selectbox(
                            "Seconde variable",
                            options=var2_options,
                            key="bivar_var2"
                        )
                    
                    # Force categorical treatment
                    force_cat = st.checkbox("Traiter les variables comme catégorielles", value=False)
                    
                    # Generate bivariate analysis
                    try:
                        from utils import generate_bivariate_analysis  # Import here to ensure correct function is used
                        fig, bivar_stats = generate_bivariate_analysis(sample_data, var1, var2, categorical=force_cat)
                        
                        # Display the visualization
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display statistics based on analysis type
                        if bivar_stats['type'] == 'categorical':
                            st.markdown(f"""
                            **Résultats du test d'indépendance:**
                            - Test Chi² : {bivar_stats['chi2']:.2f} (degrés de liberté: {bivar_stats['dof']})
                            - Valeur p : {bivar_stats['p_value']:.4f}
                            - V de Cramer : {bivar_stats['cramers_v']:.3f} (Force de l'association: {bivar_stats['association_strength']})
                            - Conclusion : Les variables sont {'indépendantes' if bivar_stats['independent'] else 'dépendantes'} (p {'>' if bivar_stats['independent'] else '<'} 0.05)
                            """)
                        
                        elif bivar_stats['type'] == 'numeric':
                            st.markdown(f"""
                            **Résultats de l'analyse de corrélation:**
                            - Corrélation de Pearson (r) : {bivar_stats['pearson_corr']:.3f} (p = {bivar_stats['pearson_p']:.4f})
                            - Corrélation de Spearman (ρ) : {bivar_stats['spearman_corr']:.3f} (p = {bivar_stats['spearman_p']:.4f})
                            - Force de la corrélation : {bivar_stats['correlation_strength']}
                            - Conclusion : La corrélation est {'significative' if bivar_stats['significant'] else 'non significative'} (p {'<' if bivar_stats['significant'] else '>'} 0.05)
                            """)
                        
                        elif bivar_stats['type'] == 'mixed':
                            st.markdown(f"""
                            **Résultats de l'analyse de variance (ANOVA):**
                            - Variable catégorielle : {bivar_stats['categorical_var']}
                            - Variable numérique : {bivar_stats['numeric_var']}
                            - Statistique F : {bivar_stats['f_statistic']:.2f}
                            - Valeur p : {bivar_stats['p_value']:.4f}
                            - Eta² (taille d'effet) : {bivar_stats['eta_squared']:.3f} ({bivar_stats['effect_size']})
                            - Conclusion : L'effet est {'significatif' if bivar_stats['significant'] else 'non significatif'} (p {'<' if bivar_stats['significant'] else '>'} 0.05)
                            """)
                    
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse bivariée: {e}")

# Stratified Sampling module
def stratified_sampling_module(sampling_frame):
    """Display the stratified sampling module"""
    st.markdown("""
    <div class="method-card">
        <div class="method-title">Méthode de Stratification à Allocation Proportionnelle</div>
        <p>Cette méthode divise la population en strates homogènes et alloue proportionnellement l'échantillon selon la taille de chaque strate.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout with two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample size input
        sample_size = st.number_input(
            "Taille de l'échantillon", 
            min_value=1, 
            max_value=sampling_frame.shape[0],
            value=min(int(sampling_frame.shape[0] * 0.1), 100),
            step=1
        )
        
        # Stratification variable selection
        strat_var_options = [col for col in sampling_frame.columns 
                          if is_categorical_like(sampling_frame[col])]
        
        if len(strat_var_options) > 0:
            strat_var = st.selectbox(
                "Variable de stratification",
                options=strat_var_options,
                help="Choisir parmi: Régions, Gouvernorats ou Délégations"
            )
        else:
            st.error("Aucune variable catégorielle appropriée pour la stratification n'a été trouvée.")
            strat_var = None
    
    with col2:
        # Confidence level
        confidence_level = st.slider(
            "Niveau de confiance (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
        
        # Auxiliary variable selection
        if strat_var:
            aux_var_options = [col for col in sampling_frame.columns if col != strat_var]
            aux_var = st.selectbox(
                "Variable auxiliaire",
                options=["Aucune"] + aux_var_options,
                help="Choisir le milieu (urbain, rural) ou la taille approximative de chaque bloc"
            )
            
            if aux_var == "Aucune":
                aux_var = None
        else:
            aux_var = None
    
    # Execute stratified sampling button
    if strat_var and st.button("Générer l'échantillon stratifié", key="generate_stratified"):
        with st.spinner("Génération de l'échantillon stratifié en cours..."):
            # Draw the stratified sample
            sample_data, allocation_df, strata_stats = stratified_sample(sampling_frame, strat_var, sample_size, aux_var)
            
            # Display the results in tabs
            tabs = st.tabs(["📊 Allocation", "📋 Échantillon", "📈 Statistiques", "🔍 Analyse avancée"])
            
            # Tab 1: Allocation
            with tabs[0]:
                st.markdown('<div class="section-header">Allocation proportionnelle par strate</div>', unsafe_allow_html=True)
                
                # Display allocation table
                st.dataframe(style_dataframe(allocation_df))
                
                # Create allocation visualization
                fig = plot_strata_allocation(allocation_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Download allocation table
                st.markdown(get_excel_download_link(allocation_df, "allocation_proportionnelle", "💾 Télécharger l'allocation (Excel)"), unsafe_allow_html=True)
                
                # Display stratum statistics if available
                if strata_stats is not None:
                    st.markdown('<div class="section-header">Statistiques par strate</div>', unsafe_allow_html=True)
                    st.dataframe(style_dataframe(strata_stats))
            
            # Tab 2: Sample
            with tabs[1]:
                st.markdown('<div class="section-header">Données de l\'échantillon stratifié</div>', unsafe_allow_html=True)
                
                # Sample metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Taille de l'échantillon", f"{sample_size:,}")
                col2.metric("Taux d'échantillonnage", f"{sample_size/sampling_frame.shape[0]:.2%}")
                col3.metric("Nombre de strates", f"{len(allocation_df)}")
                
                # Display sample data with download option
                st.dataframe(style_dataframe(sample_data))
                
                st.markdown(get_excel_download_link(sample_data, "echantillon_stratifie", "💾 Télécharger l'échantillon (Excel)"), unsafe_allow_html=True)
                
                # Sample distribution by stratum
                st.markdown('<div class="section-header">Distribution de l\'échantillon par strate</div>', unsafe_allow_html=True)
                
                # Calculate sample distribution
                sample_distribution = sample_data[strat_var].value_counts().reset_index()
                sample_distribution.columns = ['Strate', 'Effectif']
                
                # Calculate proportions
                sample_distribution['Proportion (%)'] = sample_distribution['Effectif'] / sample_distribution['Effectif'].sum() * 100
                
                # Compare with frame distribution
                frame_distribution = sampling_frame[strat_var].value_counts().reset_index()
                frame_distribution.columns = ['Strate', 'Effectif_Cadre']
                frame_distribution['Proportion_Cadre (%)'] = frame_distribution['Effectif_Cadre'] / frame_distribution['Effectif_Cadre'].sum() * 100
                
                # Merge distributions
                merged_distribution = pd.merge(sample_distribution, frame_distribution, on='Strate', how='outer').fillna(0)
                
                # Calculate difference
                merged_distribution['Différence (pts)'] = merged_distribution['Proportion (%)'] - merged_distribution['Proportion_Cadre (%)']
                
                # Display merged distribution
                st.dataframe(style_dataframe(merged_distribution))
                
                # Create distribution visualization
                fig = px.bar(
                    merged_distribution,
                    x='Strate',
                    y=['Proportion (%)', 'Proportion_Cadre (%)'],
                    barmode='group',
                    labels={'value': 'Proportion (%)', 'variable': ''},
                    title='Distribution comparative par strate',
                    color_discrete_map={'Proportion (%)': 'rgba(246, 78, 139, 0.6)', 'Proportion_Cadre (%)': 'rgba(58, 71, 80, 0.6)'}
                )
                
                fig.update_layout(legend={'title': '', 'orientation': 'h', 'y': 1.1, 'x': 0.5, 'xanchor': 'center'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 3: Statistics
            with tabs[2]:
                st.markdown('<div class="section-header">Statistiques descriptives</div>', unsafe_allow_html=True)
                
                # Numeric statistics
                numeric_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numeric_cols) > 0:
                    # Calculate stats
                    stats_df = calculate_descriptive_stats(sample_data)
                    
                    # Add comparison with frame
                    frame_stats = calculate_descriptive_stats(sampling_frame)
                    
                    # Calculate differences
                    diff_df = pd.DataFrame({
                        'Variable': stats_df.index,
                        'Moyenne (Échantillon)': stats_df['mean'],
                        'Moyenne (Cadre)': frame_stats['mean'],
                        'Écart (%)': ((stats_df['mean'] - frame_stats['mean']) / frame_stats['mean'] * 100).round(2),
                        'Écart-type (Échantillon)': stats_df['std'],
                        'CV (Échantillon)': stats_df['cv'],
                        'CV (Cadre)': frame_stats['cv']
                    })
                    
                    # Style and display
                    st.dataframe(style_dataframe(diff_df.set_index('Variable')))
                    
                    # Download link
                    st.markdown(get_excel_download_link(diff_df, "statistiques_stratifie", "💾 Télécharger les statistiques (Excel)"), unsafe_allow_html=True)
                    
                    # Select variable for more detailed statistics
                    selected_var = st.selectbox(
                        "Sélectionner une variable pour des statistiques détaillées",
                        options=numeric_cols,
                        key="detailed_stats_var"
                    )
                    
                    # Calculate statistics by stratum
                    stats_by_stratum = sample_data.groupby(strat_var)[selected_var].agg(['mean', 'std', 'min', 'max']).reset_index()
                    stats_by_stratum.columns = ['Strate', 'Moyenne', 'Écart-type', 'Minimum', 'Maximum']
                    
                    # Display statistics by stratum
                    st.markdown(f'<div class="section-header">Statistiques de {selected_var} par strate</div>', unsafe_allow_html=True)
                    st.dataframe(style_dataframe(stats_by_stratum))
                    
                    # Create visualization
                    fig = px.box(
                        sample_data,
                        x=strat_var,
                        y=selected_var,
                        title=f'Distribution de {selected_var} par strate',
                        color=strat_var
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune variable numérique trouvée pour les statistiques descriptives.")
                
                # Auxiliary variable distribution if specified
                if aux_var:
                    st.markdown(f'<div class="section-header">Distribution de la variable auxiliaire: {aux_var}</div>', unsafe_allow_html=True)
                    
                    if is_categorical_like(sampling_frame[aux_var]):
                        # For categorical auxiliary variable
                        comparison_df, _ = compare_categorical_distributions(sampling_frame, sample_data, aux_var)
                        
                        # Display comparison dataframe
                        st.dataframe(style_dataframe(comparison_df))
                        
                        # Create visualization
                        fig = px.bar(
                            comparison_df,
                            x='Modalité',
                            y=['Proportion dans le cadre (%)', 'Proportion dans l\'échantillon (%)'],
                            barmode='group',
                            title=f'Distribution comparative de {aux_var}',
                            color_discrete_map={
                                'Proportion dans le cadre (%)': 'rgba(58, 71, 80, 0.7)',
                                'Proportion dans l\'échantillon (%)': 'rgba(246, 78, 139, 0.7)'
                            }
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # For numeric auxiliary variable
                        from utils import create_distribution_plot  # Import here to avoid name conflicts
                        fig = create_distribution_plot(sampling_frame, sample_data, aux_var)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Tab 4: Advanced Analysis
            with tabs[3]:
                st.markdown('<div class="section-header">Analyse avancée de l\'échantillon stratifié</div>', unsafe_allow_html=True)
                
                # Confidence intervals for key metrics
                if len(numeric_cols) > 0:
                    st.markdown("#### Intervalles de confiance")
                    
                    # Calculate confidence intervals
                    ci_df = calculate_confidence_intervals(sample_data, numeric_cols, confidence_level)
                    
                    # Display the results
                    st.dataframe(style_dataframe(ci_df))
                    
                    # Plot confidence intervals for selected variable
                    selected_var_ci = st.selectbox(
                        "Afficher l'intervalle de confiance pour",
                        options=numeric_cols,
                        key="strat_ci_var_selector"
                    )
                    
                    # Get CI values for selected variable
                    var_row = ci_df.loc[selected_var_ci]
                    mean_val = var_row['Moyenne']
                    lower_ci = var_row['IC Inf']
                    upper_ci = var_row['IC Sup']
                    
                    # Create confidence interval visualization
                    fig = go.Figure()
                    
                    # Add mean point
                    fig.add_trace(go.Scatter(
                        x=[selected_var_ci],
                        y=[mean_val],
                        mode='markers',
                        marker=dict(size=12, color='green'),
                        name='Moyenne estimée'
                    ))
                    
                    # Add error bars for CI
                    fig.add_trace(go.Scatter(
                        x=[selected_var_ci, selected_var_ci],
                        y=[lower_ci, upper_ci],
                        mode='lines',
                        line=dict(width=3, color='green'),
                        name=f'Intervalle de confiance ({confidence_level}%)'
                    ))
                    
                    # Add population mean reference
                    pop_mean = sampling_frame[selected_var_ci].mean()
                    fig.add_trace(go.Scatter(
                        x=[selected_var_ci],
                        y=[pop_mean],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='diamond'),
                        name='Moyenne du cadre'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Intervalle de confiance ({confidence_level}%) pour {selected_var_ci}",
                        xaxis_title="",
                        yaxis_title="Valeur",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Stratification efficiency
                st.markdown("#### Efficacité de la stratification")
                
                # Calculate diagnostics with stratification information
                diagnostics = generate_sampling_diagnostics(sampling_frame, sample_data, strat_var)
                
                # Display key metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    design_effect = diagnostics.get('design_effect', 1.0)
                    st.metric(
                        "Effet de plan (DEFF)", 
                        f"{design_effect:.3f}",
                        delta="Efficace" if design_effect < 1.0 else "Neutre" if design_effect <= 1.05 else "Inefficace"
                    )
                
                with col2:
                    effective_sample = diagnostics.get('effective_sample_size', sample_size)
                    st.metric(
                        "Taille d'échantillon effective", 
                        f"{effective_sample:.1f}",
                        delta=f"{effective_sample/sample_size:.1%} de l'échantillon réel"
                    )
                
                st.markdown("""
                **Interprétation:**
                - Un effet de plan (DEFF) < 1 indique que la stratification a amélioré la précision
                - Un effet de plan = 1 indique que la stratification n'a pas d'effet sur la précision
                - Un effet de plan > 1 indique que la stratification a réduit la précision
                
                La taille d'échantillon effective indique la taille d'un échantillon aléatoire simple qui donnerait la même précision que l'échantillon stratifié.
                """)
                
                # Bivariate analysis by stratum
                if len(numeric_cols) > 0:
                    st.markdown("#### Analyse bivariée par strate")
                    
                    # Select variables for bivariate analysis
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        bivar_var1 = st.selectbox(
                            "Première variable",
                            options=numeric_cols,
                            key="strat_bivar_var1"
                        )
                    
                    with col2:
                        bivar_var2_options = [col for col in numeric_cols if col != bivar_var1]
                        if bivar_var2_options:
                            bivar_var2 = st.selectbox(
                                "Seconde variable",
                                options=bivar_var2_options,
                                key="strat_bivar_var2"
                            )
                        else:
                            bivar_var2 = None
                    
                    if bivar_var2:
                        # Create scatter plot by stratum
                        fig = px.scatter(
                            sample_data,
                            x=bivar_var1,
                            y=bivar_var2,
                            color=strat_var,
                            trendline="ols",
                            title=f"Relation entre {bivar_var1} et {bivar_var2} par strate",
                            labels={bivar_var1: bivar_var1, bivar_var2: bivar_var2},
                            trendline_scope="overall",
                            trendline_color_override="black"
                        )
                        
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation by stratum
                        corr_by_stratum = []
                        
                        for stratum in sample_data[strat_var].unique():
                            stratum_data = sample_data[sample_data[strat_var] == stratum]
                            
                            if len(stratum_data) >= 3:  # Minimum sample for correlation
                                from scipy import stats as scipy_stats  # Import here to avoid name conflict
                                pearson_corr, p_value = scipy_stats.pearsonr(
                                    stratum_data[bivar_var1], 
                                    stratum_data[bivar_var2]
                                )
                                
                                corr_by_stratum.append({
                                    'Strate': stratum,
                                    'Correlation': pearson_corr,
                                    'Valeur p': p_value,
                                    'Significatif': "✅ Oui" if p_value < 0.05 else "❌ Non"
                                })
                        
                        if corr_by_stratum:
                            corr_df = pd.DataFrame(corr_by_stratum)
                            st.dataframe(style_dataframe(corr_df))

# Main application
def main():
    """Main application function"""
    # Display header
    display_header()
    
    # Display sidebar and get settings
    uploaded_file, sampling_method, sampling_frame = display_sidebar()
    
    # Main content
    if uploaded_file is None:
        # Welcome page when no file is uploaded
        display_welcome_page()
    else:
        # Display appropriate sampling module based on selection
        if sampling_method == "Aléatoire Simple sans Remise (SAS)":
            srs_sampling_module(sampling_frame)
        else:
            stratified_sampling_module(sampling_frame)
    
    # Footer
    st.markdown("""
    <div class="app-footer">
        <div style="text-align: center; padding-right: 50px;">Développé pour le projet de Théorie de Sondage - 2ème année</div>
        <div style="text-align: center; padding-right: 50px;">Développé par : Laadhar Youssef & Khlifi Ismail</div>
        <div style="text-align: center; padding-right: 50px;">École supérieure de la statistique et l'Analyse de l'information | 2024-2025</div>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()