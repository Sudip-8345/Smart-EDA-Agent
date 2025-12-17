"""
AI EDA Agent - Streamlit Frontend
A modern, interactive data analysis tool with AI-powered insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx
import io
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AI Provider imports
try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI EDA Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== MODEL CONFIGURATION ==============
# All models are multimodal capable
MODEL_CONFIG = {
    "llama3.2-1b": {
        "provider": "ollama",
        "model_id": "llama3.2:1b",
        "display_name": "Llama 3.2 1B (Local Fast)",
        "api_key_env": None,
        "multimodal": False
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "model_id": "gemini-2.0-flash",
        "display_name": "Gemini 2.5 Flash",
        "api_key_env": "GOOGLE_API_KEY",
        "multimodal": True
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "model_id": "gemini-2.0-flash-exp",
        "display_name": "Gemini 2.0 Flash",
        "api_key_env": "GOOGLE_API_KEY",
        "multimodal": True
    },
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "display_name": "GPT-4o (OpenAI)",
        "api_key_env": "OPENAI_API_KEY",
        "multimodal": True
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini (OpenAI)",
        "api_key_env": "OPENAI_API_KEY",
        "multimodal": True
    }
}

# Default model - using local Ollama 1B (fast, no rate limits)
ACTIVE_MODEL = "llama3.2-1b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Plot selector */
    .plot-btn {
        background: #2d2d44;
        border: 1px solid #404060;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .plot-btn:hover {
        background: #404060;
        border-color: #667eea;
    }
    
    /* Success/info boxes */
    .stSuccess, .stInfo {
        border-radius: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #2d2d44;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 0.5rem;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============== AI FUNCTIONS ==============
def ai_generate(prompt: str, model_key: str = None) -> str:
    """Generate AI response using the configured provider (Google, OpenAI, or Ollama)."""
    if model_key is None:
        model_key = st.session_state.get("active_model", ACTIVE_MODEL)
    
    config = MODEL_CONFIG.get(model_key)
    if not config:
        return f"Error: Unknown model '{model_key}'"
    
    provider = config["provider"]
    model_id = config["model_id"]
    
    try:
        if provider == "google":
            return _generate_google(prompt, model_id, config)
        elif provider == "openai":
            return _generate_openai(prompt, model_id, config)
        elif provider == "ollama":
            return _generate_ollama(prompt, model_id)
        else:
            return f"Error: Unknown provider '{provider}'"
    except Exception as e:
        return f"AI Error: {str(e)}"


def _generate_google(prompt: str, model_id: str, config: dict) -> str:
    """Generate response using Google Gemini API."""
    if not GOOGLE_AVAILABLE:
        return "Error: google-generativeai package not installed. Run: pip install google-generativeai"
    
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        return f"Error: {config['api_key_env']} environment variable not set"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=500,
            temperature=0.7
        )
    )
    return response.text


def _generate_openai(prompt: str, model_id: str, config: dict) -> str:
    """Generate response using OpenAI API."""
    if not OPENAI_AVAILABLE:
        return "Error: openai package not installed. Run: pip install openai"
    
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        return f"Error: {config['api_key_env']} environment variable not set"
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content


def _generate_ollama(prompt: str, model_id: str) -> str:
    """Generate response using local Ollama server."""
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500}
            }
        )
        if response.status_code != 200:
            return f"Ollama Error: HTTP {response.status_code}"
        result = response.json()
        text = result.get("response", "")
        if not text:
            return "No response. Make sure Ollama is running (ollama serve)."
        return text


def get_analysis_plan(df_info: str) -> str:
    prompt = f"""You are an expert data analyst. Based on this dataset info, provide a concise 5-point analysis plan.
Be specific about what to analyze and why.

Dataset Info:
{df_info[:2000]}

Provide exactly 5 bullet points:"""
    return ai_generate(prompt)


def get_insights(df_info: str) -> str:
    prompt = f"""You are an expert data analyst. Based on this dataset, provide 5 key actionable insights.
Be specific with numbers and recommendations.

Dataset Info:
{df_info[:2000]}

Provide exactly 5 insights:"""
    return ai_generate(prompt)


def get_plot_summary(plot_title: str, df: pd.DataFrame) -> str:
    """Generate a summary for a specific plot."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    context = f"""Plot: {plot_title}
Dataset: {len(df)} rows, {len(df.columns)} columns
Numeric columns: {numeric_cols[:5]}
Categorical columns: {categorical_cols[:5]}"""
    
    if numeric_cols and "Correlation" in plot_title:
        corr = df[numeric_cols].corr()
        top_corrs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                top_corrs.append((abs(corr.iloc[i,j]), numeric_cols[i], numeric_cols[j]))
        top_corrs.sort(reverse=True)
        context += f"\nTop correlations: {top_corrs[:3]}"
    
    prompt = f"""Give 3 brief insights about this visualization in bullet points.
{context}"""
    return ai_generate(prompt)


# ============== DATA FUNCTIONS ==============
def get_df_info(df: pd.DataFrame) -> str:
    """Get comprehensive dataframe info string."""
    buf = io.StringIO()
    df.info(buf=buf)
    info = buf.getvalue()
    
    preview = df.head(5).to_markdown(index=False)
    
    missing = df.isna().sum()
    missing = missing[missing > 0]
    missing_str = "No missing values" if missing.empty else missing.to_string()
    
    return f"### Schema:\n```\n{info}\n```\n\n### Preview:\n{preview}\n\n### Missing:\n{missing_str}"


def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics."""
    return df.describe().round(2)


# ============== PLOTTING FUNCTIONS ==============
def create_correlation_heatmap(df: pd.DataFrame):
    """Create interactive correlation heatmap."""
    numeric_df = df.select_dtypes(include=np.number)
    if len(numeric_df.columns) < 2:
        return None
    
    corr = numeric_df.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    fig.update_layout(height=500)
    return fig


def create_histogram_grid(df: pd.DataFrame):
    """Create histogram grid for numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:9]
    if not numeric_cols:
        return None
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_cols)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(height=300 * n_rows, title="Distribution of Numeric Variables")
    return fig


def create_boxplot_grid(df: pd.DataFrame):
    """Create boxplot grid for numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:8]
    if not numeric_cols:
        return None
    
    fig = go.Figure()
    for col in numeric_cols:
        fig.add_trace(go.Box(y=df[col].dropna(), name=col))
    
    fig.update_layout(height=500, title="Boxplots - Outlier Detection")
    return fig


def create_scatter_matrix(df: pd.DataFrame):
    """Create scatter matrix for numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()[:5]
    if len(numeric_cols) < 2:
        return None
    
    fig = px.scatter_matrix(
        df[numeric_cols].dropna(),
        dimensions=numeric_cols,
        title="Scatter Matrix"
    )
    fig.update_layout(height=600)
    return fig


def create_bar_chart(df: pd.DataFrame, col: str):
    """Create bar chart for categorical column."""
    value_counts = df[col].value_counts().head(15)
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title=f"Distribution of {col}",
        labels={"x": col, "y": "Count"},
        color=value_counts.values,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=400)
    return fig


def create_pie_chart(df: pd.DataFrame, col: str):
    """Create pie chart for categorical column."""
    value_counts = df[col].value_counts().head(8)
    fig = px.pie(
        names=value_counts.index,
        values=value_counts.values,
        title=f"Distribution of {col}",
        hole=0.4
    )
    fig.update_layout(height=400)
    return fig


def create_violin_plot(df: pd.DataFrame, num_col: str, cat_col: str):
    """Create violin plot."""
    fig = px.violin(
        df,
        x=cat_col,
        y=num_col,
        color=cat_col,
        title=f"{num_col} by {cat_col}",
        box=True
    )
    fig.update_layout(height=500)
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col=None):
    """Create scatter plot."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{x_col} vs {y_col}",
        trendline="ols" if color_col is None else None
    )
    fig.update_layout(height=500)
    return fig


def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str):
    """Create line chart."""
    fig = px.line(
        df.sort_values(x_col),
        x=x_col,
        y=y_col,
        title=f"{y_col} over {x_col}"
    )
    fig.update_layout(height=400)
    return fig


# ============== SESSION STATE INIT ==============
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_plan' not in st.session_state:
    st.session_state.analysis_plan = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'current_plot' not in st.session_state:
    st.session_state.current_plot = None
if 'plot_summary' not in st.session_state:
    st.session_state.plot_summary = None


# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("# 📊 AI EDA Agent")
    st.markdown("---")
    
    # Model selector
    st.markdown("### AI Model")
    model_options = list(MODEL_CONFIG.keys())
    model_display_names = [MODEL_CONFIG[k]["display_name"] for k in model_options]
    
    # Initialize session state for model
    if "active_model" not in st.session_state:
        st.session_state.active_model = ACTIVE_MODEL
    
    selected_idx = model_options.index(st.session_state.active_model) if st.session_state.active_model in model_options else 0
    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        format_func=lambda x: MODEL_CONFIG[x]["display_name"],
        index=selected_idx,
        help="Choose an AI model for analysis. Ollama models run locally, others need API keys."
    )
    st.session_state.active_model = selected_model
    
    # Show provider info
    provider = MODEL_CONFIG[selected_model]["provider"]
    if provider == "ollama":
        st.caption("🟢 Local model (no API key needed)")
    else:
        api_key_env = MODEL_CONFIG[selected_model]["api_key_env"]
        if os.getenv(api_key_env):
            st.caption(f"🟢 {provider.title()} API key found")
        else:
            st.caption(f"🔴 Set {api_key_env} env variable")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload your dataset to begin analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"Loaded: {uploaded_file.name}")
            st.markdown(f"**Rows:** {len(df):,}")
            st.markdown(f"**Columns:** {len(df.columns)}")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("### Quick Stats")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Numeric", len(numeric_cols))
        with col2:
            st.metric("Categorical", len(categorical_cols))
        
        missing = df.isna().sum().sum()
        st.metric("Missing Values", f"{missing:,}")
        
        st.markdown("---")
        st.markdown("### Generate AI Insights")
        
        if st.button("Generate Analysis Plan"):
            with st.spinner("Generating plan..."):
                result = get_analysis_plan(get_df_info(df))
                st.session_state.analysis_plan = result
                st.rerun()
        
        if st.session_state.analysis_plan:
            st.success("Analysis Plan Generated!")
            st.markdown(st.session_state.analysis_plan)
        
        if st.button("Generate Key Insights"):
            with st.spinner("Generating insights..."):
                result = get_insights(get_df_info(df))
                st.session_state.insights = result
                st.rerun()
        
        if st.session_state.insights:
            st.success("Key Insights Generated!")
            st.markdown(st.session_state.insights)


# ============== MAIN CONTENT ==============
if st.session_state.df is None:
    # Welcome screen
    st.markdown("""
    # Welcome to AI EDA Agent
    
    **Your intelligent data analysis companion**
    
    Upload a CSV file in the sidebar to get started with:
    
    - **Interactive Visualizations** - Beautiful, interactive charts with Plotly
    - **AI-Powered Insights** - Get intelligent analysis using local LLMs
    - **One-Click Analysis** - Automatic EDA with just a few clicks
    - **Export Ready** - Download your visualizations and reports
    """)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈</h3>
            <p>Interactive Charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖</h3>
            <p>AI Insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡</h3>
            <p>Fast Analysis</p>
        </div>
        """, unsafe_allow_html=True)

else:
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(exclude=np.number).columns if 1 < df[c].nunique() < 30]
    
    # Header metrics
    st.markdown("# Data Analysis Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Numeric", len(numeric_cols))
    with col4:
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing %", f"{missing_pct:.1f}%")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview",
        "📊 Visualizations", 
        "🔍 Explore",
        "🤖 AI Insights",
        "📥 Export"
    ])
    
    # ============== TAB 1: OVERVIEW ==============
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), height=300)
        
        with col2:
            st.markdown("### Summary Statistics")
            if numeric_cols:
                st.dataframe(get_summary_stats(df), height=300)
            else:
                st.info("No numeric columns found")
        
        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Type': [str(t) for t in df.dtypes.values],
            'Non-Null': df.count().values.tolist(),
            'Null': df.isna().sum().values.tolist(),
            'Unique': df.nunique().values.tolist()
        })
        st.dataframe(col_info)
    
    # ============== TAB 2: VISUALIZATIONS ==============
    with tab2:
        viz_col1, viz_col2 = st.columns([1, 3])
        
        with viz_col1:
            st.markdown("### Select Plot")
            
            plot_options = ["Correlation Heatmap", "Distributions", "Boxplots", "Scatter Matrix"]
            
            if categorical_cols:
                plot_options.extend(["Bar Chart", "Pie Chart"])
            if numeric_cols and categorical_cols:
                plot_options.append("Violin Plot")
            if len(numeric_cols) >= 2:
                plot_options.append("Scatter Plot")
            
            selected_plot = st.radio(
                "Choose visualization:",
                plot_options,
                label_visibility="collapsed"
            )
            
            # Additional options based on plot type
            if selected_plot == "Bar Chart" and categorical_cols:
                bar_col = st.selectbox("Select column:", categorical_cols, key="bar_col")
            elif selected_plot == "Pie Chart" and categorical_cols:
                pie_col = st.selectbox("Select column:", categorical_cols, key="pie_col")
            elif selected_plot == "Violin Plot" and numeric_cols and categorical_cols:
                violin_num = st.selectbox("Numeric column:", numeric_cols, key="violin_num")
                violin_cat = st.selectbox("Category column:", categorical_cols, key="violin_cat")
            elif selected_plot == "Scatter Plot" and len(numeric_cols) >= 2:
                scatter_x = st.selectbox("X axis:", numeric_cols, key="scatter_x")
                scatter_y = st.selectbox("Y axis:", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
                scatter_color = st.selectbox("Color by:", ["None"] + categorical_cols, key="scatter_color")
            
            st.markdown("---")
            
            # Generate summary button
            if st.button("Generate Plot Summary"):
                with st.spinner("Analyzing plot with AI..."):
                    summary = get_plot_summary(selected_plot, df)
                    st.session_state.plot_summary = summary
                    st.session_state.current_plot = selected_plot
        
        with viz_col2:
            st.markdown(f"### {selected_plot}")
            
            fig = None
            
            if selected_plot == "Correlation Heatmap":
                fig = create_correlation_heatmap(df)
            elif selected_plot == "Distributions":
                fig = create_histogram_grid(df)
            elif selected_plot == "Boxplots":
                fig = create_boxplot_grid(df)
            elif selected_plot == "Scatter Matrix":
                fig = create_scatter_matrix(df)
            elif selected_plot == "Bar Chart" and categorical_cols:
                fig = create_bar_chart(df, bar_col)
            elif selected_plot == "Pie Chart" and categorical_cols:
                fig = create_pie_chart(df, pie_col)
            elif selected_plot == "Violin Plot" and numeric_cols and categorical_cols:
                fig = create_violin_plot(df, violin_num, violin_cat)
            elif selected_plot == "Scatter Plot" and len(numeric_cols) >= 2:
                color = None if scatter_color == "None" else scatter_color
                fig = create_scatter_plot(df, scatter_x, scatter_y, color)
            
            if fig:
                st.plotly_chart(fig)
            else:
                st.warning("Not enough data to create this visualization")
            
            # Show plot summary if it matches current plot
            if st.session_state.plot_summary and st.session_state.get("current_plot") == selected_plot:
                st.markdown("### AI Summary")
                st.success(st.session_state.plot_summary)
    
    # ============== TAB 3: EXPLORE ==============
    with tab3:
        st.markdown("### Data Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Filter Data")
            
            if numeric_cols:
                filter_col = st.selectbox("Select numeric column to filter:", numeric_cols)
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                
                range_vals = st.slider(
                    f"Filter {filter_col}",
                    min_val, max_val,
                    (min_val, max_val)
                )
                
                filtered_df = df[(df[filter_col] >= range_vals[0]) & (df[filter_col] <= range_vals[1])]
                st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} rows**")
            else:
                filtered_df = df
        
        with col2:
            st.markdown("#### Sort Data")
            sort_col = st.selectbox("Sort by:", df.columns)
            sort_order = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
            
            filtered_df = filtered_df.sort_values(sort_col, ascending=(sort_order == "Ascending"))
        
        st.dataframe(filtered_df, height=400)
        
        # Column distribution
        st.markdown("### Column Distribution")
        dist_col = st.selectbox("Select column:", df.columns, key="dist_col")
        
        if df[dist_col].dtype in ['int64', 'float64']:
            fig = px.histogram(df, x=dist_col, marginal="box", title=f"Distribution of {dist_col}")
        else:
            fig = px.bar(x=df[dist_col].value_counts().index[:20], 
                        y=df[dist_col].value_counts().values[:20],
                        title=f"Distribution of {dist_col}")
        
        st.plotly_chart(fig)
    
    # ============== TAB 4: AI INSIGHTS ==============
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Analysis Plan")
            if st.session_state.analysis_plan:
                st.markdown(st.session_state.analysis_plan)
            else:
                st.info("Click 'Generate Analysis Plan' in the sidebar to get AI recommendations")
        
        with col2:
            st.markdown("### Key Insights")
            if st.session_state.insights:
                st.markdown(st.session_state.insights)
            else:
                st.info("Click 'Generate Key Insights' in the sidebar to get AI insights")
        
        st.markdown("---")
        
        # Custom query
        st.markdown("### Ask AI")
        user_query = st.text_area("Ask a question about your data:", placeholder="e.g., What patterns do you see in the data?")
        
        if st.button("Get Answer"):
            if user_query:
                with st.spinner("Thinking..."):
                    df_context = get_df_info(df)
                    prompt = f"""Based on this dataset, answer the question.

Dataset info:
{df_context[:1500]}

Question: {user_query}

Answer concisely:"""
                    answer = ai_generate(prompt)
                    st.markdown("### Answer")
                    st.markdown(answer)
    
    # ============== TAB 5: EXPORT ==============
    with tab5:
        st.markdown("### Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Download Data")
            
            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "data.csv",
                "text/csv"
            )
            
            # Excel download
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            st.download_button(
                "Download Excel",
                buffer.getvalue(),
                "data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            st.markdown("#### Download Report")
            
            # Generate report
            report = f"""# EDA Report

## Dataset Overview
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Numeric columns: {len(numeric_cols)}
- Categorical columns: {len(categorical_cols)}
- Missing values: {df.isna().sum().sum():,}

## Column Summary
{col_info.to_markdown()}

## Analysis Plan
{st.session_state.analysis_plan or 'Not generated'}

## Key Insights  
{st.session_state.insights or 'Not generated'}
"""
            
            st.download_button(
                "Download Report (Markdown)",
                report,
                "eda_report.md",
                "text/markdown"
            )


# ============== FOOTER ==============
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>AI EDA Agent | Powered by Streamlit & Ollama</div>",
    unsafe_allow_html=True
)
