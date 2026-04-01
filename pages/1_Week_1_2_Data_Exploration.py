import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Theme state initialization
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom Theme Styling
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #0f172a; color: #f8fafc; }
        [data-testid="stHeader"] { background-color: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); }
        [data-testid="stSidebar"] { background-color: #1e293b !important; }
        .stMetric { background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important; }
        h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
        .stMarkdown p { color: #cbd5e1 !important; }
        .stButton>button { border-radius: 8px; background: #3b82f6; color: white; border: none; }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Week 1-2: Data Exploration", page_icon="📊", layout="wide")

head_col1, head_col2 = st.columns([12, 1])
with head_col1:
    st.title("📊 Week 1-2: Football Data Collection & Preprocessing")
with head_col2:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("### Milestone 1 & 2: Data Exploration and Feature Engineering")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("player_transfer_value_with_sentimenttttt.csv")

df = load_data()

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Overview", "🔍 EDA", "🛠️ Feature Engineering", "📊 Statistics"])

with tab1:
    st.markdown("## 📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Players", f"{df['player_name'].nunique():,}")
    with col3:
        st.metric("Total Seasons", f"{df['season'].nunique()}")
    with col4:
        st.metric("Features", f"{len(df.columns)}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Data Sample")
        st.dataframe(df.head(20), use_container_width=True, height=400)
    
    with col2:
        st.markdown("### 📈 Data Types")
        dtype_counts = df.dtypes.astype(str).value_counts()
        dtype_df = pd.DataFrame({
            'Data Type': dtype_counts.index,
            'Count': dtype_counts.values
        })
        
        fig = px.pie(dtype_df.reset_index(), values='Count', names='Data Type', 
                     title='Distribution of Data Types')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔢 Numerical Columns")
        st.write(f"**{df.select_dtypes(include=[np.number]).shape[1]}** numerical features")
        
        st.markdown("### 📝 Categorical Columns")
        st.write(f"**{df.select_dtypes(include=['object']).shape[1]}** categorical features")

with tab2:
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    # Missing values analysis
    st.markdown("### 🚨 Missing Values Analysis")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_data) > 0:
        fig = px.bar(missing_data, x='Column', y='Missing %', 
                     title='Missing Values by Column',
                     color='Missing %',
                     color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")
    
    st.markdown("---")
    
    # Distribution Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Market Value Distribution")
        fig = px.histogram(df, x='market_value_eur', nbins=50,
                          title='Distribution of Player Market Values',
                          labels={'market_value_eur': 'Market Value (€)'},
                          color_discrete_sequence=['#3b82f6'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📍 Position Distribution")
        position_counts = df['position'].value_counts()
        fig = px.pie(values=position_counts.values, names=position_counts.index,
                    title='Players by Position',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚽ Goals Distribution")
        fig = px.box(df, y='goals', color='position',
                    title='Goals Distribution by Position',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Assists Distribution")
        fig = px.box(df, y='assists', color='position',
                    title='Assists Distribution by Position',
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## 🛠️ Feature Engineering")
    
    st.markdown("""
    ### ✨ Engineered Features:
    - **Performance Trends**: Goals/90, Assists/90, Goal Contributions
    - **Age Metrics**: Age Squared, Age Decay Factor, Career Stage
    - **Efficiency Metrics**: Pass Accuracy, Shot Conversion Rate, Tackle Success Rate
    - **Injury Metrics**: Injury Burden Index, Availability Rate
    - **Sentiment Features**: VADER Scores, Polarity, Subjectivity
    - **Market Indicators**: Market Value Tier, Transfer Attractiveness Score
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Goals/90 vs Market Value")
        fig = px.scatter(df, x='goals_per90', y='market_value_eur',
                        color='position', size='matches',
                        hover_data=['player_name'],
                        title='Performance Impact on Market Value',
                        labels={'goals_per90': 'Goals per 90 min', 
                               'market_value_eur': 'Market Value (€)'},
                        color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Pass Accuracy vs Position")
        fig = px.violin(df, y='pass_accuracy_pct', x='position', box=True,
                       title='Pass Accuracy Distribution by Position',
                       color='position',
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.markdown("### 🔥 Feature Correlation Heatmap")
    
    numeric_cols = ['market_value_eur', 'goals', 'assists', 'matches', 
                   'pass_accuracy_pct', 'goals_per90', 'assists_per90',
                   'vader_compound_score', 'current_age']
    
    corr_df = df[numeric_cols].corr()
    
    fig = px.imshow(corr_df, 
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix of Key Features')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## 📊 Descriptive Statistics")
    
    # Summary statistics
    st.markdown("### 📈 Key Performance Metrics Summary")
    
    stats_cols = ['market_value_eur', 'goals', 'assists', 'matches', 'minutes_played',
                  'pass_accuracy_pct', 'goals_per90', 'assists_per90', 'vader_compound_score']
    
    summary_stats = df[stats_cols].describe().T
    summary_stats['median'] = df[stats_cols].median()
    summary_stats = summary_stats[['count', 'mean', 'median', 'std', 'min', 'max']]
    
    st.dataframe(summary_stats.style.format("{:.2f}").background_gradient(cmap='Blues'),
                use_container_width=True)
    
    st.markdown("---")
    
    # Team statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Teams by Average Market Value")
        team_avg = df.groupby('team')['market_value_eur'].mean().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=team_avg.values/1e6, y=team_avg.index, orientation='h',
                    labels={'x': 'Avg Market Value (€M)', 'y': 'Team'},
                    title='Top 10 Teams by Average Player Value',
                    color=team_avg.values,
                    color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⚡ Top Players by Market Value")
        top_players = df.nlargest(10, 'market_value_eur')[['player_name', 'market_value_eur', 'position']]
        
        fig = px.bar(top_players, x='market_value_eur', y='player_name', 
                    orientation='h', color='position',
                    labels={'market_value_eur': 'Market Value (€)', 'player_name': 'Player'},
                    title='Top 10 Most Valuable Players',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.info("""
### ✅ Week 1-2 Deliverables Completed:
- ✓ Data collection from multiple sources (StatsBomb, Transfermarkt, Social Media)
- ✓ Data cleaning and preprocessing
- ✓ Exploratory Data Analysis (EDA)
- ✓ Feature engineering (performance, age, sentiment metrics)
- ✓ Statistical analysis and visualization
""")
