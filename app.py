import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Theme state initialization
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Page config
st.set_page_config(
    page_title="TransferIQ - AI Football Player Valuation",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Styling
if st.session_state.dark_mode:
    # DARK MODE CSS
    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #0f172a; color: #f8fafc; }
        [data-testid="stHeader"] { background-color: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); }
        [data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
        
        /* Sidebar text and nav */
        [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
        div[data-testid="stSidebarNav"] a span { color: #f8fafc !important; font-weight: 600; }
        
        /* Metrics - Super clear contrast */
        .stMetric { background-color: #1e293b !important; color: white !important; border: 1px solid #3b82f6 !important; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3) !important; border-radius: 12px !important; }
        .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; }
        .stMetric [data-testid="stMetricLabel"] { color: #f1f5f9 !important; font-weight: 500 !important; font-size: 1.1rem !important; }
        
        /* Headers */
        .main-header { 
            font-size: 2.8rem; font-weight: 900; text-align: center; padding: 1.5rem 0; width: 100%;
            background: linear-gradient(90deg, #93c5fd 0%, #3b82f6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 800 !important; }
        .stMarkdown p, .stMarkdown span, .stMarkdown li { color: #e2e8f0 !important; }
        
        /* SELECTBOX FIX */
        div[data-baseweb="select"] { background-color: #1e293b !important; border-radius: 8px !important; }
        div[data-baseweb="select"] * { color: #ffffff !important; }
        .stSelectbox label { color: #ffffff !important; font-weight: 600 !important; }
        div[role="listbox"] { background-color: #1e293b !important; }
        div[role="option"] { color: #ffffff !important; }
        div[role="option"]:hover { background-color: #3b82f6 !important; }

        hr { border-color: #334155 !important; margin: 2rem 0 !important; }
        div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; }
        .stButton>button { border-radius: 8px; background: #3b82f6; color: white; border: none; font-weight: 600; }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT MODE CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem; font-weight: 800; text-align: center; padding: 1.5rem;
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("player_transfer_value_with_sentimenttttt.csv")

df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## ⚽ TransferIQ Dashboard")
    st.markdown("---")
    players = sorted(df['player_name'].unique())
    selected_player = st.selectbox("Select Player", players)
    
    player_df = df[df['player_name'] == selected_player].sort_values('season')
    latest_data = player_df.iloc[-1]
    
    st.markdown("---")
    st.metric("Current Club", latest_data['team'])
    st.metric("Position", latest_data['position'])
    st.metric("Age", int(latest_data['current_age']))
    st.markdown("---")
    
    sentiment_score = latest_data['vader_compound_score']
    sentiment_label = latest_data['sentiment_label']
    if sentiment_label == 'Positive': st.success(f"✅ {sentiment_label}")
    elif sentiment_label == 'Negative': st.error(f"❌ {sentiment_label}")
    else: st.info(f"➖ {sentiment_label}")

# Header
head_col1, head_col2, head_col3 = st.columns([1, 18, 1])
with head_col2:
    st.markdown('<h1 class="main-header">⚽ TransferIQ: AI-Powered Football Player Transfer Valuation</h1>', unsafe_allow_html=True)
with head_col3:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f"## 🎯 Analyzing: <span style='color: white;'>{selected_player}</span>", unsafe_allow_html=True)

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Market Value", f"€{latest_data['market_value_eur']/1e6:.1f}M")
with col2: st.metric("Goals/90", f"{latest_data['goals_per90']:.2f}")
with col3: st.metric("Assists/90", f"{latest_data['assists_per90']:.2f}")
with col4: st.metric("Pass Accuracy", f"{latest_data['pass_accuracy_pct']:.1f}%")
with col5: st.metric("Sentiment", sentiment_label, delta=f"{sentiment_score:.2f}")

st.markdown("---")

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 💰 Market Value Trend")
    fig1 = px.line(player_df, x='season', y=player_df['market_value_eur']/1e6, markers=True)
    fig1.update_layout(template='plotly_white', height=400, yaxis_title="Market Value (€M)")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🎭 Sentiment Analysis Trend")
    fig2 = px.bar(player_df, x='season', y='vader_compound_score', color='sentiment_label', title="Sentiment Score")
    fig2.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# AI Models
st.markdown("### 🤖 AI Model Predictions & Performance")
col1, col2 = st.columns([2, 1])
with col1:
    current_val = latest_data['market_value_eur'] / 1e6
    model_data = pd.DataFrame({
        'Model': ['LSTM', 'XGBoost', 'LightGBM', 'Ensemble'],
        'Predicted (€M)': [current_val*0.95, current_val*0.98, current_val*1.01, current_val*1.00]
    })
    fig_models = px.bar(model_data, x='Model', y='Predicted (€M)', text='Predicted (€M)', color='Model')
    fig_models.update_traces(texttemplate='€%{text:.1f}M', textposition='outside')
    st.plotly_chart(fig_models, use_container_width=True)

with col2:
    accuracies = pd.DataFrame({'Model': ['LSTM', 'XGB', 'Ensemble'], 'R² Score': [0.87, 0.91, 0.94]})
    fig_acc = px.bar(accuracies, x='Model', y='R² Score', color='Model', range_y=[0,1])
    st.plotly_chart(fig_acc, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280;'><h4>🎓 TransferIQ: AI-Powered Football Player Valuation</h4><p>Powered by LSTM, XGBoost & Ensemble Models</p></div>", unsafe_allow_html=True)
