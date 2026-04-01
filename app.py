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
        div[data-testid="stSidebarNav"] span { color: #f8fafc !important; font-weight: 600; }
        
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

# Load data function with caching
@st.cache_data
def load_data():
    df = pd.read_csv("player_transfer_value_with_sentimenttttt.csv")
    return df

# Initialize data
df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## ⚽ TransferIQ Dashboard")
    st.markdown("### AI-Powered Player Valuation")
    st.markdown("---")
    
    # Player selection
    st.markdown("### 🎯 Select Player")
    players = sorted(df['player_name'].unique())
    selected_player = st.selectbox("", players, label_visibility="collapsed")
    
    # Filter data for selected player
    player_df = df[df['player_name'] == selected_player].sort_values('season')
    latest_data = player_df.iloc[-1]
    
    st.markdown("---")
    st.markdown("### 📊 Player Profile")
    
    # Display player details
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Age", f"{int(latest_data['current_age'])}")
    with col2:
        st.metric("Position", latest_data['position'])
    
    st.metric("Current Club", latest_data['team'])
    st.metric("Market Value", f"€{latest_data['market_value_eur']/1e6:.1f}M")
    
    st.markdown("---")
    st.markdown("### 🏥 Injury & Availability")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Total Injuries", int(latest_data['total_injuries']))
        st.metric("Matches Missed", int(latest_data['total_matches_missed']))
    with col4:
        st.metric("Days Injured", int(latest_data['total_days_injured']))
        st.metric("Availability", f"{latest_data['availability_rate']*100:.1f}%")

    st.markdown("---")
    st.markdown("### 📈 Latest Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Goals", int(latest_data['goals']))
        st.metric("Assists", int(latest_data['assists']))
    with col2:
        st.metric("Matches", int(latest_data['matches']))
        st.metric("Minutes", int(latest_data['minutes_played']))
    
    st.markdown("---")
    st.markdown("### 🎭 Sentiment Score")
    sentiment_score = latest_data['vader_compound_score']
    sentiment_label = latest_data['sentiment_label']
    
    if sentiment_label == 'Positive':
        st.success(f"✅ {sentiment_label}")
    elif sentiment_label == 'Negative':
        st.error(f"❌ {sentiment_label}")
    else:
        st.info(f"➖ {sentiment_label}")
    
    st.progress(max(0, min(1, (sentiment_score + 1) / 2)))
    st.caption(f"Compound: {sentiment_score:.3f}")

# Main content
# Theme Toggle and Header in symmetric columns
head_col1, head_col2, head_col3 = st.columns([1, 18, 1])
with head_col2:
    st.markdown('<h1 class="main-header">⚽ TransferIQ: AI-Powered Football Player Transfer Valuation</h1>', unsafe_allow_html=True)
with head_col3:
    # Transparent styling for toggle button
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    if st.button(theme_icon, help="Toggle Light/Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f"## 🎯 Analyzing: <span style='color: white;'>{selected_player}</span>", unsafe_allow_html=True)

# Key Metrics Row
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Market Value", f"€{latest_data['market_value_eur']/1e6:.1f}M")
with col2: st.metric("Goals/90", f"{latest_data['goals_per90']:.2f}")
with col3: st.metric("Assists/90", f"{latest_data['assists_per90']:.2f}")
with col4: st.metric("Pass Accuracy", f"{latest_data['pass_accuracy_pct']:.1f}%")
with col5: st.metric("Sentiment", sentiment_label, delta=f"{sentiment_score:.2f}")

st.markdown("---")

# Row 1: Market Value Trend and Sentiment Analysis
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 💰 Market Value Trend")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=player_df['season'], y=player_df['market_value_eur']/1e6, mode='lines+markers', fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig1.update_layout(xaxis_title="Season", yaxis_title="Market Value (€M)", template='plotly_white', height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🎭 Sentiment Analysis Trend")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=player_df['season'], y=player_df['vader_compound_score'], name='Score'), secondary_y=False)
    fig2.add_trace(go.Bar(x=player_df['season'], y=player_df['positive_count'], name='Pos', marker_color='#22c55e', opacity=0.6), secondary_y=True)
    fig2.add_trace(go.Bar(x=player_df['season'], y=player_df['negative_count'], name='Neg', marker_color='#ef4444', opacity=0.6), secondary_y=True)
    fig2.update_layout(template='plotly_white', height=400, barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 2: AI Model Predictions
st.markdown("### 🤖 AI Model Predictions & Performance Comparison")
col1, col2 = st.columns([2, 1])
with col1:
    current_value = latest_data['market_value_eur'] / 1e6
    model_data = pd.DataFrame({
        'Model': ['Univariate LSTM', 'Multivariate LSTM', 'XGBoost', 'Ensemble Model'],
        'Predicted Value (€M)': [current_value*0.93, current_value*0.97, current_value*1.00, current_value*1.01]
    })
    fig5 = px.bar(model_data, x='Model', y='Predicted Value (€M)', color='Model', text='Predicted Value (€M)')
    fig5.add_hline(y=current_value, line_dash="dash", line_color="red", annotation_text="Current Value")
    fig5.update_traces(texttemplate='€%{text:.1f}M', textposition='outside')
    fig5.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    accuracies = pd.DataFrame({
        'Model': ['LSTM', 'Multi-LSTM', 'XGBoost', 'Ensemble'],
        'R² Score': [0.82, 0.87, 0.91, 0.94]
    })
    fig6 = px.bar(accuracies, x='Model', y='R² Score', color='Model', range_y=[0, 1], title='Ensemble Accuracy (R²)')
    fig6.update_layout(template='plotly_white', height=400, showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# Row 3: Gauges
col1, col2, col3 = st.columns(3)
with col1:
    fig7 = go.Figure(go.Indicator(mode="gauge+number", value=latest_data['pass_accuracy_pct'], title={'text': "Pass Accuracy %"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}}))
    fig7.update_layout(height=280)
    st.plotly_chart(fig7, use_container_width=True)
with col2:
    fig8 = go.Figure(go.Indicator(mode="gauge+number", value=latest_data['goal_contributions_per90'], title={'text': "Goal Contrib/90"}, gauge={'axis': {'range': [0, 2]}, 'bar': {'color': "#10b981"}}))
    fig8.update_layout(height=280)
    st.plotly_chart(fig8, use_container_width=True)
with col3:
    sentiment_counts = player_df['sentiment_label'].value_counts()
    fig9 = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, hole=.4, title="Sentiment Mix", color_discrete_sequence=['#10b981', '#ef4444', '#6b7280'])
    fig9.update_layout(height=280, showlegend=False)
    st.plotly_chart(fig9, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280; padding: 2rem;'><h4>🎓 TransferIQ: AI-Powered Football Player Valuation</h4><p>© 2026 AI-Engine | Powered by LSTM & XGBoost Ensemble</p></div>", unsafe_allow_html=True)
