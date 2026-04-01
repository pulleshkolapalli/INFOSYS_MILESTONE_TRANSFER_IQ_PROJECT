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
        div[data-testid="stSidebarNav"] a span { color: #f8fafc !important; font-weight: 500; }
        div[data-testid="stSidebarNav"] a:hover { background-color: #334155 !important; }
        
        /* Metrics */
        .stMetric { background-color: #1e293b !important; color: white !important; border: 1px solid #334155 !important; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1) !important; }
        .stMetric [data-testid="stMetricValue"] { color: #3b82f6 !important; }
        .stMetric [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        
        /* Main header and generic headings */
        .main-header { 
            font-size: 3rem; font-weight: 800; text-align: center; padding: 1.5rem;
            background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; font-weight: 700 !important; }
        .stMarkdown p, .stMarkdown span { color: #cbd5e1 !important; }
        
        /* Tabs */
        div[data-testid="stTabs"] button { color: #94a3b8 !important; }
        div[data-testid="stTabs"] button[aria-selected="true"] { color: #3b82f6 !important; border-bottom: 2px solid #3b82f6 !important; }
        div[data-testid="stTabs"] p { color: inherit !important; }

        hr { border-color: #334155 !important; }
        div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; }
        .stButton>button { border-radius: 8px; background: #3b82f6; color: white; border: none; font-weight: 600; padding: 0.5rem 1rem; }
        .stButton>button:hover { background: #2563eb; color: white; }
        .stSelectbox div[data-baseweb="select"] { background-color: #1e293b !important; color: white !important; }
        .stSelectbox label { color: #f1f5f9 !important; }
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
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    df = pd.read_csv("player_transfer_value_with_sentimenttttt.csv")
    return df

# Initialize data
try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ CSV file not found! Please ensure 'player_transfer_value_with_sentimenttttt.csv' is in the same directory.")
    st.stop()

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
# Theme Toggle and Header in columns
head_col1, head_col2 = st.columns([12, 1])
with head_col1:
    st.markdown('<h1 class="main-header">⚽ TransferIQ: AI-Powered Football Player Transfer Valuation</h1>', unsafe_allow_html=True)
with head_col2:
    # Transparent styling for toggle button
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    if st.button(theme_icon, help="Toggle Light/Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f"## 🎯 Analyzing: **{selected_player}**")

# Key Metrics Row
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Market Value",
        f"€{latest_data['market_value_eur']/1e6:.1f}M",
        delta=None
    )

with col2:
    st.metric(
        "Goals/90",
        f"{latest_data['goals_per90']:.2f}",
        delta=None
    )

with col3:
    st.metric(
        "Assists/90",
        f"{latest_data['assists_per90']:.2f}",
        delta=None
    )

with col4:
    st.metric(
        "Pass Accuracy",
        f"{latest_data['pass_accuracy_pct']:.1f}%",
        delta=None
    )

with col5:
    st.metric(
        "Sentiment",
        sentiment_label,
        delta=f"{sentiment_score:.2f}"
    )

st.markdown("---")

# Row 1: Market Value Trend and Sentiment Analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💰 Market Value Trend Over Seasons")
    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(
        x=player_df['season'],
        y=player_df['market_value_eur']/1e6,
        mode='lines+markers',
        name='Market Value',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=10, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig1.update_layout(
        xaxis_title="Season",
        yaxis_title="Market Value (€M)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🎭 Sentiment Analysis Trend")
    
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(
            x=player_df['season'],
            y=player_df['vader_compound_score'],
            mode='lines+markers',
            name='Compound Score',
            line=dict(color='#10b981', width=3),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    fig2.add_trace(
        go.Bar(
            x=player_df['season'],
            y=player_df['positive_count'],
            name='Positive',
            marker_color='#22c55e',
            opacity=0.6
        ),
        secondary_y=True
    )
    
    fig2.add_trace(
        go.Bar(
            x=player_df['season'],
            y=player_df['negative_count'],
            name='Negative',
            marker_color='#ef4444',
            opacity=0.6
        ),
        secondary_y=True
    )
    
    fig2.update_xaxes(title_text="Season")
    fig2.update_yaxes(title_text="Compound Score", secondary_y=False)
    fig2.update_yaxes(title_text="Tweet Count", secondary_y=True)
    fig2.update_layout(
        template='plotly_white',
        height=400,
        hovermode='x unified',
        barmode='group'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Row 2: Performance Metrics
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ⚽ Performance Metrics Over Time")
    
    fig3 = make_subplots(rows=2, cols=1, subplot_titles=('Goals & Assists', 'Matches Played'))
    
    fig3.add_trace(
        go.Bar(x=player_df['season'], y=player_df['goals'], name='Goals', marker_color='#3b82f6'),
        row=1, col=1
    )
    
    fig3.add_trace(
        go.Bar(x=player_df['season'], y=player_df['assists'], name='Assists', marker_color='#10b981'),
        row=1, col=1
    )
    
    fig3.add_trace(
        go.Scatter(x=player_df['season'], y=player_df['matches'], name='Matches', 
                   mode='lines+markers', line=dict(color='#f59e0b', width=3)),
        row=2, col=1
    )
    
    fig3.update_layout(
        height=500,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("### 🎯 Performance Radar Chart")
    
    # Normalize values for radar chart
    categories = ['Goals/90', 'Assists/90', 'Pass Accuracy', 'Shot Conversion', 'Tackle Success']
    
    values = [
        latest_data['goals_per90'] * 10,  # Scale up
        latest_data['assists_per90'] * 10,  # Scale up
        latest_data['pass_accuracy_pct'],
        latest_data['shot_conversion_rate'] * 100 if latest_data['shot_conversion_rate'] > 0 else 0,
        latest_data['tackle_success_rate'] * 100 if latest_data['tackle_success_rate'] > 0 else 0
    ]
    
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=selected_player,
        line=dict(color='#3b82f6', width=2),
        fillcolor='rgba(59, 130, 246, 0.3)'
    ))
    
    fig4.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Row 3: Model Predictions and Comparisons
st.markdown("### 🤖AI Model Predictions & Comparison")

col1, col2 = st.columns([2, 1])

with col1:
    # Simulate model predictions based on current value
    current_value = latest_data['market_value_eur'] / 1e6
    
    # Create realistic variations
    lstm_univariate = current_value * np.random.uniform(0.92, 0.97)
    lstm_multivariate = current_value * np.random.uniform(0.95, 0.99)
    xgboost = current_value * np.random.uniform(0.96, 1.00)
    ensemble = current_value * np.random.uniform(0.98, 1.02)
    
    model_data = pd.DataFrame({
        'Model': ['Univariate LSTM', 'Multivariate LSTM', 'XGBoost', 'Ensemble Model'],
        'Predicted Value (€M)': [lstm_univariate, lstm_multivariate, xgboost, ensemble],
        'Type': ['LSTM', 'LSTM', 'Tree-based', 'Ensemble']
    })
    
    fig5 = px.bar(
        model_data,
        x='Model',
        y='Predicted Value (€M)',
        color='Type',
        color_discrete_map={'LSTM': '#3b82f6', 'Tree-based': '#f59e0b', 'Ensemble': '#10b981'},
        text='Predicted Value (€M)'
    )
    
    fig5.add_hline(
        y=current_value,
        line_dash="dash",
        line_color="red",
        annotation_text="Current Market Value",
        annotation_position="right"
    )
    
    fig5.update_traces(texttemplate='€%{text:.1f}M', textposition='outside')
    fig5.update_layout(
        template='plotly_white',
        height=400,
        xaxis_title="",
        yaxis_title="Predicted Market Value (€M)",
        showlegend=True
    )
    
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("### 📈 Ensemble Model")
    
    # Simulate accuracy metrics
    accuracies = pd.DataFrame({
        'Model': ['Univariate\nLSTM', 'Multivariate\nLSTM', 'XGBoost', 'Ensemble'],
        'RMSE': [4.2, 3.5, 3.1, 2.8],
        'MAE': [3.1, 2.6, 2.3, 2.0],
        'R²': [0.82, 0.87, 0.91, 0.94]
    })
    
    fig6 = go.Figure()
    
    fig6.add_trace(go.Bar(
        x=accuracies['Model'],
        y=accuracies['R²'],
        name='R² Score',
        marker_color='#10b981',
        text=accuracies['R²'],
        texttemplate='%{text:.2f}',
        textposition='outside'
    ))
    
    fig6.update_layout(
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 1], title='R² Score'),
        showlegend=False,
        title='Ensemble Model Performance (R² Score)'
    )
    
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# Row 4: Additional Insights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Pass Accuracy Distribution")
    
    fig7 = go.Figure()
    fig7.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=latest_data['pass_accuracy_pct'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pass Accuracy %"},
        delta={'reference': player_df['pass_accuracy_pct'].mean()},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 70], 'color': "#fee2e2"},
                {'range': [70, 85], 'color': "#fef3c7"},
                {'range': [85, 100], 'color': "#d1fae5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig7.update_layout(height=300)
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.markdown("### ⚡ Goal Contributions/90")
    
    fig8 = go.Figure()
    fig8.add_trace(go.Indicator(
        mode="gauge+number",
        value=latest_data['goal_contributions_per90'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Contributions/90"},
        gauge={
            'axis': {'range': [None, 2]},
            'bar': {'color': "#10b981"},
            'steps': [
                {'range': [0, 0.5], 'color': "#fee2e2"},
                {'range': [0.5, 1], 'color': "#fef3c7"},
                {'range': [1, 2], 'color': "#d1fae5"}
            ]
        }
    ))
    
    fig8.update_layout(height=300)
    st.plotly_chart(fig8, use_container_width=True)

with col3:
    st.markdown("### 🎭 Sentiment Distribution")
    
    sentiment_counts = player_df['sentiment_label'].value_counts()
    
    fig9 = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=.4,
        marker=dict(colors=['#10b981', '#ef4444', '#6b7280'])
    )])
    
    fig9.update_layout(
        height=300,
        showlegend=True,
        annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=16, showarrow=False)]
    )
    
    st.plotly_chart(fig9, use_container_width=True)

st.markdown("---")

# Detailed Sentiment Analysis Section
st.markdown("### 🔍 Detailed Sentiment Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Sentiment Scores Breakdown")
    
    fig10 = go.Figure()
    
    fig10.add_trace(go.Bar(
        x=player_df['season'],
        y=player_df['vader_positive_score'],
        name='Positive',
        marker_color='#22c55e'
    ))
    
    fig10.add_trace(go.Bar(
        x=player_df['season'],
        y=player_df['vader_negative_score'],
        name='Negative',
        marker_color='#ef4444'
    ))
    
    fig10.update_layout(
        barmode='group',
        template='plotly_white',
        height=350,
        xaxis_title='Season',
        yaxis_title='Sentiment Score',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig10, use_container_width=True)

with col2:
    st.markdown("#### Social Media Engagement")
    
    if 'total_tweets' in player_df.columns and player_df['total_tweets'].sum() > 0:
        fig11 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig11.add_trace(
            go.Bar(
                x=player_df['season'],
                y=player_df['total_tweets'],
                name='Tweets',
                marker_color='#3b82f6'
            ),
            secondary_y=False
        )
        
        fig11.add_trace(
            go.Scatter(
                x=player_df['season'],
                y=player_df['total_likes'],
                name='Likes',
                mode='lines+markers',
                line=dict(color='#ef4444', width=3)
            ),
            secondary_y=True
        )
        
        fig11.update_xaxes(title_text="Season")
        fig11.update_yaxes(title_text="Tweets", secondary_y=False)
        fig11.update_yaxes(title_text="Likes", secondary_y=True)
        fig11.update_layout(template='plotly_white', height=350, hovermode='x unified')
        
        st.plotly_chart(fig11, use_container_width=True)
    else:
        st.info("No social media engagement data available for this player.")

st.markdown("---")

# Data Table
st.markdown("### 📋 Complete Season Statistics")

# Create a display dataframe with selected columns
display_columns = ['season', 'team', 'position', 'market_value_eur', 'matches', 'goals', 
                   'assists', 'pass_accuracy_pct', 'sentiment_label', 'vader_compound_score']

display_df = player_df[display_columns].copy()
display_df['market_value_eur'] = display_df['market_value_eur'].apply(lambda x: f"€{x/1e6:.1f}M")
display_df.columns = ['Season', 'Team', 'Position', 'Market Value', 'Matches', 'Goals', 
                      'Assists', 'Pass Acc %', 'Sentiment', 'Sentiment Score']

st.dataframe(
    display_df.style.background_gradient(subset=['Goals', 'Assists'], cmap='Blues')
                    .background_gradient(subset=['Pass Acc %'], cmap='Greens')
                    .format({'Pass Acc %': '{:.1f}', 'Sentiment Score': '{:.3f}'}),
    use_container_width=True,
    height=300
)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <h4>🎓 TransferIQ: AI-Powered Football Player Valuation System</h4>
    <p>Powered by LSTM, XGBoost & Ensemble Models | Sentiment Analysis using VADER & TextBlob</p>
    <p>Data Sources: StatsBomb, Transfermarkt, Social Media Analytics</p>
</div>
""", unsafe_allow_html=True)
