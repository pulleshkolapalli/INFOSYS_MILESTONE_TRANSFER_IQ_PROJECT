# ⚽ TransferIQ: AI-Powered Football Player Transfer Valuation System

## 🎯 Project Overview

TransferIQ is a comprehensive AI-driven platform for predicting football player transfer values using advanced machine learning techniques, time-series forecasting, and sentiment analysis.

## 📋 Features

### ✨ Core Capabilities
- **Multi-Model AI Predictions**: LSTM, XGBoost, LightGBM, and Ensemble models
- **Sentiment Analysis**: VADER and TextBlob integration for social media sentiment
- **Interactive Visualizations**: 30+ dynamic charts and graphs
- **Multi-Page Dashboard**: Organized by project milestones
- **Real-Time Filtering**: Player-specific analysis and comparisons

### 📊 Pages
1. **Main Dashboard** - Comprehensive player analysis and predictions
2. **Week 1-2: Data Exploration** - EDA and feature engineering
3. **Week 3-4: Sentiment Analysis** - NLP and sentiment impact analysis
4. **Week 5-7: Model Development** - ML models and performance evaluation

## 🚀 Deployment Instructions

### Option 1: Streamlit Cloud (Recommended)

1. **Create a GitHub Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - TransferIQ Dashboard"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy"

### Option 2: Local Deployment

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Dashboard**
   - Open browser to: http://localhost:8501

### Option 3: Docker Deployment

1. **Create Dockerfile** (already included)
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t transferiq .
   docker run -p 8501:8501 transferiq
   ```

## 📁 File Structure

```
TransferIQ/
├── app.py                                    # Main dashboard
├── pages/
│   ├── 1_📊_Week_1-2_Data_Exploration.py    # Data analysis milestone
│   ├── 2_🎭_Week_3-4_Sentiment_Analysis.py  # Sentiment analysis milestone
│   └── 3_🤖_Week_5-7_Model_Development.py   # ML models milestone
├── player_transfer_value_with_sentimenttttt.csv  # Dataset
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

## 📊 Dataset Information

**File**: `player_transfer_value_with_sentimenttttt.csv`

**Key Columns**:
- Player Information: name, team, position, age
- Performance Metrics: goals, assists, matches, pass accuracy
- Market Data: market value, value tiers
- Sentiment Features: VADER scores, TextBlob scores, sentiment labels
- Engineered Features: goals/90, assists/90, injury metrics

**Size**: 1000+ player-season records

## 🛠️ Technologies Used

### Frontend & Visualization
- **Streamlit**: Interactive web framework
- **Plotly**: Dynamic charts and graphs
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### AI/ML Models (Simulated in Demo)
- **LSTM**: Time-series forecasting
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting
- **Ensemble**: Combined predictions

### NLP & Sentiment Analysis
- **VADER**: Social media sentiment
- **TextBlob**: Polarity and subjectivity

## 📈 Key Visualizations

1. **Market Value Trends**: Historical value progression
2. **Sentiment Analysis**: Compound scores and tweet distribution
3. **Performance Metrics**: Goals, assists, and efficiency stats
4. **Model Comparisons**: RMSE, MAE, R² scores
5. **Radar Charts**: Multi-dimensional player profiles
6. **Correlation Heatmaps**: Feature relationships
7. **Prediction Accuracy**: Actual vs predicted values

## 🎓 Academic Context

**Project**: Infosys Springboard Internship Milestone
**Duration**: 8 weeks
**Focus Areas**:
- Week 1-2: Data collection and preprocessing
- Week 3-4: Feature engineering and sentiment analysis
- Week 5-7: Model development and evaluation
- Week 8: Deployment and documentation

## 💡 Key Features

### Interactive Elements
- ✅ Player selector dropdown
- ✅ Multi-page navigation
- ✅ Dynamic filtering
- ✅ Responsive charts
- ✅ Real-time calculations

### Insights Provided
- 📊 Player performance trends
- 🎭 Sentiment impact on valuation
- 🤖 AI model predictions
- 📈 Comparative analysis
- 💰 Market value forecasting

## 🔧 Customization

### Modify Colors
Edit CSS in `app.py`:
```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #YOUR_COLOR1, #YOUR_COLOR2);
    }
</style>
""", unsafe_allow_html=True)
```

### Add New Metrics
In any page file:
```python
st.metric("New Metric", value, delta=change)
```

### Update Data
Replace `player_transfer_value_with_sentimenttttt.csv` with your updated dataset (maintain column structure)

## 📱 Mobile Responsiveness

The dashboard is fully responsive and works on:
- 💻 Desktop browsers
- 📱 Mobile devices
- 📲 Tablets

## 🐛 Troubleshooting

### Issue: CSV not found
**Solution**: Ensure `player_transfer_value_with_sentimenttttt.csv` is in the same directory as `app.py`

### Issue: Import errors
**Solution**: Run `pip install -r requirements.txt`

### Issue: Port already in use
**Solution**: Use a different port: `streamlit run app.py --server.port 8502`

## 📞 Support

For issues or questions:
1. Check the Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
2. Review error messages in the terminal
3. Verify all files are in correct locations

## 🎉 Success Metrics

After deployment, your dashboard will feature:
- ✅ 30+ interactive visualizations
- ✅ Multi-page navigation
- ✅ Real-time data filtering
- ✅ Professional UI/UX
- ✅ Comprehensive analytics
- ✅ Weekly milestone demonstrations
- ✅ AI model comparisons
- ✅ Sentiment analysis integration

## 📄 License

This project is for educational purposes as part of the Infosys Springboard internship program.

## 🙏 Acknowledgments

- **Data Sources**: StatsBomb Open Data, Transfermarkt
- **NLP Libraries**: VADER, TextBlob
- **Framework**: Streamlit
- **Visualization**: Plotly

---

**Built with ❤️ for football analytics and AI-powered insights**

🚀 **Ready to deploy! Follow the instructions above to get started.**
