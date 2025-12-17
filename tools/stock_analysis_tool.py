import streamlit as st
import gc
from crewai.tools import tool
from agents.market_data_agent import MarketDataAgent
from agents.quant_agent import QuantAgent
from agents.prediction_agent import PredictionAgent
from agents.visualization_agent import VisualizationAgent

# --- SHARED GLOBAL KEYS (For Tool Access) ---
POLYGON_KEY_GLOBAL = None 

def set_polygon_api_key(key):
    global POLYGON_KEY_GLOBAL
    POLYGON_KEY_GLOBAL = key

@tool("run_swarm")
def run_swarm(ticker: str):
    """
    Executes a stock analysis swarm for a given ticker. 
    Returns a dictionary with price predictions and volatility.
    Generates a plot in the background.
    """
    if not POLYGON_KEY_GLOBAL:
        return {"Error": "Polygon API Key is missing. Check secrets.toml."}

    data_agent = MarketDataAgent(POLYGON_KEY_GLOBAL)
    
    # 1. Fetch
    df = data_agent.fetch_and_prepare(ticker)
    if df is None or len(df) < 50: 
        return {"Error": f"Failed to fetch data for {ticker}."}
    
    # 2. Predict & Backtest
    quant = QuantAgent(df['close'])
    
    # Run a quick backtest for plotting (Last 20 days)
    # Note: We do this inside the tool so we can visualize it
    rmse, preds, vols = quant.perform_walk_forward_validation(test_days=20)
    
    # Predict next day
    next_price, next_vol = quant.predict_tomorrow()
    
    # 3. Report
    predictor = PredictionAgent(
        ticker, 
        float(df['close'].iloc[-1]), 
        float(df['SMA_50'].iloc[-1]), 
        float(next_price), 
        float(next_vol),
        float(rmse)
    )
    report = predictor.get_report()
    
    # 4. Generate Plot (Save to Session State)
    test_dates = df.index[-20:]
    actuals = df['close'].iloc[-20:]
    
    viz = VisualizationAgent(ticker, test_dates, actuals, preds, vols, report)
    fig = viz.plot()
    
    # Save fig to session state to display in main loop
    st.session_state['latest_plot'] = fig
    
    status = 'BULLISH' if report['Current'] > report['SMA_50'] else 'BEARISH'
    
    enhanced_report = {
        **report,
        "Status": status,
        "Summary": (
            f"{ticker} is trading at ${report['Current']:.2f}. "
            f"Trend: {status} (vs SMA50: ${report['SMA_50']:.2f}). "
            f"Model predicts: ${report['Prediction']:.2f} ± ${report['Volatility']:.2f}."
        )
    }
    
    gc.collect()
    return enhanced_report
