import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon.rest import RESTClient
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from crewai import Agent, Crew, Task, LLM, Process
from crewai.tools import tool
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
import warnings
import sys
import os
import gc

# --- STREAMLIT CONFIGURATION ---
st.set_page_config(
    page_title="AI Financial Analyst Swarm",
    page_icon="📈",
    layout="wide"
)

# --- RECURSION LIMIT FIX ---
try:
    sys.setrecursionlimit(20000)
except Exception:
    pass

warnings.filterwarnings("ignore")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- LOAD SECRETS (SILENTLY) ---
try:
    if "POLYGON_API_KEY" in st.secrets and "GOOGLE_API_KEY" in st.secrets:
        user_polygon_key = st.secrets["POLYGON_API_KEY"]
        user_google_key = st.secrets["GOOGLE_API_KEY"]
    else:
        user_polygon_key = None
        user_google_key = None
except (FileNotFoundError, Exception):
    user_polygon_key = None
    user_google_key = None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🤖 Agent Status")
    
    if user_polygon_key and user_google_key:
        st.success("System Operational")
    else:
        st.error("System Configuration Error: Keys missing in secrets.")
    
    st.divider()
    st.markdown("### Active Agents")
    st.markdown("- **Market Data Agent**: Fetches raw price history.")
    st.markdown("- **Quant Agent**: Runs ARIMA/GARCH models.")
    st.markdown("- **Analyst Agent**: Synthesizes the final report.")
    st.markdown("- **Viz Agent**: Generates trend & risk charts.")

# --- SHARED GLOBAL KEYS (For Tool Access) ---
POLYGON_KEY_GLOBAL = user_polygon_key

# --- 1. THE ANALYTICAL ENGINE (Backend Logic) ---

class MarketDataAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_and_prepare(self, ticker):
        ticker = ticker.upper().strip()
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * 2)
        
        aggs = []
        try:
            for a in self.client.list_aggs(
                ticker, 1, 'day', 
                start_date.strftime("%Y-%m-%d"), 
                end_date.strftime("%Y-%m-%d"), 
                limit=50000
            ):
                aggs.append(a)
        except Exception as e:
            return None

        if not aggs: return None

        df = pd.DataFrame(aggs)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        return df

class QuantAgent:
    def __init__(self, data):
        self.data = data 
        self.history = None
        self.predictions = []
        self.vol_predictions = []
        self.actuals = []

    def perform_walk_forward_validation(self, test_days=20):
        # Limit data to recent history for speed
        subset = self.data[-300:] if len(self.data) > 300 else self.data
        
        train = subset[:-test_days].tolist()
        test = subset[-test_days:].tolist()
        
        self.history = [x for x in train]
        self.predictions = []
        self.vol_predictions = []
        self.actuals = test

        # Rolling forecast
        for t in range(len(test)):
            try:
                # Faster ARIMA order (1,1,0) for speed in Streamlit
                model = ARIMA(self.history, order=(1,1,0))
                model_fit = model.fit(method_kwargs={"maxiter": 20})
                yhat = model_fit.forecast()[0]
                self.predictions.append(yhat)
                
                # GARCH
                resid = model_fit.resid
                # Handle edge case where residues are too small/flat
                safe_resid = resid[-100:] * 100
                if np.all(safe_resid == 0):
                    vol_forecast = 0.0
                else:
                    garch = arch_model(safe_resid, vol='Garch', p=1, q=1)
                    garch_fit = garch.fit(disp='off', options={'maxiter': 20})
                    vol_forecast = np.sqrt(garch_fit.forecast(horizon=1).variance.iloc[-1, 0]) / 100
                
                self.vol_predictions.append(vol_forecast)
            except:
                # Fallback if model fails on a step
                self.predictions.append(self.history[-1])
                self.vol_predictions.append(0.0)

            self.history.append(test[t])
            
        rmse = np.sqrt(mean_squared_error(self.actuals, self.predictions))
        return rmse, self.predictions, self.vol_predictions

    def predict_tomorrow(self):
        try:
            data_subset = self.data[-300:] if len(self.data) > 300 else self.data
            data_arr = np.array(data_subset, dtype=np.float64)
            
            model = ARIMA(data_arr, order=(1,1,0))
            model_fit = model.fit(method_kwargs={"maxiter": 50})
            pred_price = float(model_fit.forecast(steps=1)[0])
            
            resid = model_fit.resid[-50:].astype(float) * 100
            garch = arch_model(resid, vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off', options={'maxiter': 50})
            pred_vol = float(np.sqrt(garch_fit.forecast(horizon=1).variance.iloc[-1, 0]) / 100)
            
            return pred_price, pred_vol
        except Exception:
            pred_price = float(self.data.iloc[-5:].mean())
            pred_vol = float(self.data.iloc[-30:].std())
            return pred_price, pred_vol

class PredictionAgent:
    def __init__(self, ticker, current_price, sma, pred_price, pred_vol, rmse=0.0):
        self.ticker = ticker
        self.current_price = current_price
        self.sma = sma
        self.pred_price = pred_price
        self.pred_vol = pred_vol
        self.rmse = rmse

    def get_report(self):
        upper = self.pred_price + (1.96 * self.pred_vol)
        lower = self.pred_price - (1.96 * self.pred_vol)
        
        return {
            "Ticker": self.ticker,
            "Current": self.current_price,
            "SMA_50": self.sma,
            "RMSE": self.rmse,
            "Prediction": self.pred_price,
            "Volatility": self.pred_vol,
            "Upper": upper,
            "Lower": lower
        }

class VisualizationAgent:
    def __init__(self, ticker, test_dates, actuals, predictions, vol_preds, report):
        self.ticker = ticker
        self.dates = test_dates
        self.actuals = actuals
        self.predictions = predictions
        self.vol_preds = vol_preds
        self.report = report 

    def plot(self):
        # Create figure using Streamlit-compatible matplotlib settings
        plt.style.use('bmh') # Cleaner style
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2.5, 1]})
        plt.subplots_adjust(hspace=0.35)

        # --- PANEL 1: Price & Forecast ---
        ax1.plot(self.dates, self.actuals, label='Actual Price', color='black', alpha=0.6, linewidth=1.5)
        ax1.plot(self.dates, self.predictions, label='AI Backtest', color='blue', linestyle='--', alpha=0.5)
        
        # Markers
        next_date = self.dates[-1] + timedelta(days=1)
        
        # Target
        ax1.errorbar(next_date, self.report['Prediction'], 
                    yerr=[[self.report['Prediction'] - self.report['Lower']], [self.report['Upper'] - self.report['Prediction']]],
                    fmt='o', color='red', ecolor='red', capsize=5, label='Next Day Forecast (95% CI)')
        
        ax1.set_title(f"{self.ticker} - Price Trend & Next Day Prediction", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # --- PANEL 2: Volatility ---
        # Align lengths if mismatch due to diffing
        min_len = min(len(self.dates), len(self.vol_preds))
        ax2.plot(self.dates[-min_len:], self.vol_preds[-min_len:], color='orange', alpha=0.8, label='Conditional Volatility (Sigma)')
        
        # Next day vol marker
        ax2.scatter(next_date, self.report['Volatility'], color='red', marker='x', s=100, label='Forecasted Volatility')
        
        ax2.set_title(f"{self.ticker} - Market Volatility (GARCH Model)", fontsize=12)
        ax2.set_ylabel("Volatility (Std Dev)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        return fig

# --- TOOL DEFINITION ---
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

# --- 2. THE CHATBOT UI & CREW ORCHESTRATION ---

st.title("📈 Agentic Stock Forecaster")
st.markdown("Chat with a swarm of AI agents that run real-time statistical models (ARIMA-GARCH).")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Check if this message had a plot attached (custom key logic could be added here, 
        # but for simplicity we just show the latest plot after new runs)

# Chat Input Handler
if prompt := st.chat_input("Ask about a stock (e.g., 'Should I buy NVDA?')"):
    
    if not user_google_key or not user_polygon_key:
        st.error("System is missing API keys. Please check secrets.toml file.")
        st.stop()
        
    os.environ["GOOGLE_API_KEY"] = user_google_key
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🤖 Agents are thinking...", expanded=True) as status:
            
            try:
                # -- Agent Setup --
                llm = LLM(model="gemini/gemini-2.5-flash")

                retriever_agent = Agent(
                    role="Financial Data Retriever",
                    goal="Extract the stock ticker from the user query: {query} and run the analysis tool.",
                    backstory="You are a data engineer who provides precise stock data.",
                    tools=[run_swarm],
                    llm=llm,
                    verbose=True
                )

                response_synthesizer_agent = Agent(
                    role="Senior Investment Analyst",
                    goal="Synthesize the statistical report into a clear buy/sell/hold assessment.",
                    backstory="You are a veteran analyst. You explain complex math (ARIMA/GARCH) in simple terms.",
                    llm=llm,
                    verbose=True
                )

                # -- Task Setup --
                retrieval_task = Task(
                    description=f"Identify the ticker in the query: '{prompt}' and use run_swarm tool to get data.",
                    expected_output="A JSON object or Dictionary containing the stock analysis stats.",
                    agent=retriever_agent
                )

                response_task = Task(
                    description=(
                        f"Using the data from the previous task, answer the user: '{prompt}'. "
                        "Provide specific numbers (Target, Confidence Interval). "
                        "\n\nIMPORTANT FORMATTING RULES:"
                        "\n- Do NOT use LaTeX math formatting (like $...$ or $$...$$) for the text report."
                        "\n- Do NOT use italic formatting that removes spaces."
                        "\n- Use standard Markdown for bolding (e.g., **$150.00**)."
                        "\n- Ensure there are normal spaces between words."
                    ),
                    expected_output="A helpful, professional natural language response recommending action based on the data. Clean plain text with standard Markdown only.",
                    agent=response_synthesizer_agent
                )

                # -- Crew Execution --
                crew = Crew(
                    agents=[retriever_agent, response_synthesizer_agent],
                    tasks=[retrieval_task, response_task],
                    process=Process.sequential
                )

                status.write("🔍 Retrieving market data...")
                result = crew.kickoff(inputs={"query": prompt})
                status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                
                # Output the text result
                st.markdown(result)
                
                # Output the plot if it exists
                if 'latest_plot' in st.session_state:
                    st.pyplot(st.session_state['latest_plot'])
                    # Clean up to prevent showing old plots later
                    del st.session_state['latest_plot']
                
                # Save to history (Note: plots in history are hard in pure Streamlit chat without complex state, 
                # so we just save the text for history)
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                error_msg = str(e)
                if "native provider not available" in error_msg:
                    status.update(label="❌ Configuration Error", state="error")
                    st.error("🚨 **Missing Driver Error**")
                    st.markdown("CrewAI requires a specific Google driver. Please run this command in your terminal:")
                    st.code('pip install "crewai[google-genai]"', language="bash")
                else:
                    status.update(label="❌ Error occurred", state="error")
                    st.error(f"An error occurred: {str(e)}")