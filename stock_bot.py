import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from polygon.rest import RESTClient
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from crewai import Agent, Crew, Task, LLM, Process
from crewai.tools import tool
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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("Enter your API keys to activate the agents.")
    
    # Input for API Keys
    user_polygon_key = st.text_input("Polygon.io API Key", type="password", help="Get from polygon.io")
    user_google_key = st.text_input("Google Gemini API Key", type="password", help="Get from aistudio.google.com")
    
    st.divider()
    st.markdown("### 🤖 Active Agents")
    st.markdown("- **Market Data Agent**: Fetches raw price history.")
    st.markdown("- **Quant Agent**: Runs ARIMA/GARCH models.")
    st.markdown("- **Analyst Agent**: Synthesizes the final report.")

# --- SHARED GLOBAL KEYS (For Tool Access) ---
# We use a global variable pattern here so the decorated tool can access the dynamic sidebar key
POLYGON_KEY_GLOBAL = user_polygon_key

# --- 1. THE ANALYTICAL ENGINE (Backend Logic) ---

class MarketDataAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_and_prepare(self, ticker):
        # Clean ticker input
        ticker = ticker.upper().strip()
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365 * 2) # 2 Years lookback
        
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

    def predict_tomorrow(self):
        try:
            # Use subset to prevent timeout/recursion errors
            data_subset = self.data[-300:] if len(self.data) > 300 else self.data
            data_arr = np.array(data_subset, dtype=np.float64)
            
            # ARIMA Model
            model = ARIMA(data_arr, order=(1,1,0))
            model_fit = model.fit(method_kwargs={"maxiter": 50})
            pred_price = float(model_fit.forecast(steps=1)[0])
            
            # GARCH Model for Volatility
            resid = model_fit.resid[-50:].astype(float) * 100
            garch = arch_model(resid, vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off', options={'maxiter': 50})
            pred_vol = float(np.sqrt(garch_fit.forecast(horizon=1).variance.iloc[-1, 0]) / 100)
            
            return pred_price, pred_vol
            
        except Exception as e:
            # Fallback logic
            pred_price = float(self.data.iloc[-5:].mean())
            pred_vol = float(self.data.iloc[-30:].std())
            return pred_price, pred_vol

class PredictionAgent:
    def __init__(self, ticker, current_price, sma, pred_price, pred_vol):
        self.ticker = ticker
        self.current_price = current_price
        self.sma = sma
        self.pred_price = pred_price
        self.pred_vol = pred_vol

    def get_report(self):
        upper = self.pred_price + (1.96 * self.pred_vol)
        lower = self.pred_price - (1.96 * self.pred_vol)
        
        return {
            "Ticker": self.ticker,
            "Current": self.current_price,
            "SMA_50": self.sma,
            "Prediction": self.pred_price,
            "Volatility": self.pred_vol,
            "Upper": upper,
            "Lower": lower
        }

# --- TOOL DEFINITION ---
@tool("run_swarm")
def run_swarm(ticker: str):
    """
    Executes a stock analysis swarm for a given ticker. 
    Use this tool when the user asks about a specific stock symbol.
    Returns a dictionary with price predictions and volatility.
    """
    # Use the key from the sidebar (global scope)
    if not POLYGON_KEY_GLOBAL:
        return {"Error": "Polygon API Key is missing. Please ask the user to provide it in the sidebar."}

    data_agent = MarketDataAgent(POLYGON_KEY_GLOBAL)
    
    # 1. Fetch
    df = data_agent.fetch_and_prepare(ticker)
    if df is None or len(df) < 50: 
        return {"Error": f"Failed to fetch data for {ticker}. The ticker might be invalid."}
    
    # 2. Predict
    quant = QuantAgent(df['close'])
    next_price, next_vol = quant.predict_tomorrow()
    
    # 3. Report
    predictor = PredictionAgent(
        ticker, 
        float(df['close'].iloc[-1]), 
        float(df['SMA_50'].iloc[-1]), 
        float(next_price), 
        float(next_vol)
    )
    report = predictor.get_report()
    
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

# Chat Input Handler
if prompt := st.chat_input("Ask about a stock (e.g., 'Should I buy NVDA?')"):
    
    # 1. Check for API Keys
    if not user_google_key or not user_polygon_key:
        st.error("Please enter both API Keys in the sidebar to proceed.")
        st.stop()
        
    os.environ["GOOGLE_API_KEY"] = user_google_key
    
    # 2. Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Initialize CrewAI (Re-initialized per run to catch dynamic keys)
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
                    description=f"Using the data from the previous task, answer the user: '{prompt}'. Provide specific numbers (Target, Confidence Interval).",
                    expected_output="A helpful, professional natural language response recommending action based on the data.",
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
                
                # Output the result
                st.markdown(result)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")