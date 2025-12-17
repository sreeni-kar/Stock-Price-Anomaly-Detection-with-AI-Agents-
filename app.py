import streamlit as st
from crewai import Agent, Crew, Task, LLM, Process
import warnings
import sys
import os

from tools.stock_analysis_tool import run_swarm, set_polygon_api_key

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
        set_polygon_api_key(user_polygon_key)
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

# --- THE CHATBOT UI & CREW ORCHESTRATION ---

st.title("📈 Agentic Stock Forecaster")
st.markdown("Chat with a swarm of AI agents that run real-time statistical models (ARIMA-GARCH).")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                llm = LLM(model="gemini/gemini-1.5-flash")

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
