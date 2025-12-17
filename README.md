# Stock Price Anomalay Detection with AI Agents

This project deploys a swarm of AI agents to perform real-time stock analysis and forecasting. Users can interact with the agents through a Streamlit-based chat interface to get insights on stock performance, trends, and predictions.

## Features

- **Agentic Workflow**: Utilizes the CrewAI framework to orchestrate a team of specialized AI agents.
- **Real-time Data**: Fetches up-to-date stock data from Polygon.io.
- **Statistical Modeling**: Employs ARIMA and GARCH models for time-series forecasting and volatility analysis.
- **Interactive UI**: A Streamlit web application for easy interaction and visualization.
- **Modular & Extensible**: The code is structured into distinct modules for agents, tools, and the main application, making it easy to extend.

## Project Structure

```
.
├── agents
│   ├── __init__.py
│   ├── market_data_agent.py
│   ├── prediction_agent.py
│   ├── quant_agent.py
│   └── visualization_agent.py
├── tools
│   ├── __init__.py
│   └── stock_analysis_tool.py
├── app.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- An environment manager (like `venv` or `conda`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI-Financial-Analyst-Swarm.git
    cd AI-Financial-Analyst-Swarm
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API keys:**

    Create a `.streamlit/secrets.toml` file in the project root with your API keys:

    ```toml
    POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
    ```

### Running the Application

To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your web browser to use the application.

## How It Works

The application is built around a "swarm" of AI agents, each with a specific role:

1.  **Market Data Agent**: Fetches historical stock data using the Polygon API.
2.  **Quant Agent**: Performs quantitative analysis, including ARIMA for price forecasting and GARCH for volatility modeling.
3.  **Analyst Agent**: Interprets the results from the Quant Agent to provide a qualitative assessment (e.g., buy/sell/hold).
4.  **Visualization Agent**: Generates plots of the price history, forecasts, and volatility.

These agents work together in a `Crew` to respond to user queries, providing a comprehensive analysis of the requested stock.
