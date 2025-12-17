import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

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
