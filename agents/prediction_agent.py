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
