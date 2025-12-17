import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error

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
