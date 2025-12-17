import pandas as pd
from datetime import datetime, timedelta
from polygon.rest import RESTClient

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
