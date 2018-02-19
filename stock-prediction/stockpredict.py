import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from fbprophet import Prophet


class StockPredict:
    def __init__(self):
        self.close = None
        self.forecast = None

    def get_data(self, name, start_date, end_date):
        start_split = start_date.split('-')
        end_split = end_date.split('-') 
        start = datetime.datetime(int(start_split[0]), int(start_split[1]), int(start_split[2]))
        end = datetime.datetime(int(end_split[0]), int(end_split[1]), int(end_split[2]))
        df = web.DataReader(name=name, data_source='google', start=start, end=end)
        self.close = df['Close']

    def predict(self, days):
        model = Prophet()
        close_df = self.close.reset_index().rename(columns={'Date':'ds', 'Close':'y'})
        close_df['y'] = np.log(close_df['y'])
        model.fit(close_df)
        future = model.make_future_dataframe(periods=days) 
        forecast = model.predict(future)
        self.forecast = forecast.set_index('ds')['yhat']
        self.forecast = np.exp(self.forecast)

    

