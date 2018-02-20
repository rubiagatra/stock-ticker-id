import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from fbprophet import Prophet


class StockPredict:
    def __init__(self):
        self.close = None
        self.forecast = None
        self.prediction = None
        self.end = None
        self.pred_df = None
        self.name = None
        
    def get_data(self, name, start_date, end_date):
        self.name = name
        start_split = start_date.split('-')
        end_split = end_date.split('-') 
        start = datetime.datetime(int(start_split[0]), int(start_split[1]), int(start_split[2]))
        self.end = datetime.datetime(int(end_split[0]), int(end_split[1]), int(end_split[2]))
        df = web.DataReader(name=name, data_source='google', start=start, end=self.end)
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
        self.forecast.name = self.name
        self.pred_df = self.forecast[self.end:]


class MultiPredict:
    def __init__(self, list_name):
        self.list_name = list_name
        self.prediction = [] 

    def predict(self, start, end, days):
        for x in self.list_name:
            forecast = StockPredict()
            forecast.get_data(x, start, end)
            forecast.predict(days)
            self.prediction.append(forecast.pred_df)
        self.prediction = pd.DataFrame(self.prediction).T

    def to_csv(self, name):
        self.prediction.to_csv(name)


        


