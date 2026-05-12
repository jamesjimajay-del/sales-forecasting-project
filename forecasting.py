import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# Generate sample monthly sales data
np.random.seed(42)
dates = pd.date_range(start='2022-01-01', periods=36, freq='M')
sales = 1200 + np.cumsum(np.random.normal(15, 80, 36)) + 300*np.sin(np.arange(36)*2*np.pi/12)
df = pd.DataFrame({'ds': dates, 'y': sales})

# Prophet Model
model = Prophet(yearly_seasonality=True, weekly_seasonality=False)
model.fit(df)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# Plot
fig = model.plot(forecast)
plt.title('Sales Forecast - Next 12 Months')
plt.show()