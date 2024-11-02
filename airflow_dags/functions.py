from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from scipy import stats 
import statsmodels.api as sm
import statsmodels.api as sm
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from datetime import date
import  os
from dotenv import load_dotenv

load_dotenv()


# Предобработка полученных от API данных для подачи в модели
def prepare_data(df_zero):
   

   crypto_data = df_zero.drop(11, axis=1)
    
   crypto_data.columns = [
                                'open_time', 
                                'open', 
                                'high', 
                                'low', 
                                'close', 
                                'volume', 
                                'close_time',
                                'quote_volume',
                                'count',
                                'taker_buy_volume',
                                'taker_buy_quote_volume'
                            ]
    
   crypto_data['open_time'] = pd.to_datetime(crypto_data['open_time'], unit='ms')
   crypto_data['close_time'] = pd.to_datetime(crypto_data['close_time'], unit='ms')
    
   col = ['open', 'high', 'low',
           'close', 'volume',
           'quote_volume',
           'count', 'taker_buy_volume',
           'taker_buy_quote_volume']

   for i in col:
        crypto_data[i]=crypto_data[i].astype('float64').round(2)
    
   df = crypto_data
   ds_el = df.iloc[[13]] # В данной переменной берется запись, соотв. вчерашнему дню для отправки в БД
    

   # Формироввание лагов для передачи в модель
   for i in range(1, 15):
        df[f"lag_open{i}"] = df['open'].shift(i)
    
   for i in range(1, 15):
        df[f"lag_volume{i}"] = df['volume'].shift(i)
    
   for i in range(1, 15):
        df[f"lag_count{i}"] = df['count'].shift(i)
    
   df.iat[14, 1] = df.iloc[14, 4]
    
   df.dropna(inplace=True)
   df.drop(['close_time', 'high', 'low', 'taker_buy_volume','taker_buy_quote_volume', 'quote_volume', 'volume','count' ], axis=1, inplace=True)
   df.set_index('open_time', inplace=True)
   df = df.drop(['close'], axis=1)


   return df, ds_el


# Прогноз DL моделью 
def dl_model_predict(df):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load('/opt/airflow/models/model_scripted.pt') # Путь модели DL с предобученными весами
    model.eval()
    scaler_test = preprocessing.StandardScaler().fit_transform(df) 
    data = np.array(scaler_test)

    data_tensor = torch.from_numpy(data).to(device) 

    model.eval()   # В режим инференса
    with torch.no_grad():  
        logits = model(data_tensor.to(torch.float32))
        output_mlp = logits.argmax(dim=1).detach().cpu().sum().item()

    return output_mlp



# Обучение ARIMA модели (с логарифмированием ряда) и формирование прогноза
def fitting_1(y_train, y_test):
    
    y_train = np.log(y_train)
    
    history = []
    history = [x for x in y_train]
    
    # Эта часть для прогноза на следующий день, на основе имеющихся данных за сегодня
    obs = np.log(y_test[0])
    history.append(obs)
    model = sm.tsa.statespace.SARIMAX(history, exog=None, order=(2, 1, 0), seasonal_order=(0, 0, 0, 0)) 
    results = model.fit()
    forecast = results.forecast(steps=1, exog=None)
    yhat = np.exp(forecast[0])
 
    # Эта часть для прогноза на сегодня, на основе имеющихся данных до этого дня для выбора следующего прогноза по наименьшей ошиьбке MAE
    arima_1 = sm.tsa.statespace.SARIMAX(y_train, exog=None, order=(2, 1, 0), seasonal_order=(0, 0, 0, 0)) 
    results_1 = arima_1.fit()
    forecast_1 = np.exp(results_1.forecast(steps=1, exog=None))
    mae_1 = mean_absolute_error(forecast_1, y_test)

    return yhat, mae_1



# Обучение ARIMA модели без преобразования ряда 
def fitting_2 (y_train, y_test):
    history = []
    history = [x for x in y_train]
    
    # Эта часть для прогноза на следующий день, на основе имеющихся данных за сегодня
    obs = y_test[0]
    history.append(obs)
    model = sm.tsa.statespace.SARIMAX(history, exog=None, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0)) 
    results = model.fit()
    forecast = results.forecast(steps=1, exog=None)
    yhat = forecast[0]

    # Эта часть для прогноза на сегодня, на основе имеющихся данных до этого дня для выбора следующего прогноза по наименьшей ошиьбке MAE
    arima_1 = sm.tsa.statespace.SARIMAX(y_train, exog=None, order=(1, 1, 0), seasonal_order=(0, 0, 0, 0)) 
    results_1 = arima_1.fit()
    forecast_1 = results_1.forecast(steps=1, exog=None)
    mae_1 = mean_absolute_error(forecast_1, y_test)

    return yhat , mae_1



# Обратное преобразование Бокса-Кокса
def inverse_boxcox(y, lambda_value):
    if lambda_value == 0:
        return np.exp(y)
    else:
        return (y * lambda_value + 1) ** (1 / lambda_value)



# Обучение модели с преобразованием ряда методом Бокса-Кокса
def fitting_3 (y_train, y_test):
    
    y_train = stats.boxcox(y_train, lmbda=0.6)
    
    history = []
    history = [x for x in y_train]

    # Эта часть для прогноза на следующий день, на основе имеющихся данных за сегодня
    obs = stats.boxcox(y_test[0], lmbda=0.6)
    history.append(obs)
    model = sm.tsa.statespace.SARIMAX(history, exog=None, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0))
    results = model.fit()
    forecast = results.forecast(steps=1, exog=None)
    yhat = inverse_boxcox(forecast[0], 0.6)

    # Эта часть для прогноза на сегодня, на основе имеющихся данных до этого дня для выбора следующего прогноза по наименьшей ошиьбке MAE
    arima_1 = sm.tsa.statespace.SARIMAX(y_train, exog=None, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0)) 
    results_1 = arima_1.fit()
    forecast_1 = inverse_boxcox(results_1.forecast(steps=1, exog=None),  0.6)
    mae_1 = mean_absolute_error(forecast_1, y_test)

    return yhat, mae_1



# Соедиенеие с БД
db = f"{os.getenv('DB_PROTOCOL')}://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"

engine = create_engine(db) 
conn = engine.connect()

# Получение данных с БД для обучения ARIMA моделей
def dataset(ds_el):

    ds_el.to_sql('dataset', engine, if_exists='append', index=False) # для заполнения датасета по всем наблюдениям

    query = "SELECT * FROM dataset"
    all_data = pd.read_sql_query(query, conn)
    all_data.drop_duplicates()

    return all_data


# Подготовка данных, полученных с моделей для отправки на БД
def sql_connect(best_price, output_mlp):

   table_tg = {
    'date': [date.today()],
    'price': [best_price],
    'direction': [output_mlp]}

   table_tg = pd.DataFrame(table_tg)
    
   #Отправка результатов на БД 
   table_tg.to_sql('forecast', engine, if_exists='append', index=False) # для передачи в последующем на прогноз тг-боту

   print('Success')






