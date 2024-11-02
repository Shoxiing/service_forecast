import logging
import pandas as pd
import pytz
import pandas as pd
from datetime import datetime

from binance.spot import Spot
from datetime import datetime, timedelta

from functions import fitting_1, fitting_2, fitting_3, prepare_data, dl_model_predict, sql_connect, dataset
from datetime import datetime, timedelta

from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago



DEFAULT_ARGS = {
    'owner': 'Yusupov_Shohruh',
    'email': 'venumtrafe@yandex.ru',
    'email_on_retry': False,
    'retry': 3,
    'retry_delay': timedelta(minutes=1)
}



dag = DAG('crypto_forecasting',
          schedule_interval="0 0 * * *", # время выбрано в полночь для запуска дага
          start_date=days_ago(2),
          catchup=False,
          tags=["Crypto"]
          )



_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


def init() -> None:
    _LOG.info("Pipeline started.")


def pipeline_models() -> None:


    # Получение данных с криптобиржи и преобразование в dataframe
    utc_timezone = pytz.utc
    current_time_utc = datetime.now(utc_timezone)
        
    previous_day = current_time_utc - timedelta(days=14)
    previous_day_time = previous_day.replace(hour=23, minute=59, second=0, microsecond=0)
        
        
    cl = Spot()
    r = cl.klines('BTCUSDT', '1d', startTime = int(previous_day_time.timestamp()*1000-86400000), endTime = int(datetime.now().timestamp() * 1000))


    # Подготовка данных
    df_zero = pd.DataFrame(r)
    x = prepare_data(df_zero)

    df = x[0]
    ds_el = x[1]
    
    # Функция инференса DL модели и получения прогноза
    output_mlp = dl_model_predict(df)


    # Получаемый из БД весь датасет через функци, отправляя наблюдение для наполнения датасета
    all_data = dataset(ds_el)

    y_train = all_data['open']
    y_test = df['open']

    # Обучение ARIMA моделей на последних данных 
    forecast_1 = fitting_1(y_train, y_test)
    forecast_2 = fitting_2(y_train, y_test)
    forecast_3 = fitting_3(y_train, y_test)

    #выбор наилучшего прогноза из 3х моделей ARIMA путем подсчета MAE по каждой из моделей, и выбора модели с наилучшим качеством относительно цены BTC предыдущего дня
    variables = {'mae1': forecast_1[1], 'mae2': forecast_2[1], 'mae3': forecast_3[1]}
    min_variable_name = min(variables, key=variables.get)

    # Запись в переменную с наименьшей ошибкой по MAE  модели по прогнозу за вчерашний день
    if min_variable_name == 'mae1':
        best_price =  forecast_1[0]
        
    elif min_variable_name == 'mae2':
        best_price =  forecast_2[0]
        
    else:
        best_price =  forecast_3[0]

    # Отправляем в базу данных:прогноз цены на завтра (best_price), направление цены (output_mlp)
    sql_connect(best_price, output_mlp)



def result_of_work() -> None:
    _LOG.info("Success.")


task_init = PythonOperator(task_id = "init", python_callable=init, dag=dag)

task_pipeline_models = PythonOperator(task_id="pipeline", python_callable=pipeline_models, dag=dag)

task_end  = PythonOperator(task_id = "result_of_work", python_callable=result_of_work, dag=dag)

task_init >> task_pipeline_models  >> task_end