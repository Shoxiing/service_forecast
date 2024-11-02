import pandas as pd
from sqlalchemy import *
from sqlalchemy.dialects import registry
import psycopg2
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from aiogram.types import BufferedInputFile
import io
import yfinance as yf

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Соедиенеие с БД
conn = psycopg2.connect(host=os.getenv("POSTGRES_HOST"),
dbname=os.getenv("POSTGRES_DB"),
user=os.getenv("POSTGRES_USER"),
password=os.getenv("POSTGRES_PASSWORD"),
target_session_attrs='read-write')


# Функция доставки данных из БД
async def forecast():
    try:
        query = "SELECT * FROM forecast ORDER by date DESC limit 1"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return None, None
        
        price = df['price'][0]
        dir = df['direction'][0]
        return price, dir
    except KeyError:
        return None, None
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None, None
    


# Функция выгрузки данных в нужном формате с yfinance
async def data_all(start_date, end_date, tickers):

    df = pd.DataFrame(yf.download(tickers[0], start=start_date, end=end_date))
    return df



# Визуализация исторических данных.
async def plot_history(stock_history, tickers):
    """
    :param stock_history: Данные из функции data_loader`;
    :param tickers: Криптовалюта
    :return: Изображение BufferedInputFile сохраненное в буфере
    """
    # Create the graph
    plt.figure(figsize=(12, 5))
    plt.plot(stock_history[tickers])
    plt.title(f"{tickers[0]} Исторические данные стоимости")
    plt.xlabel("Дата")
    plt.ylabel("Стоимость")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    file = BufferedInputFile(buf.read(), filename="stock_price_history.png")
    return file



# Загрузка данных с yfinance.
async def data_loader(start_date, end_date, tickers, col_value):
    """
    :param start_date: Время начала рассмотрения данных;
    :param end_date: Время окончания рассмотрения данных;
    :param tickers: Криптовалюта для загрузки;
    :param col_value: Колонка для парсинга с yfinance;
    :return: Таблица загруженных данных со столбцом 'col_value'.
    """

    data = pd.DataFrame(columns=tickers)
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start_date, end_date)[col_value]
    return data



#Функция группировки данных по дате и средней стоимости
async def data_grp(data, tickers):

    data = data.reset_index()
    df_grp = (
        data.groupby(data["Date"].dt.strftime("%B %Y"))[tickers[0]].mean().reset_index()
    )
    df_grp["Date"] = pd.to_datetime(
        df_grp["Date"], format="%B %Y"
    )  # Преобразование столбца Date в формат даты
    df_grp = df_grp.sort_values(by="Date")  # Сортировка по столбцу Date
    return df_grp



# Функция визуализации среднемесячной стоимости
async def viz_avg(df_grp, tickers):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_grp["Date"],
            y=df_grp[tickers[0]],
            mode="lines+markers",
            name=tickers[0],
        )
    )

    fig.update_layout(
        title="Динамика среднемесячной стоимости по периоду " + tickers[0],
        xaxis_title="Месяц - год",
        yaxis_title="Стоимость , USD",
        xaxis_tickangle=-45,
    )

    img_bytes = fig.to_image(format="png")

    file = BufferedInputFile(img_bytes, filename="avg_price.png")

    return file
