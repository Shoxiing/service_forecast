import asyncio
import logging
import os
from aiogram.fsm.context import FSMContext
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.filters.state import State, StatesGroup
import re
from dotenv import load_dotenv
from aiogram import types
from functions_tg import forecast, data_loader, plot_history, data_grp, viz_avg

load_dotenv()


# Включаем логирование
logging.basicConfig(level=logging.INFO)


# Создание экземпляра бота
BOT_TOKEN = os.getenv("TELEGRAM_BOT_API_KEY")

# Объект бота
bot = Bot(token=BOT_TOKEN)

# Диспетчер
dp = Dispatcher()





class Form(StatesGroup):
    coin = State()
    time_range = State()
    time_ranger = State()
    time_rangers = State()
    horizon_predict = State()
    review = State()
    expect = State()



# Кнопки ввода
@dp.message(Command("start"))
async def welcome(message: types.Message) -> None:
    kb = [[types.KeyboardButton(text="Прогноз цены BTC на завтра")],
          [types.KeyboardButton(text="Динамика стоимости")],
        [types.KeyboardButton(text="Динамика среднемесячной стоимости")]]   
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kb, resize_keyboard=True, input_field_placeholder="Не является финансовым советом"
    )
    await message.answer("Нажмите на кнопку:", reply_markup=keyboard)



# Формирование ответа
@dp.message(lambda message: message.text == "Прогноз цены BTC на завтра")
async def btc_forecast_tomorrow(message: types.Message):
    price, direction = await forecast()
    if price is None:
        await message.answer("Данные не найдены.")
        return

    direction_text = "повышение" if direction == 1 else "понижение"
    forecast_send = f"Завтра ожидается {direction_text} цены.\nОжидаемая цена в USD: {round(price, 2)}"
    
    await message.answer(forecast_send)


# Колонка для парсинга с yfinance
COL_VALUE = "Adj Close"

# Тикеры криптовалют
TICKERS = []




@dp.message(F.text.lower() == "динамика стоимости")
async def get_name_ticker(message: types.Message) -> None:
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text="BTC-USD", callback_data="BTC-USD"))

    builder.add(types.InlineKeyboardButton(text="ETH-USD", callback_data="ETH-USD"))
    await message.answer("Выберите монету:", reply_markup=builder.as_markup())


@dp.callback_query(lambda query: query.data in ["BTC-USD", "ETH-USD"])
async def get_find_ticker(callback: types.CallbackQuery, state: FSMContext):
    selected_coin = callback.data
    TICKERS.append(selected_coin)

    await bot.answer_callback_query(callback.id)
    await state.update_data(coin=selected_coin)

    if selected_coin == "BTC-USD":
        await bot.send_message(callback.from_user.id, "Вы выбрали BTC-USD.")

    elif selected_coin == "ETH-USD":
        await bot.send_message(callback.from_user.id, "Вы выбрали ETH-USD.")

    await state.set_state(Form.time_range)
    await bot.send_message(
        callback.from_user.id,
        "Введите временной интервал для построения графика в формате: "
        "YYYY-MM-DD YYYY-MM-DD (например, 2023-01-01 2023-12-31):",
    )


# История цен на криптовалюты

@dp.message(Form.time_range)
async def send_stock_history(message: types.Message, state: FSMContext):
    """
    Отправляет историю цен на криптовалюты
     в указанном временном диапазоне в виде графика.

    :param message: Объект сообщения,
     содержащий время начала и конца временного диапазона.
    :param state: Состояние конечного автомата для управления состоянием бота.
    :return: Отравка сообщения в виде графика

    Исключения:
    Если произошла ошибка при загрузке данных или построении графика,
    будет отправлено сообщение с описанием ошибки пользователю.
    """
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2}$")

    time_range = await state.update_data(time_range=message.text)
    if not pattern.match(time_range["time_range"]):
        await message.reply(
            "Неверный формат временного интервала. Попробуйте все заново."
        )
        return

    start_date, end_date = time_range["time_range"].split()

    await state.clear()
    try:
        data = await data_loader(start_date, end_date, TICKERS, COL_VALUE)
        image = await plot_history(data, TICKERS)
        TICKERS.pop()
        await bot.send_photo(message.chat.id, photo=image)
    except Exception as e:
        await message.reply(f"An error occurred: {e}")

    await state.set_state(Form.expect)



# Формирует среднемесячную стоимость крипт. в заданном временном интервале

@dp.message(F.text.lower() == "динамика среднемесячной стоимости")
async def get_name_ticker_01(message: types.Message) -> None:
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(text="BTC-USD avg", callback_data="BTC-USD avg")
    )
    builder.add(
        types.InlineKeyboardButton(text="ETH-USD avg", callback_data="ETH-USD avg")
    )

    await message.answer("Выберите монету:", reply_markup=builder.as_markup())


@dp.callback_query(lambda query: query.data in ["BTC-USD avg", "ETH-USD avg"])
async def get_find_ticker_01(callback: types.CallbackQuery, state: FSMContext):
    selected_coin = callback.data
    TICKERS.append(selected_coin[:-4])

    await bot.answer_callback_query(callback.id)
    await state.update_data(coin=selected_coin)

    if selected_coin == "BTC-USD avg":
        await bot.send_message(callback.from_user.id, "Вы выбрали BTC-USD.")

    elif selected_coin == "ETH-USD avg":
        await bot.send_message(callback.from_user.id, "Вы выбрали ETH-USD.")

    await state.set_state(Form.time_ranger)
    await bot.send_message(
        callback.from_user.id,
        "Введите временной интервал для построения графика средней стоимости в формате: "
        "YYYY-MM-DD YYYY-MM-DD (например, 2023-01-01 2023-12-31):",
    )


@dp.message(Form.time_ranger)
async def send_crypto_avg(message: types.Message, state: FSMContext):
    """
    Отправляет историю цен  средней стоимости криптовалют
     в указанном временном диапазоне в виде графика

    """
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2}$")

    time_ranger = await state.update_data(time_ranger=message.text)
    if not pattern.match(time_ranger["time_ranger"]):
        await message.reply(
            "Неверный формат временного интервала." " Попробуй все заново."
        )
        return

    start_date, end_date = time_ranger["time_ranger"].split()

    await state.clear()
    try:

        data = await data_loader(start_date, end_date, TICKERS, COL_VALUE)
        data_gr = await data_grp(data, TICKERS)
        image = await viz_avg(data_gr, TICKERS)
        TICKERS.pop()
        await bot.send_photo(message.chat.id, photo=image)
    except Exception as e:
        await message.reply(f"An error occurred: {e}")

    await state.set_state(Form.expect)




async def main():
    try:
        await dp.start_polling(bot, skip_updates=True)
    finally:
        await dp.storage.close()


if __name__ == "__main__":
    asyncio.run(main())
