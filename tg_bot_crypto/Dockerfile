FROM python:3.11

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY tg_app.py tg_app.py

CMD ["python", "tg_app.py"]
