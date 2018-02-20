FROM python:3.6-slim

RUN mkdir /stock-ticker-id
WORKDIR /stock-ticker-id

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8050
CMD python app.py
