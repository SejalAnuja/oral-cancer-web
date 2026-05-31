FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p static/uploads static/heatmaps reports

EXPOSE 7860

CMD ["python", "app.py"]