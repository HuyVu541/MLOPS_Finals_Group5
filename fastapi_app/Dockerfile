# ./fastapi_app/Dockerfile

FROM python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

