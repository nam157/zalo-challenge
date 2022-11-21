FROM python:3

WORKDIR /f/zalo-challenge/

COPY . .

RUN pip install -r requirements.txt

CMD ["python","test.py"]