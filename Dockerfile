FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN chmod -R 777 .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "--app", "main", "run", "--host=0.0.0.0", "--port=5000"]