FROM python:3
RUN mkdir /app
ADD . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
