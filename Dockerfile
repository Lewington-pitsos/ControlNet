FROM python:3.8.16-bullseye
WORKDIR controlnet
ADD ./requirements.txt .
RUN pip install -r requirements.txt

ADD . .

