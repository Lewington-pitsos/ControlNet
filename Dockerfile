FROM python:3.8.16-bullseye
WORKDIR controlnet
ADD ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U xformers
RUN pip install colorgram.py

ADD ./training training
ADD ./models models
ADD . .

# Runpod requires template dockerfiles run forever 
ENTRYPOINT ["tail", "-f", "/dev/null"] 