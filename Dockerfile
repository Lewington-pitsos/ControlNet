FROM python:3.8.16-bullseye
WORKDIR controlnet
ADD ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U xformers

ADD ./training training
ADD ./models models
ADD . .

ENTRYPOINT ["tail", "-f", "/dev/null"] 