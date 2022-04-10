FROM ubuntu:latest
EXPOSE 19000

RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip
RUN pip install allennlp-models==2.2.0

RUN pip install --upgrade click==7.1.1
COPY . /
CMD python3 server.py