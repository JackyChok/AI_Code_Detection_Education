FROM python:3.7.3-slim-stretch

WORKDIR /
RUN apt-get update
RUN apt-get install -y python-dev
RUN apt-get install -y build-essential
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

COPY detector-large.pt /detector-large.pt
# TODO: keep root/.cache/torch/transformers/ between runs
COPY requirements.txt requirements.txt
RUN pip3 --no-cache-dir install -r requirements.txt
COPY detector/ /detector

ENTRYPOINT ["python3", "-m", "detector.server", "detector-large.pt"]
