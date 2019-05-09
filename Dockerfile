FROM ubuntu:18.04

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get -y install build-essential python3-dev python3-setuptools \
                     python3-numpy python3-scipy python3-pip \
                     libatlas3-base
RUN apt-get -y install python-matplotlib

RUN pip3 install -U scikit-learn seaborn pandas

WORKDIR /files
ADD . /files

VOLUME /home/df/models
