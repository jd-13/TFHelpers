FROM ubuntu:artful

CMD bash

RUN apt-get update
RUN apt-get install -y python3-pip curl
RUN pip3 install wheel
RUN pip3 install tensorflow
RUN pip3 install sklearn
RUN pip3 install pytest pytest-cov
RUN pip3 install scipy
