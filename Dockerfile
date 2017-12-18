FROM ubuntu:zesty

CMD bash

RUN apt-get update
RUN apt-get install -y python3-pip curl
RUN pip3 install tensorflow
RUN pip3 install sklearn
RUN pip3 install pytest
RUN pip3 install scipy
