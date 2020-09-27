# set base image
FROM python:3.8

# copy FeatherMap package into container
COPY setup.py /tmp/
COPY feathermap /tmp/feathermap

# install feathermap package
WORKDIR /tmp
RUN pip3 install -e .

# command to run on container start
ENTRYPOINT ["python", "feathermap/ffnn_main.py"]
