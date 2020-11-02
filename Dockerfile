# set base image
FROM python:3.8

# copy FeatherMap package into container
COPY setup.py .
COPY feathermap ./feathermap

# install feathermap package
RUN pip3 install -e .

# command to run on container start
ENTRYPOINT ["python", "feathermap/train.py"]
