# Base image
FROM python:3.8.10-buster

# Running every next command wih this user
USER root

# Creating work directory in docker
WORKDIR /usr/app

# Copying files to docker
ADD . '/usr/app'

# Installing Flask App
#RUN pip install flask


RUN apt-get update && \
    apt-get -y install python3-pandas

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Exposing the flask app port from container to host
EXPOSE 5000

# Starting application
CMD ["python", "detect_api.py"]