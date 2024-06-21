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

RUN pip install --trusted-host pypi.python.org -r requirements2.txt

# Exposing the flask app port from container to host
EXPOSE 5001
EXPOSE 5567

# Starting application
CMD ["python", "server.py"]