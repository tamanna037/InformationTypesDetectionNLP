#  Copyright 2022 Tamanna, Licensed under MIT. For more information , check LICENSE.txt

# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /OSSProject

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt


# copy the content of the local source directory to the working directory
COPY code/ ./code/

# copy the content of the local data directory to the working directory
COPY data/ ./data/

# Use these two commands to run-> first change directory , next run the code.
# 1. cd ./code
# 2. python InfoTypesDetectionOss.py
