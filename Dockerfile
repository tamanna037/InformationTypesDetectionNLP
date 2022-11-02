# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt


# copy the content of the local source directory to the working directory
COPY code/ ./code/

# copy the content of the local data directory to the working directory
COPY data/ ./data/

# command to run on container start
#CMD [ "CD", "./source" ]
#CMD [ "python", "./process_post_text.py" ]