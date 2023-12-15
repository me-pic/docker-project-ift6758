#!/bin/bash

#echo "TODO: fill in the docker run command"

# Using d: run in background
# Using p: publish a container's port(s) to the host
# Using e: set environment variable (COMET_API_KEY)
# using -i -t: instructs Docker to allocate a pseudo-TTY connected to the container's stdin
# ref: https://docs.docker.com/engine/reference/commandline/run/

#first port: accept the HTTP requests on that port, second: local port

echo "Running serving container on port ${SERVING_PORT}"

docker run -d -p 5000:5000 -e COMET_API_KEY=$COMET_API_KEY -ti --name group8 ift6758/serving:0.0.1