#!/bin/bash

#echo "TODO: fill in the docker run command"

# Using d: run in background
# Using p: publish a container's port(s) to the host
# Using e: set environment variable (COMET_API_KEY)
# using -i -t: instructs Docker to allocate a pseudo-TTY connected to the container's stdin
# ref: https://docs.docker.com/engine/reference/commandline/run/

docker run -d -p 5000:5000 -e COMET_API_KEY=$COMET_API_KEY -i -t ./ift6758/serving:latest