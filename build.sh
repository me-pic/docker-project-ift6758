#!/bin/bash

#echo "TODO: fill in the docker build command"

# -f: specify Dockerfile

docker build -t ift6758/serving:1.0.0 . -f Dockerfile.serving