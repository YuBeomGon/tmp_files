#!/bin/bash

# build the base image
docker build --file dockerfiles/dockerfile.pose.base -t beomgon/deepstream:pose_base .
docker build --file dockerfiles/dockerfile.pose.notebook -t beomgon/deepstream:pose_base.notebook .
docker build --file dockerfiles/dockerfile.deepstream.inference -t beomgon/deepstream:inference .
docker build --file dockerfiles/dockerfile.deepstream.postprocess -t beomgon/deepstream:postprocess .

#docker push beomgon/deepstream:pose_base
#docker push beomgon/deepstream:pose_base.notebook
