#!/bin/bash

docker run -it --rm -v ./example/demo:/app/vita_example kevinh0718/vita_talou_cco:latest sh -c "cd vita_example && rm -rf CMakeFiles CMakeCache.txt Makefile cmake_install.cmake vessel_synthesis *.cco *.vtp"