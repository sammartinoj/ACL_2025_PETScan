#!/bin/bash

# install dependencies
python3 -m pip install --upgrade pip

# comment this line out if already run experiment previously 
python3 -m pip install --no-cache-dir -r requirements.txt

python3 src/launch.py 