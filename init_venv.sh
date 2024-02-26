#!/bin/bash

python3.11 -m venv venv
source venv/bin/activate
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt --upgrade