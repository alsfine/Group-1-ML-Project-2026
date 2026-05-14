#!/bin/bash

mkdir -p ./data/raw
mkdir -p ./data/processed

curl -L -o ./data/oulad.zip\
  https://www.kaggle.com/api/v1/datasets/download/anlgrbz/student-demographics-online-education-dataoulad 

unzip ./data/oulad.zip -d ./data/raw/
