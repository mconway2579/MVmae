#!/bin/bash

# Define variables
FILE_ID="1i4mJz9xsDwhzWes7sVLXuhLKP9eNtbBG"
FILE_NAME="messytable.zip"
OUTPUT_DIR="MessyTableData"

# Step 1: Download the file using gdown (install if not available)
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing..."
    pip install gdown
fi

echo "Downloading file..."
gdown --id $FILE_ID -O $FILE_NAME

# Step 2: Unzip into MessyTableData
echo "Unzipping..."
mkdir -p $OUTPUT_DIR
unzip -q $FILE_NAME -d $OUTPUT_DIR

# Step 3: Create csvs directory inside MessyTableData
echo "Creating csvs directory..."
mkdir -p "$OUTPUT_DIR/csvs"

echo "Done!"