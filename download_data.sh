#!/bin/bash

# Define variables
FILE_ID="1i4mJz9xsDwhzWes7sVLXuhLKP9eNtbBG"
FILE_NAME="messytable.zip"
OUTPUT_DIR="MessyTableData"
TEMP_DIR="temp_unzip"

# Function to check and install required packages
install_gdown() {
    if ! command -v gdown &> /dev/null
    then
        echo "gdown not found. Installing..."
        pip install --quiet gdown
    fi
}

# Step 1: Try downloading the file using gdown
install_gdown

echo "Downloading file..."
if ! gdown "https://drive.google.com/uc?export=download&id=$FILE_ID" -O $FILE_NAME; then
    echo "gdown failed. Trying wget method..."
    
    # Attempt to bypass Google's virus scan warning and download using wget
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
        "https://drive.google.com/uc?export=download&id=${FILE_ID}" -O- | \
        sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

    wget --load-cookies /tmp/cookies.txt \
        "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
        -O $FILE_NAME && rm -rf /tmp/cookies.txt
fi

# Verify if the file was downloaded
if [ ! -f "$FILE_NAME" ]; then
    echo "Download failed. Please check the file permissions and try again."
    exit 1
fi

# Step 2: Unzip into a temporary directory
echo "Unzipping..."
mkdir -p $TEMP_DIR
unzip -q $FILE_NAME -d $TEMP_DIR

# Find the extracted folder and rename/move it
EXTRACTED_DIR=$(ls -d $TEMP_DIR/*/ 2>/dev/null | head -n 1)

if [ -d "$EXTRACTED_DIR" ]; then
    mv "$EXTRACTED_DIR" "$OUTPUT_DIR"
else
    mv "$TEMP_DIR"/* "$OUTPUT_DIR"
fi

# Step 3: Create csvs directory inside MessyTableData
echo "Creating csvs directory..."
mkdir -p "$OUTPUT_DIR/csvs"

# Cleanup
rm -rf "$TEMP_DIR"
rm -f "$FILE_NAME"

echo "Done! The data is in '$OUTPUT_DIR'."
