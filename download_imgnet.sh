#!/usr/bin/env bash

# Script: download_imagenet.sh
# Purpose: Download ImageNet 2012 files into a directory named "imgnet"

set -e  # Exit if any command fails

# Create the imgnet directory if it doesn't already exist
mkdir -p imgnet
cd imgnet

# Download the training images tarball
wget --continue https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

# Download the webpage (contains info about training images - not a tar file)
wget --continue "https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php#images:~:text=Training%20images%20(Task%203)"

# Download the validation images tarball
wget --continue https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Download the test images tarball
wget --continue https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar

echo "Download completed. All files are saved in the 'imgnet' folder."
