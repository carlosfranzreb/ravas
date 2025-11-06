#!/bin/bash

# create conda environment and install dependencies
conda create -p ./venv python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate ./venv
cd ravas
pip install .


# prepare chrome extension for the avatar renderer
cd ../rpm
ensure_chrome_installed || { echo "Could not ensure Google Chrome is installed" >&2; exit 1; }
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --pack-extension=resources/dist/chrome-extension --pack-extension-key=resources/chrome-extension-packing/privkey.pem 
mkdir dist
mv resources/chrome-extension.crx dist/chrome-extension.crx

# compile mimi models
cd ../ravas
mkdir onnx
python -m stream_processing.models.mimivc.compile

# download Mimi target features
mkdir target_feats
cd target_feats
mkdir mimivc
cd mimivc

for file in jessica jeffrey nadine norbert; do
  curl -LO "https://github.com/carlosfranzreb/ravas/releases/download/v0.7/${file}.pt"
done

cd ../..

# run GUI
python -m run_gui
