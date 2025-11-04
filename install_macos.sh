#!/bin/bash

check_conda() {
    if command -v conda >/dev/null 2>&1; then
        conda --version 2>/dev/null || true
        return 0
    fi

    return 1
}

install_miniconda() {
    arch=$(uname -m)
    case "$arch" in
        arm64) installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh" ;;
        x86_64|i386) installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
        *) installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh" ;;
    esac

    curl -fsSL "$installer_url" -o "$tmp"
    wget "$installer_url" -O ~/miniconda.sh


    # ensure current session can use conda
    export PATH="$HOME/miniconda3/bin:$PATH"
    eval "$(conda shell.bash hook)"
}

ensure_chrome_installed() {
    if ls -f "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" >/dev/null 2>&1; then
        return 0
    fi

    echo "Installing Google Chrome via Homebrew..."
    if brew list --cask google-chrome >/dev/null 2>&1 || brew install --cask google-chrome; then
        return 0
    else
        echo "Failed to install Google Chrome" >&2
        return 1
    fi
}


if check_conda; then
    echo "conda already installed"
else
    install_miniconda
    if check_conda; then
        echo "conda installed successfully"
    else
        echo "conda installation failed" >&2
        exit 1
    fi
fi

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
