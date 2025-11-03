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

conda create -p ./venv python=3.10 -y
conda activate ./venv
cd ravas
pip install .
python -m run_gui