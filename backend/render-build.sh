#!/bin/bash

# Exit immediately on error
set -e

echo " Installing Rust"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

echo " Loading Rust environment"
source $HOME/.cargo/env

echo " Setting Rust to stable"
rustup default stable

echo " Installing Python dependencies"
pip install -r requirements.txt

echo "Build script completed successfully"
