#!/bin/bash

# Install Rust only if not already installed
if ! command -v cargo &> /dev/null; then
  echo "Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
else
  echo "Rust is already installed."
fi

# Set up Rust environment
source $HOME/.cargo/env
rustup default stable

# Optional: print rust version for debug
rustc --version
cargo --version

# Install Python dependencies
pip install -r requirements.txt
