#!/bin/bash

# Exit on error
set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Virtual environment is ready."