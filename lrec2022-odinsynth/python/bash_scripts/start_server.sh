#!/bin/bash

python_dir="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

uvicorn api:app --workers 1 --timeout-keep-alive 900 --port $1 --app-dir $python_dir
