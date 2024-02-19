#!/bin/bash

echo "sourcing aider virtual pyenv before running aider ...\n"
source aenv/bin/activate
aider --no-auto-commits --no-dirty-commits --model gpt-4-0125-preview