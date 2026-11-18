#!/bin/bash

# Add all files
git add .

# Commit with the message passed as the first argument (empty if none given)
git commit --allow-empty-message -m "${1:-}"

# Push to the current branch
git push