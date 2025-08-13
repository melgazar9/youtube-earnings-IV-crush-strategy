#!/usr/bin/env bash
set -e

echo "Running pre-commit hooks..."
pre-commit run --all-files --show-diff-on-failure
