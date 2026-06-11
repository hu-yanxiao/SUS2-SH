#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash dev_test/sh_two_layer_residual_init_check.sh
