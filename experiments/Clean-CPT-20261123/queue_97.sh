#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
bash "$root/code/launch_clean.sh" formal Clean_C1_0k_30k
exec bash "$root/code/launch_clean.sh" formal Clean_C1_30k_to_100k
