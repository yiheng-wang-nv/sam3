#!/bin/bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bash "${DIR}/task7_no_cover_scripts/run_parallel_segmentation.sh"
bash "${DIR}/task7_scripts/run_parallel_segmentation.sh"
