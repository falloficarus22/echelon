#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate icarus_env
cd /root/echelon
echo "Wrapper starting at $(date)" >> uci_wrapper.log

# Run the UCI engine with unbuffered output
exec python3 uci.py 2>> uci_wrapper.log