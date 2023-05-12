#!/bin/bash

sim=${1}
device=${2}

if [ -z "${sim}" ]; then
  read -rp "enter simulation category: " sim
fi
if [ -z "${device}" ]; then
  read -rp "enter device: " device
fi

# Shift to remove the first two positional arguments
# then combine the remaining arguments into one
shift 2
args="${*}"

cd ..

fit="python3 -m ae.train_ae ${sim} ${device} ${args}"
eval "${fit}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'