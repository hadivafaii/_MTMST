#!/bin/bash

name=${1}
device=${2}

if [ -z "${name}" ]; then
  read -rp "enter fit name: " name
fi
if [ -z "${device}" ]; then
  read -rp "enter device: " device
fi

# Shift to remove the first two positional arguments
# then combine the remaining arguments into one
shift 2
args="${*}"

cd ..

fit="python3 -m analysis.glm ${name} ${device} ${args}"
eval "${fit}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'