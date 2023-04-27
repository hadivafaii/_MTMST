#!/bin/bash

model_name=${1}
fit_name=${2}
device=${3}

if [ -z "${model_name}" ]; then
  read -rp "enter model name: " model_name
fi
if [ -z "${fit_name}" ]; then
  read -rp "enter fit name: " fit_name
fi
if [ -z "${device}" ]; then
  read -rp "enter device: " device
fi

shift 3
args="${*}"

cd ..

fit="python3 -m analysis.glm ${model_name} ${fit_name} ${device} ${args}"
fit=${fit//(/\\(}
fit=${fit//)/\\)}
eval "${fit}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'