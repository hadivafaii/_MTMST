#!/bin/bash

category=${1}
n_obj=${2}
device=${3}
n_ch=${4:-32}
# mode=${3:-"bold"}
task=${4:-"all"}
metric=${5:-"pearson"}
key=${6:-"all"}
full=${7-false}

if [ -z "${category}" ]; then
  read -rp "enter stimulus category: " category
fi
if [ -z "${n_obj}" ]; then
  read -rp "enter number of objects: " n_obj
fi
if [ -z "${device}" ]; then
  read -rp "enter device: " device
fi


cd ..

run_net () {
  if ${7}; then
    python3 -m vae.train_vae \
    "${1}" "${2}" \
    --mode "${3}" \
    --task "${4}" \
    --metric "${5}" \
    --key "${6}" \
    --full
  else
    python3 -m vae.train_vae \
    "${1}" "${2}" \
    --mode "${3}" \
    --task "${4}" \
    --metric "${5}" \
    --key "${6}"
  fi
}

# run algorithm
run_net "${nn}" "${ll}" "${mode}" "${task}" "${metric}" "${key}" "${full}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'