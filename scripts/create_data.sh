#!/bin/bash

n_tot=${1-750000}
n_batch=${2-50000}
dim=${3-17}
min_obj_size=${4-3.5}
dtype=${5-"float32"}

cd ..

run_dataset () {
  python3 -m base.dataset \
  "${1}" \
  --n_batch "${2}" --dim "${3}" \
  --min_obj_size "${4}" --dtype "${5}"
}

# run algorithm
run_dataset "${n_tot}" "${n_batch}" "${dim}" "${min_obj_size}" "${dtype}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'