#!/bin/bash

tres=${1-25}

cd ..

run_process () {
  python3 -m utils.process --tres "${1}"
}

# run algorithm
run_process "${tres}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'