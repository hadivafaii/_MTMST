#!/bin/bash

# Define function to set argument with default value
set_arg_with_default() {
  local arg_name="$1"
  local default_val="$2"

  # Check if value is provided, otherwise use default value
  if [[ -n "${!arg_name}" ]]; then
    declare -g "$arg_name=${!arg_name}"
  else
    declare -g "$arg_name=$default_val"
  fi
}

# Define default values for arguments
arg1_default="default1"
arg2_default="default2"
# ... Define default values for other arguments here

# Parse command line arguments using getopts
while getopts ":a:b:" opt; do
  case $opt in
    a)
      arg1_val="$OPTARG"
      ;;
    b)
      arg2_val="$OPTARG"
      ;;
    # ... Add cases for other arguments here
    *)
      echo "Unknown option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Call function to set argument values
set_arg_with_default "arg1_val" "$arg1_default"
set_arg_with_default "arg2_val" "$arg2_default"
# ... Call the function for other arguments here

# Use the argument values in your script
echo "arg1_val: $arg1_val"
echo "arg2_val: $arg2_val"
# ... Use other argument values in your script here



# cd ..


printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'