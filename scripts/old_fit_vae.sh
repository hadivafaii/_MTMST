#!/bin/bash

sim=${1}
device=${2}
# VAE
n_ch=${3:-32}
n_enc_cells=${4:-1}
n_enc_nodes=${5:-3}
n_dec_cells=${6:-1}
n_dec_nodes=${7:-2}
n_pre_cells=${8:-3}
n_pre_blocks=${9:-1}
n_post_cells=${10:-3}
n_post_blocks=${11:-1}
n_latent_scales=${12:-3}
n_latent_per_group=${13:-15}
n_groups_per_scale=${14:-8}
spectral_norm=${15:-0}
ada_groups=${16:-true}
compress=${17:-true}
# Trainer
kl_beta=${18:-0.2}
lr=${19:-0.002}
epochs=${20:-200}
batch_size=${21:-500}
warm_restart=${22:-0}
warmup_portion=${23:-0.02}
optimizer=${24:-"adamax_fast"}
lambda_norm=${25:-0.001}
grad_clip=${26:-500}

if [ -z "${sim}" ]; then
  read -rp "enter simulation category: " sim
fi
if [ -z "${device}" ]; then
  read -rp "enter device: " device
fi


cd ..

fit_vae () {
  if ${16}; then
    if ${17}; then
      python3 -m vae.train_vae "${1}" "${2}" --n_ch "${3}" \
      --n_enc_cells "${4}" --n_enc_nodes "${5}" \
      --n_dec_cells "${6}" --n_dec_nodes "${7}" \
      --n_pre_cells "${8}" --n_pre_blocks "${9}" \
      --n_post_cells "${10}" --n_post_blocks "${11}" \
      --n_latent_scales "${12}" --n_latent_per_group "${13}" --n_groups_per_scale "${14}" \
      --spectral_norm "${15}" --kl_beta "${18}" \
      --lr "${19}" --epochs "${20}" --batch_size "${21}" \
      --warm_restart "${22}" --warmup_portion "${23}" \
      --optimizer "${24}" --lambda_norm "${25}" --grad_clip "${26}" \
      --ada_groups --compress
    else
      python3 -m vae.train_vae "${1}" "${2}" --n_ch "${3}" \
      --n_enc_cells "${4}" --n_enc_nodes "${5}" \
      --n_dec_cells "${6}" --n_dec_nodes "${7}" \
      --n_pre_cells "${8}" --n_pre_blocks "${9}" \
      --n_post_cells "${10}" --n_post_blocks "${11}" \
      --n_latent_scales "${12}" --n_latent_per_group "${13}" --n_groups_per_scale "${14}" \
      --spectral_norm "${15}" --kl_beta "${18}" \
      --lr "${19}" --epochs "${20}" --batch_size "${21}" \
      --warm_restart "${22}" --warmup_portion "${23}" \
      --optimizer "${24}" --lambda_norm "${25}" --grad_clip "${26}" \
      --ada_groups
    fi
  else
    if ${17}; then
      python3 -m vae.train_vae "${1}" "${2}" --n_ch "${3}" \
      --n_enc_cells "${4}" --n_enc_nodes "${5}" \
      --n_dec_cells "${6}" --n_dec_nodes "${7}" \
      --n_pre_cells "${8}" --n_pre_blocks "${9}" \
      --n_post_cells "${10}" --n_post_blocks "${11}" \
      --n_latent_scales "${12}" --n_latent_per_group "${13}" --n_groups_per_scale "${14}" \
      --spectral_norm "${15}" --kl_beta "${18}" \
      --lr "${19}" --epochs "${20}" --batch_size "${21}" \
      --warm_restart "${22}" --warmup_portion "${23}" \
      --optimizer "${24}" --lambda_norm "${25}" --grad_clip "${26}" \
      --compress
    else
            python3 -m vae.train_vae "${1}" "${2}" --n_ch "${3}" \
      --n_enc_cells "${4}" --n_enc_nodes "${5}" \
      --n_dec_cells "${6}" --n_dec_nodes "${7}" \
      --n_pre_cells "${8}" --n_pre_blocks "${9}" \
      --n_post_cells "${10}" --n_post_blocks "${11}" \
      --n_latent_scales "${12}" --n_latent_per_group "${13}" --n_groups_per_scale "${14}" \
      --spectral_norm "${15}" --kl_beta "${18}" \
      --lr "${19}" --epochs "${20}" --batch_size "${21}" \
      --warm_restart "${22}" --warmup_portion "${23}" \
      --optimizer "${24}" --lambda_norm "${25}" --grad_clip "${26}"
    fi
  fi
}

# run algorithm
fit_vae "${sim}" "${device}" "${n_ch}" \
"${n_enc_cells}" "${n_enc_nodes}" "${n_dec_cells}" "${n_dec_nodes}" \
"${n_pre_cells}" "${n_pre_blocks}" "${n_post_cells}" "${n_post_blocks}" \
"${n_latent_scales}" "${n_latent_per_group}" "${n_groups_per_scale}" \
"${spectral_norm}" "${ada_groups}" "${compress}" \
"${kl_beta}" "${lr}" "${epochs}" "${batch_size}" \
"${warm_restart}" "${warmup_portion}" \
"${optimizer}" "${lambda_norm}" "${grad_clip}"

printf '**************************************************************************\n'
printf "Done! (%s)\n" "$(date '+%m/%d/%Y %H:%M:%S')"
printf '**************************************************************************\n\n'