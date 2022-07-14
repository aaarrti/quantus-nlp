#!/bin/zsh
gcloud compute tpus execution-groups create \
  --name="$1" \
  --zone=europe-west4-a \
  --tf-version=2.9.1 \
  --accelerator-type=v2-8 \
  --preemptible