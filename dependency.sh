#!/bin/bash

jid_pre=$(sbatch --parsable run.sh)
jid_w01=$(sbatch --parsable --dependency=afterok:${jid_pre} post.sh)