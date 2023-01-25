#!/bin/bash

jid_s0=$(sbatch --parsable stage0.sh)
jid_s1=$(sbatch --parsable --dependency=afterok:${jid_s0} stage1.sh)
jid_s2=$(sbatch --parsable --dependency=afterok:${jid_s1} stage2.sh)
jid_s3=$(sbatch --parsable --dependency=afterok:${jid_s2} stage3.sh)
