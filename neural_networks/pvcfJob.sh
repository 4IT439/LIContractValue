#!/bin/bash
#PBS -N pvcf
#PBS -l select=1:ncpus=16:mem=32gb:scratch_local=20gb
#PBS -l walltime=1:00:00
#PBS -m ae
# The 4 lines above are options for scheduling system: job will run 1 hour at maximum, 1 machine with 16 processors + 32gb RAM memory + 20gb scratch memory are requested, email notification will be sent when the job aborts (a) or ends (e)

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/brno3-cerit/home/bendima1 # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# loads the application modules
module add conda-modules-py37

# activate prepared conda environment
conda activate /auto/brno2/home/bendima1/myenv

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input files to scratch directory
# if the copy operation fails, issue error message and exit
cp $DATADIR/scripts/pvcf.py $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }
cp $DATADIR/data/df_merged_train_test.pickle $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

# move into scratch directory
cd $SCRATCHDIR

# run the script
# if the calculation ends with an error, issue error message an exit
python pvcf.py -b ./ -f df_merged_train_test.pickle -a [256,256,256,1] --frac 0.2 --train_size 0.75 --val_size 0.25 --activation relu --optimizer Adam -l mse --fit_val_split 0.25 --batch 256 -e 1 -p 5 -o out || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

# move the output to user's DATADIR or exit in case of failure
cp out* $DATADIR/out || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch