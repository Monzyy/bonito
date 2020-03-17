#!/usr/bin/env bash
MINPORTNUM=1024
MAXPORTNUM=65535
# Choose a random port number between $MINPORTNUM-$MAXPORTNUM
# It should be randomized to avoid collision with other users.
# It may be more convenient if you "freeze" the port number to that random number after the first run. For example to 12345
PORT=$(shuf -i $MINPORTNUM-$MAXPORTNUM -n 1)
PORT=39089 # Pick the random port number you get after the first run
NODE=nv-ai-01.srv.aau.dk # If it doesn't work, pick nv-ai-01.srv.aau.dk
FENODE=ai-pilot.srv.aau.dk
SUSER="${1:-magnha14@student.aau.dk}" # Username to slurm front-end and compute node.
DPATH="/user/student.aau.dk/magnha14/data"

echo "#!/bin/bash
singularity exec --nv instance://redebug /usr/bin/python3 \"\$@\" " > spython

# make sure it's executable
chmod 755 spython

echo "Copying spython wrapper to $FENODE."
echo "rsync -av spython $SUSER@$FENODE:~/"

rsync -av spython $SUSER@$FENODE:~/

# remove spython from your laptop
rm spython

# SingularityInstanceCmd="singularity instance start --nv /user/student.aau.dk/magnha14/bonito/bonito.simg redebug"
SingularityInstanceCmd="singularity instance start --nv /user/student.aau.dk/magnha14/bonito/bonito.simg redebug"
# The following wrapper make it possible to run bash with $SingularityInstanceCmd and stay inside the interactive shell
BashWrapper="bash -c '$SingularityInstanceCmd; bash -l'"
COMMAND="srun --pty --nodelist=$NODE --gres=gpu:1 --qos=allgpus $BashWrapper"

echo Openning tunnel with port $PORT at your laptop - localhost, connecting to $NODE via $FENODE.
echo Use port $PORT for setting up remote SSH interpreter in your PyCharm
echo "Keep this terminal window open to maintain the tunnel!"
# exit 0
echo -e "ssh -t -L $PORT:$NODE:22 $SUSER@$FENODE \"$COMMAND\""
ssh -t -L $PORT:$NODE:22 $SUSER@$FENODE "$COMMAND"
