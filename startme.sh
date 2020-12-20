#!/usr/bin/env bash

# nvidia-docker became docker --gpus all now and probably the NV_GPU flag doesn't work anymore. But maybe someone might want to still use nvidia-docker2 package, so this script needs to become slightly more generic.

#Tthis was horrible and it took me long to fix
#PASSWD=$1
echo "Password for $USER"
read -s PASSWD
MYUSERNAME=frederico
DOCKERHOSTNAME=poop
THISVOLUMENAME=sshvolume-workspace-torch_new
DOCKERMACHINEIP=172.28.5.31
DOCKERMACHINENAME=tch_new
MACHINEHOSTNAME=torch_machine3
THISWSPATH=/workspace
#DOCKERFILE=docker/pytorch/ ## standard should be .
#BUILDINDIR=$PWD/pytorch ##standard should be $PWD
DOCKERFILE=.
BUILDINDIR=$PWD
#export NV_GPU=1
if [ -z "$PASSWD" ]
then
  echo "you need to input your own password to mount the internal ssh volume that is shared between docker and the docker host!"
  echo "usage is: $0 <your-password-here>"
else
  while true; do
    {
    #echo "doing nothing"
    OLDDIR=$PWD
    cd $BUILDINDIR
    docker build -t $DOCKERMACHINENAME $DOCKERFILE
    #nvidia-docker build --no-cache -t $DOCKERMACHINENAME .
    cd $OLDDIR
    } ||
    {
    echo "something went wrong..." &&
    break
    }
  echo "STARTING ROS PYTORCH ROS DOCKER..."

  ISTHERENET=`docker network ls | grep br0`
  if [ -z "$ISTHERENET" ]
  then
    echo "docker network br0 not up. creating one..."
    docker network create \
      --driver=bridge \
      --subnet=172.28.0.0/16 \
      --ip-range=172.28.5.0/24 \
      --gateway=172.28.5.254 \
      br0
  else
    echo "found br0 docker network."
  fi
  /usr/bin/expect <<EOD
spawn scripts/enable_forwarding_docker_host.sh
expect "hello"
send -- "something"
expect "*?assword*"
send -- "$PASSWD\n"
interact
EOD
  #nvidia-docker run --rm -it -p 8888:8888 -h $MACHINEHOSTNAME --network=br0 --ip=$DOCKERMACHINEIP $DOCKERMACHINENAME #bash
  {
  docker volume create --driver vieux/sshfs   -o sshcmd=$MYUSERNAME@$DOCKERHOSTNAME:$PWD/workspace -o password=$PASSWD $THISVOLUMENAME
  } ||
  {
    echo "could not mount ssh volume. perhaps vieux is not installed?" &&
    echo "install with: docker plugin install vieux/sshfs" &&
    break
  }
#  nvidia-docker run --rm -it -u root -p 8888:8888 -p 222:22 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $THISVOLUMENAME:/catkin_ws -h $MACHINEHOSTNAME --network=br0 --ip=$DOCKERMACHINEIP $DOCKERMACHINENAME bash # -c "jupyter notebook --port=8888 --no-browser --ip=$DOCKERMACHINEIP --allow-root &" && bash -i
   # the k40 is useless with cuda 10.1
   docker run --gpus '"device=0"' --rm -it -u root -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $THISVOLUMENAME:$THISWSPATH -v /mnt/share:/mnt/share -h $MACHINEHOSTNAME --network=br0 --ip=$DOCKERMACHINEIP $DOCKERMACHINENAME bash # -c "jupyter notebook --port=8888 --no-browser --ip=172.28.5.4 --allow-root &" && bash -i
   #nvidia-docker run --rm -it -u root -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $THISVOLUMENAME:$THISWSPATH -v /mnt/share:/mnt/share -h $MACHINEHOSTNAME --network=br0 --ip=$DOCKERMACHINEIP $DOCKERMACHINENAME bash # -c "jupyter notebook --port=8888 --no-browser --ip=172.28.5.4 --allow-root &" && bash -i


  ## if I add this with -v I can't catkin_make it with entrypoint...
  #-v /temporal-segment-networks/catkin_ws:$PWD/catkin_ws/src
  #
  docker volume rm $THISVOLUMENAME
  break
  done
fi
