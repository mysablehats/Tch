#!/usr/bin/env bash
#from http://wiki.ros.org/kinetic/Installation/Source with minor adaptations to make it compile

OLDDIR=$PWD
#set -e
{
rosdep init
rosdep update
    } ||
    {
    echo "oops"
    }

#add-apt-repository ppa:ondrej/apache2

addedtorep=`cat /etc/apt/sources.list | grep ondrej`

if [ -z "$addedtorep" ]; then

echo "not there"

echo "deb http://ppa.launchpad.net/ondrej/apache2/ubuntu xenial main " >>  /etc/apt/sources.list

echo "deb-src http://ppa.launchpad.net/ondrej/apache2/ubuntu xenial main " >>  /etc/apt/sources.list

apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 14AA40EC0831756756D7F66C4F4EA0AAE5267A6C

apt-get update

else echo "there"; fi

mkdir -p ~/ros_catkin_ws
cd ~/ros_catkin_ws
#cp /root/fix.py ./

if [ -f "kinetic-ros_comm-wet.rosinstall" ]
then
	echo "installation file already exists. using this one"
else
	rosinstall_generator ros_comm sensor_msgs image_transport common_msgs --rosdistro kinetic --deps --wet-only > kinetic-ros_comm-wet.rosinstall
fi

### this needs to be python2.7 and not conda's 3.6 version
# export OLDPATH=$PATH
# export PATH=/usr/bin:$PATH
#python2.7 fix.py

#wstool init -j`nproc` src kinetic-ros_comm-wet-fixed.rosinstall
wstool init -j`nproc` src kinetic-ros_comm-wet.rosinstall

rosdep install --from-paths src --ignore-src --rosdistro kinetic -y  --os=ubuntu:xenial

#no idea if this will work...
./src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release -DSETUPTOOLS_DEB_LAYOUT=OFF --cmake-args -DPYTHON_VERSION=3.6

cd  /usr/lib/x86_64-linux-gnu/
###oh, i changed from 3.5 to 3.6 so this might break as well...
#ln -s /usr/lib/x86_64-linux-gnu/libboost_python-py35.so libboost_python3.so

# export PATH=$OLDPATH
cd $OLDDIR
