#!/bin/sh
set -e
echo "10.0.0.8  scitos" >> /etc/hosts
echo "10.0.0.239  SATELLITE-S50-B" >> /etc/hosts
echo "172.28.5.2 tsn_denseflow" >> /etc/hosts
echo "history -s /tmp/start.sh " >> /root/.bashrc
echo "history -s wandb login " >> /root/.bashrc
echo "history -s jupyter notebook --port=8888 --no-browser --ip=172.28.5.31 --allow-root " >> /root/.bashrc
echo "history -s git clone https://github.com/amreelab/esa_project.git " >> /root/.bashrc


#bash  /root/face_recognition/catkin_ws.sh
service ssh restart
cat /root/banner.txt
exec "$@"
