FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         gnupg2 \
         libjpeg-dev \
         lsb-core \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN chmod -R a+w /workspace


############# needs sshd and ros with python3 running (copy what I did for fr machine)

#### ROS stuff

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

##after adding the key we need to update it again!

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
     python3-pip \
     python-pip \
     openssh-server\
     libssl-dev \
     lsb-core \
     python-sh \
     tar\
     libboost-all-dev \
     ros-melodic-ros-base \
     python-rosdep \
     python-rosinstall \
     python-rosinstall-generator \
     python-wstool \
     && apt-get clean && rm -rf /tmp/* /var/tmp/*

     # some more ros stuff
RUN rosdep init && rosdep update

# to get ssh working for the ros machine to be functional: (adapted from docker docs running_ssh_service)
RUN mkdir /var/run/sshd \
     && echo 'root:ros_ros' | chpasswd \
     && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
     && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
ADD requirements_tch.txt /root/

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

ENV ROS_MASTER_URI=http://SATELLITE-S50-B:11311

#add my snazzy banner
ADD banner.txt /root/
### try to run jupyter so we can do some coding...
RUN pip install jupyter && pip install -r /root/requirements_tch.txt
#&& jupyter tensorboard enable --system
##jupyter notebook --port=8888 --no-browser --ip=172.28.5.31 --allow-root

ADD scripts/entrypoint.sh /root/

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN git clone https://github.com/huggingface/transformers && python transformers/utils/download_glue_data.py
ENTRYPOINT ["/root/entrypoint.sh"]
     ###needs the catkin stuff as well.
