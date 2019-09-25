FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda install -y -c pytorch magma-cuda90 && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
#RUN pip install ninja
# This must be done before pip so that requirements.txt is available
WORKDIR /opt

RUN git clone --recursive https://github.com/mysablehats/pytorch.git
RUN cd pytorch && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    pip install -v .

RUN git clone https://github.com/pytorch/vision.git && cd vision && pip install -v .

WORKDIR /workspace
RUN chmod -R a+w /workspace


############# needs sshd and ros with python3 running (copy what I did for fr machine)

##merge later when it works!

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update
RUN apt-get install -y --fix-missing \
  python3-pip \
  python-pip \
  openssh-server\
  libssl-dev \
  #python-sh is needed for the fix.py. once that is solved, remove it.
  python-sh \
  tar\
  libboost-all-dev \
  && apt-get clean && rm -rf /tmp/* /var/tmp/*

# to get ssh working for the ros machine to be functional: (adapted from docker docs running_ssh_service)
RUN mkdir /var/run/sshd \
    && echo 'root:ros_ros' | chpasswd \
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22

#### ROS stuff

ADD requirements_ros.txt /root/
RUN pip3 install --trusted-host pypi.python.org -r /root/requirements_ros.txt && \
    pip2 install --trusted-host pypi.python.org -r /root/requirements_ros.txt && \
    python -m pip install --trusted-host pypi.python.org -r /root/requirements_ros.txt

ADD scripts/ros.sh /root/
### microsoft broke github, so I need this to run wstool. probably need to remove this when it gets fixed!
#ADD scripts/fix.py /root/
RUN /root/ros.sh \
    && echo "source /root/ros_catkin_ws/install_isolated/setup.bash" >> /etc/bash.bashrc

ENV ROS_MASTER_URI=http://SATELLITE-S50-B:11311

#add my snazzy banner
ADD banner.txt /root/

ADD scripts/entrypoint.sh /root/
RUN pip install sklearn
ENTRYPOINT ["/root/entrypoint.sh"]
###needs the catkin stuff as well.
