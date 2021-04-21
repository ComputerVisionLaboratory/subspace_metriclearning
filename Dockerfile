FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/*
RUN apt update && apt upgrade -y
RUN apt -y install curl wget vim

##
RUN apt -y install python3.9
RUN echo "alias python=python3.9" >> ~/.bashrc
RUN echo "alias python3=python3.9" >> ~/.bashrc
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
RUN update-alternatives --config python3
RUN source ~/.bashrc
RUN apt -y install python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py


##
RUN apt -y install libopencv-dev
RUN apt -y install git

##
RUN pip3 install -U pip setuptools
RUN pip3 install matplotlib ipython seaborn jupyter scipy
RUN pip3 install opencv-python Pillow scikit-learn scikit-image
RUN pip3 install tqdm pandas black numpy flake8 prospector mypy bandit vulture
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pytorch-lightning hydra-core


##
# RUN pip3 install pymanopt==0.2.4
RUN git clone https://github.com/pymanopt/pymanopt.git \
    && cd pymanopt \
    && python setup.py install \
    && cd ..  \
    && cp -r pymanopt/pymanopt.egg-info /usr/local/lib/python3.9/dist-packages/  \
    && cp -r pymanopt /usr/local/lib/python3.9/dist-packages/  \
    && rm -r pymanopt



ENV GEOMSTATS_BACKEND=pytorch
# commit id a5eee33
RUN git clone https://github.com/geomstats/geomstats.git \
    && cd geomstats && pip3 install -r requirements.txt \
    && python3.9 setup.py install \
    && cd .. \
    && cp -r geomstats/geomstats.egg-info /usr/local/lib/python3.9/dist-packages/ \
    && cp -r geomstats /usr/local/lib/python3.9/dist-packages/ \
    && rm -r geomstats

WORKDIR /root/workspace
