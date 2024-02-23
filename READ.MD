# install conda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n tf_env python=3.8 -y
conda activate tf_env -y
conda install tensorflow-gpu

# tensorflow ดูว่าของตัวเองใช้อันไหนได้
https://www.tensorflow.org/install/source#tested_build_configurations
pip install tensorflow==2.3.*

git https://github.com/TMDDraGon/sign-Language-cnn.git
code Documents/sign-Language-cnn/


conda deactivate
conda remove --name tf_env --all -y

`ref`\
    `Eng`\
    - https://www.mdpi.com/1424-8220/23/18/7970 \
    - https://cs231n.github.io/convolutional-networks/\
    `Eng`\
    - https://www.mdpi.com/1424-8220/23/16/7156 \
    - https://cs231n.github.io/convolutional-networks/ \
    - https://medium.com/@605162020010/19-07-2562-%E0%B8%AB%E0%B8%A5%E0%B8%B1%E0%B8%87%E0%B8%88%E0%B8%B2%E0%B8%81%E0%B8%97%E0%B8%B5%E0%B9%88%E0%B8%9C%E0%B8%A1%E0%B8%AB%E0%B8%B2%E0%B8%A2%E0%B9%84%E0%B8%9B%E0%B8%99%E0%B8%B2%E0%B8%99%E0%B8%95%E0%B8%B4%E0%B8%94%E0%B8%AD%E0%B8%A2%E0%B8%B9%E0%B9%88%E0%B8%AB%E0%B8%A5%E0%B8%B2%E0%B8%A2%E0%B9%80%E0%B8%A3%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%87%E0%B8%97%E0%B8%B1%E0%B9%89%E0%B8%87%E0%B9%80%E0%B8%A3%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%87%E0%B9%80%E0%B8%A3%E0%B8%B5%E0%B8%A2%E0%B8%99-%E0%B8%9B-%E0%B9%82%E0%B8%97-498452e3fe84 \
    - https://blog.pjjop.org/transfer-learning-with-keras-for-computer-vision-applications/