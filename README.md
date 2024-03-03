安装指南

```
# Recursively initialize our submodule
git submodule update --init --recursive

sudo apt-get install dv-processing
```

```
conda create -n cocoa python=3.8
conda activate cocoa
```

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install scipy matplotlib tensorboard
pip install external/dv-toolkit/.
pip install external/pointnet2_ops_lib/. （设置gcc版本低于10）
```

C++命名规则
私有变量在最后面加下划线，全部采用驼峰命名法（首字母小写，对齐dv-processing），成员变量不加m
https://developer.aliyun.com/article/1269516
