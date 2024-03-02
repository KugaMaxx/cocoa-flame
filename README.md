安装指南


```
conda create -n cocoa python=3.8
conda activate cocoa
```

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install matplotlib tensorboard
pip install external/pointnet2_ops_lib/. （设置gcc版本低于10）
pip install external/dv-toolkit/.
```

