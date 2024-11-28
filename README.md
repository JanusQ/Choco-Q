# Choco

## 配置环境（二选一）

环境配置过程中会出现中间文件，不要动，结束后会自动删除，
如果网络不好，GPU版本安装时间较长约，请耐性等待

提示找不到包可尝试：
禁用用户指定包：export PYTHONNOUSERSITE=1

如果安装中断或失败，需要删除环境后重新安装
conda remove -n choco_xxx --all

### 仅CPU版本

1. conda env create -f environment_cpu.yml
2. conda activate choco_cpu
3. pip install .

### 支持GPU版本（需要GPU支持CUDA12）

1. conda env create -f environment_gpu.yml
2. conda activate choco_gpu
3. pip install .


## 测试一

根据按装环境测试运行对应测试文件:
.testbed_cpu.py
.testbed_gpu.py

看到 "Environment configuration is successful!" 即表示配置成功

配置失败考虑：
1. 终端是否正确activate对应conda环境
2. 是否在切换后执行 pip install . 安装chocoq库
3. python执行环境是否选择对应conda环境


## 复现
自定义目标函数优化问题  
./implementions/0_test.ipynb

论文中 Table  
./implementions/1_table.ipynb