# code

config.py 文件内是所有超参数配置。 其中，-pygame 是选择是否使用gui。 在训练过程中不加此参数，在可视化过程中加上此参数，并用env.render() 更新屏幕。

env.py 为针对强化学习封装的环境类

run.py 为训练脚本。

```
强化学习训练使用了 stable-baselines 库。
如果没有该库,先安装

pip install stable-baselines -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com/pypi
```
## 运行方法

```
python run.py  -param
```
