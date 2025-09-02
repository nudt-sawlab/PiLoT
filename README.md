准备：3dtiles模型，query序列，gt pose.txt， checkpoint
如果是猎影系统环境，额外按照pykalman, segmentation-models-pytorch(0.3.4版本)    
1. 安装3DTilesRender库，将编译好的.so文件放到pixloc/utils/osg下
2. 编译DirectAbsoluteCostCuda库：
    a) rm -rf build/ direct_abs_cost_cuda.egg-info/
    b) python setup.py build_ext --inplace
    c) pip install .
3. (Optional for Google map) 挂载模型sudo ln -s .../v1 /v1
4. 修改config/google.yaml中文件路径，包括dataset_path和checkpoint
5. 执行./run.sh

最后：cuda结束有可能报warning，残留multiprocess进程，需要手动kill掉