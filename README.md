# DCTPN

基于Pytorch的时序动作检测神经网络

# 环境要求

  - pytorch
  - pickle
  - skimage
  - progressbar
  - h5py

# 使用指南
  - 修改video_list_testing.txt,将需要运行的文件名称写入。修改args参数，运行feature_extractor_another_gpu_0将视频提取为图片
  - 修改args参数，运行DCTPN_train进行训练
  - 修改hdf5路径，运行write_json，得到json结果
