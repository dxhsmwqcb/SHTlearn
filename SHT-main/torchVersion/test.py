"""测试pytprch版本"""
import torch
print(torch.__version__)  # 应该输出 1.11.0
print(torch.cuda.is_available())  # 应该输出 True
print(torch.version.cuda)  # 应该输出 11.4
