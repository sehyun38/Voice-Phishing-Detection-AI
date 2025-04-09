import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# import tensorflow as tf
#
# print("TF 버전:", tf.__version__)
# print("GPU 디바이스:", tf.config.list_physical_devices('GPU'))
# print("GPU 사용 가능 여부:", tf.test.is_built_with_cuda())
# print("CUDA 연산 가능 여부:", tf.test.is_gpu_available(cuda_only=True))