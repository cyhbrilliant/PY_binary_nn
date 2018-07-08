import glob
import torch
from os import path as osp
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = [osp.join(abs_path, 'cudalib/mmxnor_cuda_kernel.so')]
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')

ffi = create_extension(
    'mmxnor',
    headers=['src/mmxnor_cuda.h'],
    sources=['src/mmxnor_cuda.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects
)


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
