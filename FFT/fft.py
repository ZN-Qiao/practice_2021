import torch
import numpy as np

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


class LowPass():
    def __init__(self, image_size, bandwidth, use_cuda):
        
        half_bandwidth = int(bandwidth/2)
        # assert half_bandwidth >= 1
        half_w, half_h = int(image_size/2), int(image_size/2)
        
        mask = np.zeros((image_size, image_size))
        print(half_w-half_bandwidth, half_w+half_bandwidth, half_h-half_bandwidth, half_h+half_bandwidth)
        mask[half_w-half_bandwidth:half_w+half_bandwidth, half_h-half_bandwidth:half_h+half_bandwidth] = 1
        self.mask = torch.from_numpy(mask).float()[None,:,:,None]
        if use_cuda:
            self.mask = self.mask.cuda()
        
    def apply(self, x):
        x = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
        x = batch_fftshift2d(x)
        x = x * self.mask
        
        x = batch_ifftshift2d(x)
        x = torch.irfft(x, signal_ndim=2, normalized=False, onesided=False)
        
        return x


class HighPass():
    def __init__(self, image_size, bandwidth, use_cuda):
        
        half_bandwidth = int(bandwidth/2)
        # assert half_bandwidth >= 1
        half_w, half_h = int(image_size/2), int(image_size/2)
        
        mask = np.ones((image_size, image_size))
        mask[half_w-half_bandwidth:half_w+half_bandwidth, half_h-half_bandwidth:half_h+half_bandwidth] = 0
        self.mask = torch.from_numpy(mask).float()[None,:,:,None]
        if use_cuda:
            self.mask = self.mask.cuda()
        
    def apply(self, x):
        x = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
        x = batch_fftshift2d(x)
        
        x = x * self.mask
        
        x = batch_ifftshift2d(x)
        x = torch.irfft(x, signal_ndim=2, normalized=False, onesided=False)
        
        return x

def demo():
    # print(torch.__version__)
    y = torch.randn(2, 3, 8, 8)
    # y = torch.ones(1, 3, 8, 8)*3

    low_filer = LowPass(8, 3, False)
    low_filer.apply(y)
    print(y)


demo()