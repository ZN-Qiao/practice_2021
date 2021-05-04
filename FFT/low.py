# import cv2
#
# import numpy as np
#
#
# # Gaussian filter
#
# def gaussian_filter(img, K_size=3, sigma=1.3):
#     if len(img.shape) == 3:
#
#         H, W, C = img.shape
#
#     else:
#
#         img = np.expand_dims(img, axis=-1)
#
#         H, W, C = img.shape
#
#     ## Zero padding
#
#     pad = K_size // 2
#
#     out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
#
#     out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
#
#     ## prepare Kernel
#
#     K = np.zeros((K_size, K_size), dtype=np.float)
#
#     for x in range(-pad, -pad + K_size):
#
#         for y in range(-pad, -pad + K_size):
#             K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
#
#     K /= (2 * np.pi * sigma * sigma)
#
#     K /= K.sum()
#
#     tmp = out.copy()
#
#     # filtering
#
#     for y in range(H):
#
#         for x in range(W):
#
#             for c in range(C):
#                 out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
#
#     out = np.clip(out, 0, 255)
#
#     out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
#
#     return out
#
#
# # Read image
#
# img = cv2.imread("2.jpg")
#
# # Gaussian Filter
#
# out = gaussian_filter(img, K_size=10, sigma=10)
# # Save result
#
# cv2.imwrite("out.jpg", out)
#
# cv2.imshow("result", out)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply gradient filtering
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y,
                          beta=0.5, gamma=0)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
# Plot the images
images = [sobel_x, sobel_y, blended, laplacian]
names = ['sobel_x', 'sobel_y', 'blended', 'laplacian']
plt.figure(figsize = (14, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap = 'gray')
    plt.title(names[i],fontsize=20,color='white')
    plt.axis('off')
plt.show()

