import imageio as img 
import numpy as np 
import matplotlib.pyplot as plt

def localThreshold(image,block_size,c):
    imgPad = np.pad(image,pad_width=1,mode='constant', constant_values=0)
    threshold = np.zeros_like(image)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            local_region = imgPad[i: i+block_size, j: j+block_size]
            local_mean = np.mean(local_region)
            threshold[i,j] = 255 if image[i,j] > (local_mean - c) else 0 

    return threshold        
image =  img.imread("C://Users//komputer 34//Downloads//images (2).jpg", mode='F')

result = localThreshold(image,15,10)


plt.figure(figsize = (15,5))

plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.imshow(result, cmap='gray')
plt.show()
