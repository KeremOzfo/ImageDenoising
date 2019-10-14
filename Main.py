import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#img = Image.open(path)
# On successful execution of this statement,
# an object of Image type is returned and stored in img variable)

img = Image.open("testingpic.jpg")
print(img.size)
#print (len(img.histogram()))  #yukarıdan aşağıya
print (img.mode)
#print (img.tobitmap())
pix_val = list(img.getdata())

print(pix_val) #visits each pixel

iar = np.asarray(img)  #Same values, process the image differently sağa doğru
print(iar)

plt.imshow(iar)
plt.show()