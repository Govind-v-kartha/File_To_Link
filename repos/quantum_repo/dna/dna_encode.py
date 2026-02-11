import numpy as np 

def dna_encode(permuted_image, ksk):
    height, width = permuted_image.shape
    DNi0 = np.zeros((height, width), dtype=np.uint8)
    DNi1 = np.zeros((height, width), dtype=np.uint8)
    DNi2 = np.zeros((height, width), dtype=np.uint8)
    DNi3 = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            binary_pixel = bin(permuted_image[i, j])[2:].zfill(8)
            for k in range(4):
                dna_base_index = int(binary_pixel[k*2:(k+1)*2], 2)
                dna_base_index = (dna_base_index + int(ksk[k % len(ksk)])) % 4

                if k == 0: DNi0[i,j] = dna_base_index
                elif k == 1: DNi1[i,j] = dna_base_index
                elif k == 2: DNi2[i,j] = dna_base_index
                elif k == 3: DNi3[i,j] = dna_base_index

    return DNi0, DNi1, DNi2, DNi3

