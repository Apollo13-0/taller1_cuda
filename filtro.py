import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

imagen = cv2.imread('img/image5.png', cv2.IMREAD_GRAYSCALE)

inicio = time.time()

# Definir las masks
sobel_x = np.array([[-1, 0, 1], 
                    [-2, 0, 2], 
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]])

#Convolucion para calcular gradiente en X y Y
gradiente_x = cv2.filter2D(imagen, -1, sobel_x)
gradiente_y = cv2.filter2D(imagen, -1, sobel_y)

#magnitud del gradiente
magnitud = np.sqrt(gradiente_x**2 + gradiente_y**2)

# Normalizar la magnitud
magnitud = np.uint8(255 * magnitud / np.max(magnitud))

fin = time.time()
tiempo_ejecucion = fin - inicio

print(f"Tiempo de ejecucion: {tiempo_ejecucion:.6f} segundos")

plt.imshow(magnitud, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')
plt.show()
