# Operaciones de Matrices con CUDA y Python

Este proyecto contiene varios scripts en CUDA y Python para realizar operaciones de matrices y detección de bordes.

## Archivos

- `multiplicacionMat.cu`: Programa en CUDA para multiplicar dos matrices 4x4.
- `vecadd.cu`: Programa en CUDA para la suma de vectores.
- `filtro.py`: Script en Python para aplicar un filtro a una imagen.
- `edge_detect.cu`: Programa en CUDA para la detección de bordes en imágenes.

## Requisitos

- CUDA Toolkit
- Python 3
- NumPy (para `filtro.py`)
- OpenCV (para `filtro.py`)

## Compilación y Ejecución

### `multiplicacionMat.cu`

1. Compila el programa en CUDA:
    ```sh
    nvcc multiplicacionMat.cu -o multiplicacionMat
    ```

2. Ejecuta el archivo compilado:
    ```sh
    ./multiplicacionMat
    ```

### `vecadd.cu`

1. Compila el programa en CUDA:
    ```sh
    nvcc vecadd.cu -o vecadd
    ```

2. Ejecuta el archivo compilado:
    ```sh
    ./vecadd
    ```

### `filtro.py`

1. Ejecuta el script en Python:
    ```sh
    python3 filtro.py
    ```

### `edge_detect.cu`

1. Compila el programa en CUDA:
    ```sh
    nvcc edge_detect.cu -o edge_detect
    ```

2. Ejecuta el archivo compilado:
    ```sh
    ./edge_detect imageName.png
    ```

