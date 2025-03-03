import cv2
import numpy as np 
import pycuda.driver as cuda 
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time 
import ctypes 


def detect_objects(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image couldn't be loaded")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    objects = []
    for i, (x, y, w, h) in enumerate(faces):
        objects.append((x, y, w, h))
        print(f"\nObject {i}: x={x}, y={y}, width={w}, height={h}")

    return img, objects

def select_object(objects):
    if not objects:
        return None
    
    selection = int(input("\nEnter the object number to blur: "))
    if 0 <= selection < len(objects):
        return objects[selection]
    else:
        return None

def blur_with_python(image, x, y, w, h, blur_radius):
    roi = image[y:y+h, x:x+w]
    blurred_roi = cv2.GaussianBlur(roi, (blur_radius, blur_radius), 0)
    image[y:y+h, x:x+w] = blurred_roi
    return image



image_path = 'image.jpg'
blur_radius = 3

cuda_lib = ctypes.CDLL("./blur_kernel.so")

cuda_lib.blur_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),     # unsigned char type input image
    ctypes.POINTER(ctypes.c_ubyte),     # unsigned char type output image
    ctypes.c_int,                       # int width
    ctypes.c_int,                       # int height
    ctypes.c_int                        # int blur_radius
]

def blur_with_cuda(image, x, y, w, h, blur_radius): 
    # extract region of interest
    roi = image[y:y+h, x:x+w]
    height, width = roi.shape

    # allocating memory on HOST
    blurred_roi = np.empty_like(roi)

    # kernel launch
    block_size = (32, 32)
    grid_size = ((width + block_size[0] - 1)//block_size[0], (height + block_size[1] - 1)//block_size[1], 1)

    # getting pointers to the data
    in_ptr = roi.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    out_ptr = blurred_roi.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # kernel call
    start = time.time()
    cuda_lib.blur_kernel(in_ptr, out_ptr, width, height, blur_radius, block=block_size, grid=grid_size)
    
    cuda.Context.synchronize()
    end = time.time()
    cuda_time = end - start 

    # placing blurred roi onto the image
    image[y:y+h, x:x+w] = blurred_roi 

    return image, cuda_time 
    

try:
    image, objects = detect_objects(image_path)

    if objects:
        selected_object = select_object(objects)
        if selected_object:
            x, y, w, h = selected_object
            blurred_image, cuda_time = blur_with_cuda(image, x, y, w, h, blur_radius)
            
            print(f"\nCUDA Blur Time: {cuda_time:.5f} seconds")
            
            cv2.imwrite("blurred_output.jpg", blurred_image)

            start = time.time()
            blurred_python = blur_with_python(image.copy(), x, y, w, h, blur_radius)
            end = time.time()
            python_time = end - start
            print(f"\nPython Blur Time: {python_time:.5f} seconds")
            cv2.imwrite("blurred_python.jpg", blurred_python)

            print(f"\nCUDA Blur {python_time/cuda_time:.2f}x faster!\n")

        else:
            print("Invalid Object Selection")
    else:
        print("No Objects Detected")

except ValueError as e:
    print(e)
except Exception as e:
    print("Error: {e}")

