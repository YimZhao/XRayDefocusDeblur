import numpy as np
import scipy.fftpack as sf
import math
import matplotlib.pyplot as plt
import numpy as np

# angular spectrum method
def propagate(array, direction, x_pixel_size_m, y_pixel_size_m, wavelength_m, z_m):
    k = 2. * np.pi / wavelength_m
    nx, ny = np.shape(array)
    spectrum = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(array))) / np.sqrt(np.size(array))

    dkx = 2.*np.pi / (nx * x_pixel_size_m)
    dky = 2.*np.pi / (ny * y_pixel_size_m)

    skx = dkx * nx / 2
    sky = dky * ny / 2

    kxx = np.linspace(-skx, skx-dkx, nx)
    kyy = np.linspace(-sky, sky-dky, ny)
    kx,ky = np.meshgrid(kxx, kyy)

    phase = np.sqrt(k**2 - kx**2 - ky**2) * z_m * direction

    spectrum *= np.exp(1j*phase)
    array_prop = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(spectrum))) * np.sqrt(np.size(array))
    
    return array_prop

def propagate_pupil(array, direction, x_pixel_size_m, y_pixel_size_m, wavelength_m, z_m):
    far_field_m = z_m
    nx, ny = np.shape(array)

    x_pixel_size_detector_m = wavelength_m * far_field_m / (nx * x_pixel_size_m)
    y_pixel_size_detector_m = wavelength_m * far_field_m / (ny * y_pixel_size_m)

    print(far_field_m, direction, x_pixel_size_detector_m)
    array_prop = np.zeros_like(array)
    phase_onplane2 = np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            phase_temp = np.pi * (((i - nx/2)*x_pixel_size_m)**2 + ((j - ny/2)*y_pixel_size_m)**2) \
                / (wavelength_m * far_field_m * direction)
            array_prop[i,j] = array[i,j] * np.exp(1j * phase_temp)
            phase_onplane2[i,j] = np.pi * (((i - nx/2)*x_pixel_size_detector_m)**2 + ((j - ny/2)*y_pixel_size_detector_m)**2) \
                / (wavelength_m * far_field_m * direction)

    array_prop = np.exp(1j * phase_onplane2) \
        * np.fft.fftshift(np.fft.fftn(np.fft.fftshift(array_prop))) / np.sqrt(np.size(array_prop)*1.)
    return array_prop
