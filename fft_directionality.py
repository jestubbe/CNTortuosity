import numpy as np
from scipy.fft import fft2, fftshift

def init_fourier_fields(image):
    """
    Initialize Fourier fields based on the dimensions of the input image.

    Parameters:
    image (ndarray): The input image.

    Returns:
    dict: A dictionary containing initialization parameters for Fourier analysis.
    """
    # Compute dimensions
    height, width = image.shape[:2]
    
    if width == height:
        npadx = 1
        npady = 1
        long_side = width
        small_side = width
        step = 0
    else:
        small_side = min(width, height)
        long_side = max(width, height)
        npad = long_side // small_side + 1
        if width == long_side:
            npady = npad
            npadx = 1
        else:
            npady = 1
            npadx = npad
        delta = long_side - small_side
        step = delta // (npad - 1)
    
    # Compute the power of 2 image dimension for padding
    pad_size = 2
    while pad_size < small_side:
        pad_size *= 2

    return {
        "npadx": npadx,
        "npady": npady,
        "long_side": long_side,
        "small_side": small_side,
        "step": step,
        "pad_size": pad_size
    }

def make_r_matrix(nx, ny):
    """
    Generate a 2D matrix of radius values centered in the middle of the image.

    Parameters:
    nx (int): The width of the matrix.
    ny (int): The height of the matrix.

    Returns:
    ndarray: A 2D array of radius values.
    """
    xc, yc = nx // 2, ny // 2
    x, y = np.meshgrid(np.arange(nx) - xc, np.arange(ny) - yc)
    r = np.sqrt(x ** 2 + y ** 2).astype(np.float32)
    return r

def make_theta_matrix(nx, ny):
    """
    Generate a 2D matrix of angle values centered in the middle of the image.

    Parameters:
    nx (int): The width of the matrix.
    ny (int): The height of the matrix.

    Returns:
    ndarray: A 2D array of angle values in radians.
    """
    xc, yc = nx // 2, ny // 2
    x, y = np.meshgrid(np.arange(nx) - xc, np.arange(ny) - yc)
    theta = np.arctan2(-y, x).astype(np.float32)
    return theta

def make_fft_filters(pad_size, nbins, r, theta, FREQ_THRESHOLD=5):
    """
    Generate angular filters for Fourier analysis.

    Parameters:
    pad_size (int): The size of the padded image.
    nbins (int): The number of angular bins.
    r (ndarray): The radius matrix.
    theta (ndarray): The theta matrix.
    FREQ_THRESHOLD (int, optional): The frequency threshold. Defaults to 5.

    Returns:
    ndarray: A 3D array containing the filters.
    """
    bins = np.linspace(0, np.pi, nbins + 1)[:-1]
    filters = np.zeros((nbins, pad_size, pad_size), dtype=np.float32)

    r_px = r.flatten()
    theta_px = theta.flatten()
    theta_bw = bins[1] - bins[0]
    r_c = pad_size / 4
    r_bw = r_c / 2

    for i in range(nbins):
        theta_c = bins[i]
        theta_c = np.mod(theta_c, np.pi) - np.pi / 2  # Handle angle wrapping

        # Compute radial part: Gaussian centered at r_c with bandwidth r_bw
        radial_part = np.exp(-((r_px - r_c) ** 2) / (r_bw ** 2))

        # Compute angular part: Cosine function centered at theta_c
        angle_diff = np.mod(theta_px - theta_c, 2 * np.pi)
        angle_diff[angle_diff > np.pi] -= 2 * np.pi
        angular_part = np.cos(angle_diff / theta_bw * np.pi / 2)
        angular_part[np.abs(angle_diff) >= theta_bw] = 0

        # Filter out low and high frequencies
        radial_mask = np.logical_and(r_px >= FREQ_THRESHOLD, r_px <= pad_size / 2)

        # Combine radial and angular parts
        pixels = (angular_part ** 2) * radial_part
        pixels[~radial_mask] = 0

        filters[i] = pixels.reshape((pad_size, pad_size))

    return filters

def fourier_component(ip, pad_size, small_side, npadx, npady, step, filters, nbins, window_pixels):
    """
    Perform Fourier component analysis on the image.

    Parameters:
    ip (ndarray): The input image.
    pad_size (int): The size of the padded image.
    small_side (int): The size of the small side of the image.
    npadx (int): Number of horizontal padding steps.
    npady (int): Number of vertical padding steps.
    step (int): Step size for padding.
    filters (ndarray): The filters to be applied.
    nbins (int): The number of angular bins.
    window_pixels (ndarray): The windowing function applied to the image.

    Returns:
    ndarray: The directional components of the image.
    """
    original_square = ((pad_size - small_side) // 2, (pad_size - small_side) // 2, small_side, small_side)
    dirs = np.zeros(nbins, dtype=np.float64)

    for ix in range(npadx):
        for iy in range(npady):
            # Extract and window the square block from the image
            square_block = ip[ix * step: ix * step + small_side, iy * step: iy * step + small_side]
            square_block = square_block * window_pixels

            # Pad the square block to pad_size
            padded_square_block = np.zeros((pad_size, pad_size), dtype=np.float32)
            padded_square_block[
                (pad_size - small_side) // 2: (pad_size - small_side) // 2 + small_side,
                (pad_size - small_side) // 2: (pad_size - small_side) // 2 + small_side
            ] = square_block

            # Compute the FFT and power spectrum
            fft = fftshift(fft2(padded_square_block))
            pspectrum = np.abs(fft) ** 2

            # Center the power spectrum with the right size
            centered_pspectrum = pspectrum[
                original_square[0]: original_square[0] + original_square[2],
                original_square[1]: original_square[1] + original_square[3]
            ]

            # Pad the centered power spectrum back to pad_size
            padded_pspectrum = np.zeros((pad_size, pad_size), dtype=np.float32)
            padded_pspectrum[
                (pad_size - small_side) // 2: (pad_size - small_side) // 2 + small_side,
                (pad_size - small_side) // 2: (pad_size - small_side) // 2 + small_side
            ] = centered_pspectrum

            # Apply filters and compute directional components
            for b in range(nbins):
                dirs[b] += np.sum(padded_pspectrum.flatten() * filters[b].flatten())

    return dirs / np.sum(dirs)
    
    
def dir_table(image, nbins=90):
    # Initialize Fourier fields
    fields = init_fourier_fields(image)
    npadx, npady, long_side, small_side, step, pad_size = fields.values()
    
    # Create windowing function
    window_pixels = np.outer(np.blackman(small_side),np.blackman(small_side))
    
    # Create radial and theta matrices
    r = make_r_matrix(pad_size, pad_size)
    theta = make_theta_matrix(pad_size, pad_size)
    
    # Create FFT filters
    filters = make_fft_filters(pad_size, nbins, r, theta)
    
    # Perform Fourier component analysis
    return fourier_component(image, pad_size, small_side, npadx, npady, step, filters, nbins, window_pixels)
    
    