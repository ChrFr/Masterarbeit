# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data, io, transform
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.morphology import watershed, disk

from skimage import segmentation
from skimage.segmentation import mark_boundaries
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import color
from skimage.future import graph
from skimage.filters import sobel
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage import exposure
from skimage.filters import threshold_otsu, rank
from skimage.filters import gabor_kernel

def dilation():
    # Convert to float: Important for subtraction later which won't work with uint8
    #image = img_as_float(data.coins())
    image = io.imread('D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/DSC_5820.JPG')
    shape = image.shape
    shape = (shape[0] / 4, shape[1] / 4, shape[2])
    image = transform.resize(image, shape)
    image = gaussian_filter(image, 1)
    
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    
    dilated = reconstruction(seed, mask, method='dilation')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5), sharex=True, sharey=True)
    
    ax1.imshow(image)
    ax1.set_title('original image')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    
    ax2.imshow(dilated, vmin=image.min(), vmax=image.max())
    ax2.set_title('dilated')
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    
    ax3.imshow(image - dilated)
    ax3.set_title('image - dilated')
    ax3.axis('off')
    ax3.set_adjustable('box-forced')
    
    fig.tight_layout()
    plt.show()
    print()
    
def watershed_():    
    
    #image = img_as_ubyte(data.camera())
    image = io.imread('D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/DSC_5820.JPG')
    image = color.rgb2gray(image)
    
    shape = image.shape
    shape = (shape[0] / 4, shape[1] / 4)
    image = transform.resize(image, shape)    
    # denoise image
    denoised = rank.median(image, disk(2))
    
    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 50
    markers = ndi.label(markers)[0]
    
    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))
    
    # process the watershed
    labels = watershed(gradient, markers)
    
    # display results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title("Original")
    
    ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
    ax[1].set_title("Local Gradient")
    
    ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax[2].set_title("Markers")
    
    ax[3].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
    ax[3].set_title("Segmented")
    
    for a in ax:
        a.axis('off')
    
    fig.tight_layout()
    plt.show()    
    
def segmentation_():

    image = io.imread('D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/DSC_5820.JPG')
    shape = image.shape
    shape = (shape[0] / 4, shape[1] / 4, shape[2])
    img = transform.resize(image, shape)
    #gradient = sobel(color.rgb2gray(img))
    #ssegments_watershed = segmentation.watershed(gradient, markers=2, compactness=0.001)
    #segments_fz = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = segmentation.slic(img, n_segments=200, compactness=10, sigma=1)
    #segments_quick = segmentation.quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    
    #print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
    #print("Slic number of segments: %d" % len(np.unique(segments_slic)))
    #print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))
    
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    fig.set_size_inches(8, 3, forward=True)
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
    
    #ax[0].imshow(gradient)
    #ax[0].set_title("Felzenszwalbs's method")
    ax[1].imshow(mark_boundaries(img, segments_slic))
    ax[1].set_title("SLIC")
    #ax[2].imshow(mark_boundaries(img, segments_quick))
    #ax[2].set_title("Quickshift")
    #ax[3].imshow(mark_boundaries(img, segments_watershed))
    #ax[3].set_title('Compact watershed')
    for a in ax:
        a.set_xticks(())
        a.set_yticks(())
    plt.show()
    
def seg2():
    image = io.imread('D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/IMAG0780.JPG')
    #image = io.imread(u'D:/Eigene Dateien/Dokumente/Studium/Masterarbeit/Eigenes Set/16_10  04  Blaetter f. Christoph/DSC_8482.jpg')
    shape = image.shape
    shape = (shape[0] / 4, shape[1] / 4, shape[2])
    img = transform.resize(image, shape)    
    #img = exposure.equalize_adapthist(img)#, clip_limit=0.0005)
    
    #img = data.coffee()
    
    labels1 = segmentation.slic(img, compactness=30, n_segments=1000)
    out1 = color.label2rgb(labels1, img, kind='avg')
    
    g = graph.rag_mean_color(img, labels1)
    labels2 = graph.cut_threshold(labels1, g, 0.1)
    out2 = color.label2rgb(labels2, img, kind='avg')
    
    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True,
                           figsize=(6, 8))
    
    ax[0].imshow(img)
    ax[1].imshow(out1)
    ax[2].imshow(out2)
    
    for a in ax:
        a.axis('off')
    
    plt.tight_layout() 
    plt.show()
    
def active_c():
    # Test scipy version, since active contour is only possible
    # with recent scipy version
    import scipy
    split_version = scipy.__version__.split('.')
    if not(split_version[-1].isdigit()): # Remove dev string if present
        split_version.pop()
    scipy_version = list(map(int, split_version))
    new_scipy = scipy_version[0] > 0 or \
        (scipy_version[0] == 0 and scipy_version[1] >= 14)
    
    img = data.astronaut()
    #img = color.rgb2gray(img)
    image = io.imread('D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/IMAG0798.jpg')
    shape = image.shape
    shape = (shape[0] / 8, shape[1] / 8, shape[2])
    img = transform.resize(image, shape)    
    img = color.rgb2gray(img)
    
    #img = sobel(color.rgb2gray(img))
    
    s = np.linspace(0, 2*np.pi, shape[1])
    x = shape[1] / 2 + shape[1]/2*np.cos(s)
    y = shape[0] / 2 + shape[1]/2*np.sin(s)
    init = np.array([x, y]).T
    
    if not new_scipy:
        print('You are using an old version of scipy. '
              'Active contours is implemented for scipy versions '
              '0.14.0 and above.')
    
    if new_scipy:
        snake = active_contour(gaussian(img, 1),
                               init, alpha=0.05, beta=0.001, gamma=0.01)
    
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        plt.gray()
        ax.imshow(img)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])    
        plt.show()
        
def hsv_():
    image = io.imread('D:/Eigene Dateien/Dokumente/OneDrive/Dokumente/Studium/Master HTW/Masterarbeit/Repository/Python/src/masterarbeit/DSC_5827.JPG')
    image = color.rgb2hsv(image)
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                           figsize=(6, 8))
    
    ax[0].imshow(image)
    h = image[:,:,0]
    s = image[:,:,1]
    v = image[:,:,2]
    #img = img[:,:,1]
    threshold_global_otsu = threshold_otsu(s)
    global_otsu = s <= threshold_global_otsu
    ax[1].imshow(global_otsu)
    plt.show()
    
def gabor_():
    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats
    
    
    def match(feats, ref_feats):
        min_error = np.inf
        min_i = None
        for i in range(ref_feats.shape[0]):
            error = np.sum((feats - ref_feats[i, :])**2)
            if error < min_error:
                min_error = error
                min_i = i
        return min_i
    
    
    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    
    
    shrink = (slice(0, None, 3), slice(0, None, 3))
    brick = img_as_float(data.load('brick.png'))[shrink]
    grass = img_as_float(data.load('grass.png'))[shrink]
    wall = img_as_float(data.load('rough-wall.png'))[shrink]
    image_names = ('brick', 'grass', 'wall')
    images = (brick, grass, wall)
    
    # prepare reference features
    ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
    ref_feats[0, :, :] = compute_feats(brick, kernels)
    ref_feats[1, :, :] = compute_feats(grass, kernels)
    ref_feats[2, :, :] = compute_feats(wall, kernels)
    
    print('Rotated images matched against references using Gabor filter banks:')
    
    print('original: brick, rotated: 30deg, match result: ', end='')
    feats = compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels)
    print(image_names[match(feats, ref_feats)])
    
    print('original: brick, rotated: 70deg, match result: ', end='')
    feats = compute_feats(ndi.rotate(brick, angle=70, reshape=False), kernels)
    print(image_names[match(feats, ref_feats)])
    
    print('original: grass, rotated: 145deg, match result: ', end='')
    feats = compute_feats(ndi.rotate(grass, angle=145, reshape=False), kernels)
    print(image_names[match(feats, ref_feats)])
    
    
    def power(image, kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    
    # Plot a selection of the filter bank kernels and their responses.
    results = []
    kernel_params = []
    for theta in (0, 1):
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, [power(img, kernel) for img in images]))
    
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
    plt.gray()
    
    fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)
    
    axes[0][0].axis('off')
    
    # Plot original images
    for label, img, ax in zip(image_names, images, axes[0][1:]):
        ax.imshow(img)
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    
    for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
        # Plot Gabor kernel
        ax = ax_row[0]
        ax.imshow(np.real(kernel), interpolation='nearest')
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    
        # Plot Gabor responses with the contrast normalized for each filter
        vmin = np.min(powers)
        vmax = np.max(powers)
        for patch, ax in zip(powers, ax_row[1:]):
            ax.imshow(patch, vmin=vmin, vmax=vmax)
            ax.axis('off')
    
    plt.show()    
        
if __name__ == '__main__':
    seg2()
    