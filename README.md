# (Almost Canny) Edge Detection 

A [Canny-style ](https://en.wikipedia.org/wiki/Canny_edge_detector) edge detector implemented in both C++ (CPU) and CUDA (GPU). Given a JPEG image, it outputs the edges as an image.

## Methodology

1. **Greyscale conversion**: RGB to single channel using `L = 0.21R + 0.72G + 0.07B`
2. **Gaussian blur**: 3×3 kernel to suppress noise before edge detection
3. **Sobel filter**: horizontal (Gx) and vertical (Gy) gradient computation
4. **Magnitude & thresholding**: `G = sqrt(Gx² + Gy²)`, pixels above threshold are marked as edges

## Requirements

- C++20 or later (for type_traits)
- CUDA Toolkit (for the GPU build)

## Building & running

**CPU:**

```
make cpu
```

**CUDA:**
```
make cuda
```

## Results
As expected, Cuda version (\~3.5 ms) outperforms the CPU implementation (\~42.5 ms).

![Original image](/resources/turin.jpg)
![Output image](/resources/edges_gpu.jpg)


## TO-DO
Adding non-maximum suppression and hysteresis to make it a complete Canny implementation.

