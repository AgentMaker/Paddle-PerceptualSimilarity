import paddle
import paddle_lpips as lpips
from IPython import embed

use_gpu = False         # Whether to use GPU
spatial = True         # Return a spatial map of perceptual distance.

if(use_gpu):
	paddle.set_device('gpu')
else:
	paddle.set_device('cpu')

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'alex' 
# loss_fn = lpips.LPIPS(net='alex, spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'alex' 

## Example usage with dummy tensors
dummy_im0 = paddle.zeros([1,3,64,64]) # image should be RGB, normalized to [-1,1]
dummy_im1 = paddle.zeros([1,3,64,64])
dist = loss_fn.forward(dummy_im0,dummy_im1)

## Example usage with images
ex_ref = lpips.im2tensor(lpips.load_image('./imgs/ex_ref.png'))
ex_p0 = lpips.im2tensor(lpips.load_image('./imgs/ex_p0.png'))
ex_p1 = lpips.im2tensor(lpips.load_image('./imgs/ex_p1.png'))

ex_d0 = loss_fn.forward(ex_ref,ex_p0)
ex_d1 = loss_fn.forward(ex_ref,ex_p1)

if not spatial:
    print('Distances: (%.3f, %.3f)'%(ex_d0, ex_d1))
else:
    print('Distances: (%.3f, %.3f)'%(ex_d0.mean(), ex_d1.mean()))            # The mean distance is approximately the same as the non-spatial distance
    
    # Visualize a spatially-varying distance map between ex_p0 and ex_ref
    import pylab
    pylab.imshow(ex_d0[0,0].numpy())
    pylab.show()
