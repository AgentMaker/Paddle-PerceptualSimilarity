import argparse
import paddle_lpips as lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

if(opt.use_gpu):
	paddle.set_device('gpu')
else:
	paddle.set_device('cpu')

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)

# Load images
img0 = lpips.im2tensor(lpips.load_image(opt.path0)) # RGB image from [-1,1]
img1 = lpips.im2tensor(lpips.load_image(opt.path1))

# Compute distance
dist01 = loss_fn.forward(img0,img1)
print('Distance: %.3f'%dist01)
