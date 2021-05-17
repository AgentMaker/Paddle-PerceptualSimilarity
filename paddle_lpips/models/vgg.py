import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url
from typing import Any


__all__ = ['vgg16']


model_urls = {
    'vgg16': 'https://bj.bcebos.com/v1/ai-studio-online/76647bc7b65246d6b35b076c3f8685a73aa810ba8d7a4a0abdf2fba6bb4a84d1?/lpips_vgg16.pdparams',
}


def vgg16(pretrained: bool = False, from_torch: bool = False, **kwargs: Any) -> paddle.vision.models.vgg.VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        from_torch (bool): If True, converts weights from a pytorch model pre-trained on ImageNet
    """
    model = paddle.vision.models.vgg16(pretrained=False)
    if from_torch:
        from torchvision import models as tv
        model_torch = tv.vgg16(pretrained=True)
        state_dict = {
            k: paddle.to_tensor(v.cpu().numpy().astype('float32')) if v.ndim != 2 else 
               paddle.to_tensor(v.cpu().numpy().transpose([1,0]).astype('float32'))
            for k, v in model_torch.state_dict().items()
        }
        model.set_state_dict(state_dict)
    if pretrained:
        state_dict = paddle.load(get_weights_path_from_url(model_urls['vgg16']))
        model.set_state_dict(state_dict)
    return model


if __name__ == '__main__':
    import torch
    from torchvision import models as tv
    
    model_torch = tv.vgg16(pretrained=True)
    model_torch.eval()
    model_paddle = vgg16(pretrained=False, from_torch=True)
    model_paddle.eval()

    e = 0
    for i in range(100):
        with torch.no_grad():
            x = torch.randn(1,3,64,64)
            y_torch = model_torch(x).cpu().numpy()
        x = paddle.to_tensor(x.cpu().numpy())
        with paddle.no_grad():
            y_paddle = model_paddle(x).numpy()
        e = e + (y_torch - y_paddle).mean()
    e = abs(e / 100)

    print(f'Converting vgg16 pretrained model from pytorch to paddlepaddle with error {float(e)}...')
    paddle.save(model_paddle.state_dict(), 'lpips_vgg16.pdparams')
    print('Converting task finished.')
