import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_1']

model_urls = {
    'squeezenet1_1': 'https://bj.bcebos.com/v1/ai-studio-online/315295ebb9d3459c9cbb8d1798ba24e62376d63ad9b64ff99c0af853da06a750?/lpips_sq11.pdparams',
}


class Fire(nn.Layer):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2D(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2D(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2D(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return paddle.concat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Layer):

    def __init__(
        self,
        version: str = '1_1',
        num_classes: int = 1000
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2D(3, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2D(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2D(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(),
            nn.AdaptiveAvgPool2D((1, 1))
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return paddle.flatten(x, 1)


def _squeezenet(version: str, pretrained: bool, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = paddle.load(get_weights_path_from_url(model_urls[arch]))
        model.set_state_dict(state_dict)
    return model


def squeezenet1_1(pretrained: bool = False, from_torch: bool = False, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        from_torch (bool): If True, converts weights from a pytorch model pre-trained on ImageNet
    """
    model = _squeezenet('1_1', pretrained, **kwargs)
    if from_torch:
        from torchvision import models as tv
        model_torch = tv.squeezenet1_1(pretrained=True)
        state_dict = {k: paddle.to_tensor(v.cpu().numpy().astype('float32')) for k, v in model_torch.state_dict().items()}
        model.set_state_dict(state_dict)
    return model


if __name__ == '__main__':
    import torch
    from torchvision import models as tv
    
    model_torch = tv.squeezenet1_1(pretrained=True)
    model_torch.eval()
    model_paddle = squeezenet1_1(pretrained=False, from_torch=True)
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

    print(f'Converting squeezenet1_1 pretrained model from pytorch to paddlepaddle with error {float(e)}...')
    paddle.save(model_paddle.state_dict(), 'lpips_sq11.pdparams')
    print('Converting task finished.')
