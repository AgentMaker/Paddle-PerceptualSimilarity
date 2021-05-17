import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url
from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://bj.bcebos.com/v1/ai-studio-online/bc13fe1a96fc4ccca19394e237a4e3b5d49572730a9f42f79cd01d42cf1a7e01?/lpips_alex.pdparams',
}


class AlexNet(nn.Layer):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            nn.Conv2D(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, from_torch: bool = False, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        from_torch (bool): If True, converts weights from a pytorch model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if from_torch:
        from torchvision import models as tv
        model_torch = tv.alexnet(pretrained=True)
        state_dict = {
            k: paddle.to_tensor(v.cpu().numpy().astype('float32')) if v.ndim != 2 else 
               paddle.to_tensor(v.cpu().numpy().transpose([1,0]).astype('float32'))
            for k, v in model_torch.state_dict().items()
        }
        model.set_state_dict(state_dict)
    if pretrained:
        state_dict = paddle.load(get_weights_path_from_url(model_urls['alexnet']))
        model.set_state_dict(state_dict)
    return model


if __name__ == '__main__':
    import torch
    from torchvision import models as tv
    
    model_torch = tv.alexnet(pretrained=True)
    model_torch.eval()
    model_paddle = alexnet(pretrained=False, from_torch=True)
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

    print(f'Converting alexnet pretrained model from pytorch to paddlepaddle with error {float(e)}...')
    paddle.save(model_paddle.state_dict(), 'lpips_alex.pdparams')
    print('Converting task finished.')
