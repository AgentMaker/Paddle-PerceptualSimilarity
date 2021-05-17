import paddle.io

class BaseDataset(paddle.io.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        
    def name(self):
        return 'BaseDataset'
    
    def initialize(self):
        pass

