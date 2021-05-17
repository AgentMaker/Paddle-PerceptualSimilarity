if __name__ == '__main__':
    import os
    import glob
    import torch
    import paddle


    dirname = os.path.dirname(os.path.abspath(__file__))
    weight_files = glob.glob(f'{dirname}/**/*.pth', recursive=True)

    for weight_file in weight_files:
        new_file = weight_file.replace('.pth', '.pdparams')
        print(f'Converting weight file from `{weight_file}` to `{new_file}`...')
        state_dict = torch.load(weight_file)
        state_dict = {k: paddle.to_tensor(v.cpu().numpy().astype('float32')) for k, v in state_dict.items()}
        paddle.save(state_dict, new_file)
        print('Converting task finished.')
