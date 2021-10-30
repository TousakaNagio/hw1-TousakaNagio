import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from model import Net
from PIL import Image


def main(config):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = Net().to(device)

    state = torch.load(config.model_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)
    out_filename = config.save_dir
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn)
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                output = model(data.to(device))
                pred = output.max(1, keepdim=True)[1]
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='./hw1_data/p1_data/val_50')
    parser.add_argument('--save_dir', type=str, default='../')
    parser.add_argument('--model_path', default='./model2500.pth', type=str, help='Checkpoint path.')
    
    config = parser.parse_args()
    print(config)
    main(config)
