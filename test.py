"""
TEST GANOMALY

. Example: Run the following command from the terminal.
    #python test.py 
    # --dataset fashion_mnist 
    # --isize 32 
    # --nc 1 
    # --niter 15 
    # --abnormal_class "Bag" 
    # --manualseed 0 
    # --count_test 25 
    # --class_test "Bag"
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data_mod import load_data
from lib.model import Ganomaly
import torch
from torchvision import datasets, transforms
import os
import torchvision.utils as vutils

def get_image(opt):
    if opt.dataset in ['fashion_mnist']:
        classes = {'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3, 'Coat': 4, 'Sandal': 5,
                        'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9 }
        target_label = classes[opt.class_test]
        transform = transforms.Compose(
                [
                    transforms.Resize(opt.isize),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )

        # Carica il dataset Fashion-MNIST
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        indices = torch.where(dataset.targets == target_label)[0]
    elif opt.dataset in ['mnist']:
        target_label = int(opt.class_test)
        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        indices = torch.where(dataset.targets == target_label)[0]
    elif opt.dataset in ['cifar10']:
        classes = {
            'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }
        target_label = classes[opt.class_test]
        transform = transforms.Compose(
            [
                transforms.Resize(opt.isize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        indices = torch.where(torch.tensor(dataset.targets) == target_label)[0]
    else:
        print("Errore nome dataset!")
        exit(1)

    random_index = torch.randperm(len(indices))[0]
    image, label = dataset[indices[random_index]]
    image = image.unsqueeze(0)
    label_tensor = torch.tensor(label).view(1)
    return [image, label_tensor]

##
def test():
    """ Testing
    """
    ##
    # ARGUMENTS
    opt = Options().parse()
    data = get_image(opt)
    
    ##
    # LOAD DATA
    # dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, None)
    ##
    # TEST MODEL
    image_fake, error = model.test_one_image(data)
    
    dst = os.path.join(opt.outf, opt.name, str(opt.perc_pullation), 'test', 'images')
    if not os.path.isdir(dst):
        os.makedirs(dst)
        
    i=opt.count_test
    vutils.save_image(data[0], '%s/real_%03d.jpg' % (dst, i+1), normalize=True)
    vutils.save_image(image_fake, '%s/fake_%03d.jpg' % (dst, i+1), normalize=True)

    with open(os.path.join(dst, "score.txt"), "a") as file:
        file.write("CLASSE ANOMALA=" + opt.abnormal_class + "\n" + str(i+1) +" = " + str(error.item()) + "\n\n")
    print("ANOMALY SCORE: ", error.item())
    

if __name__ == '__main__':
    test()
