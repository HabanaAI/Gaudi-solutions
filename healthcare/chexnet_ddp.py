import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
import habana_frameworks.torch.distributed.hccl
from habana_frameworks.torch.utils.library_loader import load_habana_module
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
import argparse
import os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


# Create ChestXrayDataSet class for loading data using data primitive torch.utils.data.Dataset
class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        '''
        data_dir: path to image directory.
        image_list_file: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        '''
        image_names = []
        labels = []
        with open(image_list_file, 'r') as f:
            for line in f:
                items = line.split()
                image_name = os.path.join(data_dir, items[0])
                image_names.append(image_name)
                label = [int(i) for i in items[1:]]
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        '''
        index: the index of item
        Return image and its labels
        '''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


N_CLASSES = 14
CLASS_NAMES = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia']


# Download densenet121 model and modify its classifier to 14 classes.
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_features, out_size)

    def forward(self, x):
        x = self.densenet121(x)
        return x

# Initialize the default distributed process group and distributed package


def init_distributed_mode(args):

    world_size = int(os.environ[args.env_world_size])
    local_rank = int(os.environ[args.env_rank])

    print('distributed init (rank {})'.format(local_rank), flush=True)

    os.environ['ID'] = str(local_rank)
    backend = 'hccl'

    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=local_rank)


# Define main function that will allow training the model on multiple Gaudi machines
def main(args):

    # Get environment variables related to Data Distributed Parallel (DDP)
    torch.multiprocessing.set_start_method('spawn')
    world_size = int(os.environ[args.env_world_size])
    local_rank = int(os.environ[args.env_rank])

    if local_rank == 0:
        print(vars(args))

    # Load habana module and define HPU device
    load_habana_module()
    device = torch.device('hpu')

    os.environ['PT_HPU_LAZY_MODE'] = '1'

    if world_size > 1:
        init_distributed_mode(args)

    print('Using %s device.' % (device))

    # Compose together all image tranformations (resize, normalize, casting to tensor)
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])

    transform = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        normalize
    ])

    # Load training dataset and pass transform object to loader
    train_dataset = ChestXrayDataSet(
        data_dir=args.data_dir,
        image_list_file=args.train_image_list,
        transform=transform,
    )

    train_sampler = None
    if world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)

    # sampler option is mutually exclusive with shuffle
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    if local_rank == 0:
        print('training %d batches %d images' %
              (len(train_loader), len(train_dataset)))

    # Initialize and load the model
    net = DenseNet121(N_CLASSES)

    if args.model_path:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        if local_rank == 0:
            print('model state has loaded.')

    # Set model to HPU device and prepare it for training
    net = net.to(device)
    # net.load_state_dict(torch.load('model/checkpoint_2classes.pth'))
    optimizer = FusedAdamW(net.parameters(), lr=args.lr)

    if world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        bucket_cap_mb=100,
                                                        broadcast_buffers=False,
                                                        gradient_as_bucket_view=True)

    # Offset imbalance of training data
    fraction_of_positive_per_class = np.array([
        0.1019, 0.0248, 0.1180, 0.1774, 0.0508, 0.0557, 0.0125, 0.0472, 0.0416,
        0.0215, 0.0229, 0.0149, 0.0290, 0.0018
    ])
    pos_weight = (1-fraction_of_positive_per_class) / \
        fraction_of_positive_per_class
    pos_weight = torch.from_numpy(pos_weight).float()
    pos_weight.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Run training loop
    for epoch in range(args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        # initialize the ground truth and output tensor
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        net.train()
        mean_loss = 0.0
        mean_acc = 0.0
        with tqdm(desc=f'Epoch {epoch} - ', unit='it', total=len(train_loader)) as pbar:
            for index, (images, labels) in enumerate(train_loader, 1):

                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                loss = criterion(outputs, labels)

                # backward and optimize
                optimizer.zero_grad()

                loss.backward()
                htcore.mark_step()
                optimizer.step()
                htcore.mark_step()

                pbar.update()

                # mid-training loss and accuracy
                y_true = labels.detach().cpu()
                y_pred_org = outputs.detach().cpu()
                y_pred = (y_pred_org > 0).float()
                loss_val = loss.item()

                mean_loss += loss_val
                acc = accuracy_score(y_true[:], y_pred[:])
                mean_acc += acc

                if local_rank == 0 and index % 10 == 0:
                    torch.set_printoptions(precision=2)
                    print('\repoch %3d/%3d batch %5d/%5d' % (epoch+1, args.epochs, index, len(train_loader)), end='')
                    print(' train loss %6.4f' % mean_loss, end='')
                    print(' Accuracy: ', mean_acc)
                    mean_loss = 0
                    mean_acc = 0

        if local_rank == 0:
            print('')
            torch.save(net.state_dict(),
                       f'model/checkpoint_{epoch}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_world_size', default='WORLD_SIZE', type=str)
    parser.add_argument('--env_rank', default='LOCAL_RANK', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--data_dir', default='images', type=str)
    parser.add_argument('--hpu', action='store_true', default=False)
    parser.add_argument('--use_lazy_mode', action='store_true', default=False)
    parser.add_argument('--train_image_list',
                        default='labels/train_list.txt', type=str)
    args = parser.parse_args()

    main(args)
