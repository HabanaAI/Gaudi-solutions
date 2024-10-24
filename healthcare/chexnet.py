# pip install numpy Pillow sklearn pandas torch torchvision matplotlib tqdm
# https://nihcc.app.box.com/v/ChestXray-NIHCC
import os
import argparse
from tqdm import tqdm
import time

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms

from habana_frameworks.torch.utils.library_loader import load_habana_module
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.distributed.hccl
from habana_frameworks.torch.hpu import wrap_in_hpu_graph
from habana_frameworks.torch.hpex.optimizers import FusedAdamW


class ChestXrayData(torch.utils.data.Dataset):
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
        return self.get_sample(index, self.transform)

    def get_sample(self, index, transform):
        '''
        index: the index of item
        transform: the transform used on item
        Return transformed image and its labels
        '''
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]

        if transform:
            image = transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

    # We assume the ChestX-ray14 dataset downloaded in local storage.


# preprocessing for visualization
resize_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

# preprocessing for model input
normalize_transform = transforms.Compose([
    resize_transform,
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

CLASS_COUNT = 14
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
    'Hernia'
]

# Offset imbalance of training data
fraction_of_positive_per_class = np.array([
    0.1019, 0.0248, 0.1180, 0.1774, 0.0508, 0.0557, 0.0125, 0.0472, 0.0416,
    0.0215, 0.0229, 0.0149, 0.0290, 0.0018
])
pos_weight = (1-fraction_of_positive_per_class) / \
    fraction_of_positive_per_class
pos_weight = torch.from_numpy(pos_weight).float()

class CheXNet(nn.Module):
    def __init__(self, out_size):
        super(CheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_features, out_size)

    def forward(self, x):
        x = self.densenet121(x)
        return x

def inference(model, data, device, args):
    test_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    model.eval()
    iterations = len(test_loader)
    if args.iterations is not None:
        iterations = min(iterations, args.iterations)

    total_processed_samples_number = test_loader.batch_size * iterations

    mean_acc = 0.0
    forward_time = 0
    loop_start_time = time.time()
    with torch.no_grad():
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        with tqdm(desc=f'Evaluation: ', unit='it', total=iterations) as pbar:
            for indx, (images, labels) in enumerate(test_loader, 1):
                images = images.to(device)
                labels = labels.to(device)

                forward_start_time = time.time()
                outputs = model(images)
                # for perf mesuraments
                htcore.mark_step()
                forward_time += time.time() - forward_start_time

                pbar.update()

                # accuracy
                if args.check_accuracy:
                    y_true = labels.detach().cpu()
                    y_pred_org = outputs.detach().cpu()
                    y_pred = (y_pred_org > 0).float()

                    acc = accuracy_score(y_true[:], y_pred[:])
                    mean_acc += acc

                if indx == iterations:
                    break

    loop_time = time.time() - loop_start_time

    return mean_acc, loop_time, forward_time, total_processed_samples_number



def trainig(model, data, device, args):
    # sampler option is mutually exclusive with shuffle
    train_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = FusedAdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = 1
    iterations = len(train_loader)
    total_processed_samples_number = None
    if args.iterations is None and args.epochs is not None:
        epochs = args.epochs
        iterations = len(train_loader) * epochs
    elif args.epochs is None and args.iterations is not None:
        iterations = args.iterations
        epochs = iterations // len(train_loader) + 1
    else:
        raise RuntimeError("Please, define number of epochs or iterations.")

    total_processed_samples_number = train_loader.batch_size * iterations

    forward_backward_time = 0
    final_accuracy = 0
    training_loop_start_time = time.time()
    model.train()
    for epoch in range(epochs):
        # initialize the ground truth and output tensor
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        mean_loss = 0.0
        mean_acc = 0.0
        with tqdm(desc=f'Epoch {epoch + 1}: ', unit='it',
                  total=(iterations - epoch * len(train_loader))
            ) as pbar:
            for index, (images, labels) in enumerate(train_loader, 1):
                images = images.to(device)
                labels = labels.to(device)

                forward_start_time = time.time()
                outputs = model(images)

                loss = criterion(outputs, labels)

                # backward and optimize
                optimizer.zero_grad()

                loss.backward()
                htcore.mark_step()
                optimizer.step()
                htcore.mark_step()
                forward_backward_time += time.time() - forward_start_time

                pbar.update()

                # mid-training loss and accuracy
                if args.check_accuracy:
                    y_true = labels.detach().cpu()
                    y_pred_org = outputs.detach().cpu()
                    y_pred = (y_pred_org > 0).float()
                    loss_val = loss.item()

                    mean_loss += loss_val
                    acc = accuracy_score(y_true[:], y_pred[:])
                    mean_acc += acc
                    final_accuracy = mean_acc

                    if index % 10 == 0:
                        print()
                        torch.set_printoptions(precision=2)
                        print('\repoch %3d/%3d batch %5d/%5d' % \
                            (epoch+1, epochs, index, len(train_loader)),
                            end=''
                        )
                        print(f' Train loss {mean_loss}', end='')
                        print(f' Accuracy: {mean_acc}')
                        mean_loss = 0
                        mean_acc = 0

                if epoch == epochs - 1 and index == iterations % len(train_loader):
                    break

        model_dir = args.output_dir + '/model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        models_path = model_dir + f'/checkpoint_{epoch}.pth'
        torch.save(model.state_dict(), models_path)
        print(f'Model has been saved: {models_path}')

    trainig_loop_time = time.time() - training_loop_start_time
    return final_accuracy, trainig_loop_time, forward_backward_time, total_processed_samples_number



def main(args):
    # ./images - directory containing image files
    # ./labels - directory containing .txt files with list of image names and their respective
    #            ground truth labels on every row
    data_dir = args.data_dir
    images_dir = data_dir + '/images'
    label_dir = data_dir + '/labels'

    # Each line contains the filename of an x-ray image and respective labels
    train_image_list_file = label_dir + '/train_list.txt'
    test_image_list_file = label_dir + '/test_list.txt'

    device = 'cpu'
    if args.hpu:
        load_habana_module()
        device = 'hpu'
        if args.use_lazy_mode:
            os.environ['PT_HPU_LAZY_MODE'] = '1'
        else:
            os.environ['PT_HPU_LAZY_MODE'] = '0'

    model = CheXNet(CLASS_COUNT)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

    model = model.to(device)
    # model = wrap_in_hpu_graph(model)

    accuracy = None
    loop_time = None
    forward_backward_time = None
    forward_time = None
    throughput = None
    if args.training:
        # train data set object
        train_data = ChestXrayData(
            data_dir=images_dir,
            image_list_file=train_image_list_file,
            transform=normalize_transform,
        )
        result = trainig(model, train_data, device, args)
        accuracy, loop_time, forward_backward_time, total_processed_samples_number = result

        throughput = total_processed_samples_number / forward_backward_time
        forward_backward_time /= total_processed_samples_number
    elif args.inference:
        test_data = ChestXrayData(
            data_dir=images_dir,
            image_list_file=test_image_list_file,
            transform=normalize_transform,
        )
        result = inference(model, test_data, device, args)
        accuracy, loop_time, forward_time, total_processed_samples_number = result

        throughput = total_processed_samples_number / forward_time
        forward_time /= total_processed_samples_number

    if args.check_accuracy:
        print(f"Accuracy: {accuracy:.3f}")
    print(f"Total processed images number: {total_processed_samples_number}")
    print(f"Loop time: {loop_time:.3f} (s)")
    if forward_backward_time is not None:
        print(f"Average (forward + backward) time: {forward_backward_time*100:.3f} (ms)")
    if forward_time is not None:
        print(f"Average forward time: {forward_time*100:.3f} (ms)")
    print(f"Throughput: {throughput:.3f} (img/s)")

    #model = htcore.hpu_set_inference_env(model) #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env_world_size', default='WORLD_SIZE', type=str)
    # parser.add_argument('--env_rank', default='LOCAL_RANK', type=str)
    parser.add_argument('--training', action='store_true', default=False)
    parser.add_argument('--inference', action='store_true', default=False)
    parser.add_argument('--check_accuracy', action='store_true', default=False)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--iterations', default=None, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--hpu', action='store_true', default=False)
    parser.add_argument('--use_lazy_mode', action='store_true', default=False)

    args = parser.parse_args()

    main(args)
