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
        self.densenet121 = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        )
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_features, out_size)

    def forward(self, x):
        x = self.densenet121(x)
        return x


IS_DISTRIBUTED = False
WORLD_SIZE = 1
RANK = 0
LOCAL_RANK = 0
def setup_distributed(world_size):
    global IS_DISTRIBUTED, WORLD_SIZE, RANK, LOCAL_RANK

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12340'

    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
    WORLD_SIZE, RANK, LOCAL_RANK = initialize_distributed_hpu()
    if world_size <= WORLD_SIZE:
        WORLD_SIZE = world_size
    else:
        raise RuntimeError("Number of devices is bigger than actual WORLD_SIZE.")


    #Import the distributed package for HCCL, set the backend to HCCL
    import habana_frameworks.torch.distributed.hccl
    torch.distributed.init_process_group(backend='hccl', rank=RANK, world_size=WORLD_SIZE)

    IS_DISTRIBUTED = True


def destroy_distributed():
    torch.distributed.destroy_process_group()


def inference(model, data, device, args):
    sampler = None
    if IS_DISTRIBUTED:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data)
    test_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
    )

    model.eval()
    iterations = len(test_loader)
    if args.iterations is not None:
        iterations = min(iterations, args.iterations)

    total_processed_samples_number = test_loader.batch_size * iterations

    mean_acc = 0.0
    forward_time = 0
    loop_time_start = 0
    with torch.no_grad():
        if LOCAL_RANK == 0:
            print("Warming up...")
        warmup_iterations = min(len(test_loader), 3)
        for indx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            htcore.mark_step()
            if warmup_iterations == indx:
                break

        loop_time_start = time.time()
        with tqdm(desc=f'Evaluation: ', unit='it', total=iterations) as pbar:
            for indx, (images, labels) in enumerate(test_loader, 1):
                images = images.to(device)
                forward_start_time = time.time()
                outputs = model(images)
                # for perf mesuraments
                htcore.mark_step(sync=True)
                forward_time += time.time() - forward_start_time

                pbar.update()

                # accuracy
                if args.check_accuracy:
                    y_true = labels.cpu()
                    y_pred_org = outputs.cpu()
                    y_pred = (y_pred_org > 0).float()

                    acc = accuracy_score(y_true[:], y_pred[:])
                    mean_acc += acc

                if indx == iterations:
                    break

    mean_acc /= iterations
    loop_time_finish = time.time()

    return mean_acc, loop_time_start, loop_time_finish, forward_time, total_processed_samples_number


def trainig(model, data, device, args):
    # sampler option is mutually exclusive with shuffle
    sampler = None
    if IS_DISTRIBUTED:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data)
    train_loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
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

    if args.use_hpu_graph:
        image, lable = train_loader.dataset[0]
        batch_image_shape = [train_loader.batch_size] + list(image.shape)
        batch_lable_shape = [train_loader.batch_size] + list(lable.shape)
        static_input = torch.randn(batch_image_shape, device='hpu')
        static_target = torch.rand(batch_lable_shape, device='hpu')

        # warming up
        if LOCAL_RANK == 0:
            print("Warming up...")
        s = htcore.hpu.Stream()
        s.wait_stream(htcore.hpu.current_stream())
        with htcore.hpu.stream(s):
            for _ in range(5):
                optimizer.zero_grad(set_to_none=True)
                y_pred = model(static_input)
                loss = criterion(y_pred, static_target)
                loss.backward()
                optimizer.step()
        htcore.hpu.current_stream().wait_stream(s)

        # capturing
        g = htcore.hpu.HPUGraph()
        optimizer.zero_grad(set_to_none=True)
        with htcore.hpu.graph(g):
            static_y_pred = model(static_input)
            static_loss = criterion(static_y_pred, static_target)
            static_loss.backward()
            optimizer.step()

    forward_backward_time = 0
    loop_time_start = time.time()
    for epoch in range(epochs):
        if IS_DISTRIBUTED:
            sampler.set_epoch(epoch)

        mean_loss = 0.0
        mean_acc = 0.0
        final_accuracy = 0
        model.train()
        tqdm_total = len(train_loader)
        if epoch == epochs - 1 and iterations % len(train_loader) != 0:
            tqdm_total = iterations % len(train_loader)

        with tqdm(desc=f'Epoch {epoch + 1}: ', unit='it', total=tqdm_total) as pbar:
            for index, (images, labels) in enumerate(train_loader, 1):
                if args.use_hpu_graph:
                    static_input.copy_(images)
                    static_target.copy_(labels)

                    forward_start_time = time.time()
                    g.replay()
                    forward_backward_time += time.time() - forward_start_time
                else:
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
                    htcore.mark_step(sync=True)
                    forward_backward_time += time.time() - forward_start_time

                pbar.update()

                # mid-training loss and accuracy
                if args.check_accuracy:
                    y_true = labels.detach().cpu()
                    if args.use_hpu_graph:
                        y_pred_org = static_y_pred.detach().cpu()
                    else:
                        y_pred_org = outputs.detach().cpu()
                    y_pred = (y_pred_org > 0).float()
                    loss_val = loss.item()

                    mean_loss += loss_val
                    acc = accuracy_score(y_true[:], y_pred[:])
                    mean_acc += acc
                    final_accuracy += acc

                    if LOCAL_RANK == 0 and index % 10 == 0:
                        print()
                        torch.set_printoptions(precision=2)
                        print('\repoch %3d/%3d batch %5d/%5d' % \
                            (epoch+1, epochs, index, len(train_loader)),
                            end=''
                        )
                        mean_loss /= index
                        mean_acc /= index
                        print(f' Train loss {mean_loss}', end='')
                        print(f' Accuracy: {mean_acc}')

                if epoch == epochs - 1 and index == iterations % len(train_loader):
                    break 

        if LOCAL_RANK == 0 and args.output_dir is not None:
            model_dir = args.output_dir + '/model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            models_path = model_dir + f'/checkpoint_{epoch + 1}.pth'
            torch.save(model.state_dict(), models_path)
            print(f'\nModel has been saved: {models_path}')

        final_accuracy /= tqdm_total

    loop_time_finish = time.time()

    return final_accuracy, loop_time_start, loop_time_finish, forward_backward_time, total_processed_samples_number


def main(args):
    if args.devices > 1:
        setup_distributed(args.devices)

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
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

    model = model.to(device)

    if IS_DISTRIBUTED:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, bucket_cap_mb=100, broadcast_buffers=False, gradient_as_bucket_view=True)

    accuracy = None
    loop_time_start = None
    loop_time_finish = None
    avg_forward_backward_time = None
    avg_forward_time = None
    total_forward_backward_time = None
    total_forward_time = None
    if args.training:
        # train data set object
        train_data = ChestXrayData(
            data_dir=images_dir,
            image_list_file=train_image_list_file,
            transform=normalize_transform,
        )
        result = trainig(model, train_data, device, args)
        accuracy, loop_time_start, loop_time_finish, total_forward_backward_time, total_processed_samples_number = result

        avg_forward_backward_time = total_forward_backward_time / total_processed_samples_number
    if args.inference:
        test_data = ChestXrayData(
            data_dir=images_dir,
            image_list_file=test_image_list_file,
            transform=normalize_transform,
        )
        if args.use_hpu_graph:
            if IS_DISTRIBUTED:
                raise RuntimeError("`--use_hpu_graph` can't be used with multiple devices.")
            model = wrap_in_hpu_graph(model,
                                      asynchronous=True,
                                      disable_tensor_cache=True
                                )

        result = inference(model, test_data, device, args)
        accuracy, loop_time_start, loop_time_finish, total_forward_time, total_processed_samples_number = result

        avg_forward_time = total_forward_time / total_processed_samples_number

    loop_time = loop_time_finish - loop_time_start
    if IS_DISTRIBUTED:
        # reduce metrics
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        if accuracy is not None:
            accuracy = comm.reduce(accuracy)
            if LOCAL_RANK == 0:
                accuracy /= WORLD_SIZE
        
        if avg_forward_time is not None:
            avg_forward_time = comm.reduce(avg_forward_time)
            if LOCAL_RANK == 0:
                avg_forward_time /= WORLD_SIZE
        
        if avg_forward_backward_time is not None:
            avg_forward_backward_time = comm.reduce(avg_forward_backward_time)
            if LOCAL_RANK == 0:
                avg_forward_backward_time /= WORLD_SIZE
        
        if total_forward_time is not None:
            total_forward_time = comm.reduce(total_forward_time)

        if total_forward_backward_time is not None:
            total_forward_backward_time = comm.reduce(total_forward_backward_time)

        total_processed_samples_number = comm.reduce(total_processed_samples_number)

        loop_time_finish = comm.reduce(loop_time_finish, MPI.MAX)
        loop_time_start = comm.reduce(loop_time_start, MPI.MIN)
        if LOCAL_RANK == 0:
            loop_time = loop_time_finish - loop_time_start

    if LOCAL_RANK == 0:
        if args.check_accuracy:
            print(f"Accuracy: {accuracy:.3f}")
        print(f"Total processed images number: {total_processed_samples_number}")
        print(f"Total loop time: {loop_time:.3f} (s)")
        print(f"\tthroughput: {(total_processed_samples_number / loop_time):.3f} (img/s)")
        if avg_forward_backward_time is not None:
            print(f"Average (forward + backward) time: {avg_forward_backward_time*100:.3f} (ms)")
            print(f"\tthroughput: {total_processed_samples_number / total_forward_backward_time:.3f} (img/s)")
        if avg_forward_time is not None:
            print(f"Average forward time: {avg_forward_time*100:.3f} (ms)")
            print(f"\tthroughput: {total_processed_samples_number / total_forward_time:.3f} (img/s)")

    if IS_DISTRIBUTED:
        destroy_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='1', type=int)
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
    parser.add_argument('--use_hpu_graph', action='store_true', default=False)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        if IS_DISTRIBUTED:
            destroy_distributed()
        raise e
