import numpy as np
import torch
import torchvision
from tqdm import tqdm


def get_split_cifar100(task_id, start_class=None, end_class=None, batch_size=32, shuffle=False):
    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    if start_class is None:
        start_class = (task_id - 1) * 5
        end_class = task_id * 5
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
    ])

    train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=transforms)
    test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=transforms)

    # train_size = int(0.8 * len(train))
    # val_size = len(train) - train_size
    # train, val = torch.utils.data.random_split(train, [train_size, val_size])
    # print(len(train), len(val))

    targets_train = torch.tensor(train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

    # targets_val = torch.tensor(val.targets)
    # target_val_idx = ((targets_val >= start_class) & (targets_val < end_class))

    targets_test = torch.tensor(test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.dataset.Subset(train, np.where(target_train_idx == 1)[0]), batch_size=batch_size)

    # val_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.dataset.Subset(val, np.where(target_val_idx == 1)[0]), batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(test, np.where(target_test_idx == 1)[0]),
                                              batch_size=batch_size // 2)

    return train_loader, test_loader


def train(model, train_loader, test_loader, train_scheduler, optimizer, loss_fn, mask, opt):
    for epoch in range(opt.train_epochs):
        model.train()
        correct = 0.0
        total = 0.0
        pbar = tqdm(train_loader)
        for image, target in pbar:
            image, target = image.to(opt.device), target.to(opt.device)
            optimizer.zero_grad()
            output = model(image)
            loss = loss_fn(output, target)
            loss.backward()
            if torch.is_tensor(mask):
                with torch.no_grad():
                    model.fc.weight.grad[mask] = 0.0
            optimizer.step()
            _, preds = output.max(1)
            correct += preds.eq(target).sum()
            total += target.shape[0]
            pbar.set_description(
                'Epoch: %s, Train accuracy: %f, Loss: %f' % (str(epoch), (correct / total).item(), loss.item()))
        pbar.close()
        if epoch % opt.test_freq == 0:
            test(model, test_loader, opt)
        train_scheduler.step()


def test(model, test_loader, opt):
    model.eval()
    correct = 0.0
    total = 0.0
    pbar = tqdm(test_loader)
    for image, target in pbar:
        image, target = image.to(opt.device), target.to(opt.device)
        output = model(image)
        _, preds = output.max(1)
        correct += preds.eq(target).sum()
        total += target.shape[0]
        accuracy = (correct / total).item()
        pbar.set_description('Test accuracy: %f' % accuracy)

    pbar.close()
    return (correct / total).item()


def update_mask(model, opt, old_mask=None):
    model.cpu()
    if not torch.is_tensor(old_mask):
        with torch.no_grad():
            new = model.fc.weight.cpu().numpy()
            print(torch.histc(model.fc.weight.data, bins=20, min=-1.0, max=1.0))
            mask = (new <= (new.max() - opt.zero_threshold)) & (new >= (new.min() + opt.zero_threshold / 3))
            print('New mask thresholding range: [', new.min() + opt.zero_threshold / 3, ',',
                  new.max() - opt.zero_threshold, '], Number of weights released:', mask.sum())
            new[mask] = 0.0
            model.fc.weight.data = torch.tensor(new)
    else:
        h, w = old_mask.shape
        with torch.no_grad():
            new = model.fc.weight.cpu()
            new_task_weight = new[h:, w:]
            old_task_weight = new[:h, :w]
            old_task_trainable = old_task_weight[old_mask]

            # TODO update old mask
            # TODO update new task

    return model.to(opt.device), mask


def add_task(model, task_id, opt):
    model.cpu()
    with torch.no_grad():
        old_weights, old_bias = model.fc.weight.data, model.fc.bias.data
        model.fc = torch.nn.Linear(model.fc.in_features, task_id * 5)
        model.fc.weight.data[:(task_id - 1) * 5, :], model.fc.bias.data[:(task_id - 1) * 5] = old_weights, old_bias
    # after add task you should train until test threshold accuracy
    model.to(opt.device)
    return model

def train_calib(model_calib, train_loader, test_loader, optimizer, loss_fn, opt):
    model_calib.to(opt.device)
    target_calib = torch.autograd.Variable(torch.tensor([1.0]), requires_grad=True).to(opt.device)
    for i in range(10):
        pred_model_accuracy = test(model, test_loader, opt)
        loss = loss_fn(pred_model_accuracy, target_calib)
        print('loss: ', loss.item())
        loss.backward()
        optimizer.step()
