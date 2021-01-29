import torch
from torch import nn
from model import resnet18, new_model
from utils import *
from tqdm import tqdm
from matplotlib import pyplot as plt

def MSE(output, target):
    loss = torch.mean((output - target)**2)
    return loss

def get_model_acc(model, train_loader, loss_fn, device, batch_limit):
    model.eval()
    end = batch_limit
    correct = 0
    total = 0
    for data, targets in tqdm(train_loader):
        data, targets = data.to(device), targets.to(device)
        preds = model(data)
        loss = loss_fn(preds, targets)
        _, preds = preds.max(1)
        correct += preds.eq(targets).sum()
        total += targets.shape[0]
        end -= 1
        if end == 0:
            end = batch_limit
            break
    acc = correct/total
    return acc

def train(model, new_model, train_loader, test_loader, optimizer, loss_fn, loss_calib, device):
    target_calib = torch.tensor([1.0])
    best_acc = 0.0
    best_calib = 0.0
    best_weight = 0.0
    for i in range(100):
        model.cpu()
        weight = model.fc.weight.data.detach().cpu().clone()
        weight = new_model(weight)
        with torch.no_grad():
            model.fc.weight.data = weight
        model.to(device)
        pred = get_model_acc(model, train_loader, loss_fn, device, batch_limit=1) # batch_limit=len(train_loader)
        print('Acc: ', pred)
        optimizer.zero_grad()
        pred = torch.autograd.Variable(pred.cpu(), requires_grad=True)
        loss = loss_calib(pred, target_calib)
        loss.backward()
        new_model.calib.data += pred.grad*0.01
        print('Calib: ', new_model.calib.data)
        optimizer.step()
        if best_acc <= pred:
            best_acc = pred
            best_calib = new_model.calib.data.detach().cpu().clone()
        if new_model.calib.data <= 0.81:
            break
    print(new_model.calib.data)
    return best_acc, best_calib

def test(model, test_loader, device):
    model.eval()
    correct = 0.0
    total = 0.0
    pbar = tqdm(test_loader)
    for image, target in pbar:
        image, target = image.to(device), target.to(device)
        output = model(image)
        _, preds = output.max(1)
        correct += preds.eq(target).sum()
        total += target.shape[0]
        accuracy = (correct / total).item()
        pbar.set_description('Test accuracy: %f' % accuracy)
    pbar.close()
    return (correct / total).item()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18()
    model.load_state_dict(torch.load('./resnet18_cifar100.pt').state_dict())
    added_weight = torch.load('added_weigths.pt')
    model.fc.requires_grad_ = False
    print(model.fc.requires_grad_)
    orig_weight = model.fc.weight.data.detach().clone()
    with torch.no_grad():
        model.fc.weight.data = added_weight
    model.to(device)
    train_loader, test_loader = get_split_cifar100(task_id=None, start_class=0, end_class=100, batch_size=128,
                                                       shuffle=True)
    test_acc = test(model, test_loader, device)
    print('Before train calibration model accuracy: ', test_acc)
    model_new = new_model()
    print('Calibration model Parameter: ', list(model_new.parameters()))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_new.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    best_acc, best_calib = train(model, model_new, train_loader, test_loader, optimizer, loss_function, MSE, device)
    print(best_acc, best_calib)
    # model = resnet18()
    model.cpu()
    print(model.fc.weight.data[0,0], added_weight[0,0], best_calib)

    model.fc.requires_grad_ = False
    with torch.no_grad():
        model.fc.weight.data = torch.nn.parameter.Parameter(torch.mul(added_weight, torch.tensor((100-i)*0.01))
    model.to(device)
    test_acc = test(model, test_loader, device)
    plt.imshow(model.fc.weight.data.cpu().detach().numpy())
    plt.title('Task by task added Linear calibrated accuracy: %.02f'%test_acc)
    plt.savefig('best_weight.png', dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                metadata=None)
    plt.show()

if __name__ == "__main__":
    main()
