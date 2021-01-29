import argparse
import os

import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from matplotlib import pyplot as plt

from model import resnet18, fc_calib
from utils import *

def main():
    parser = argparse.ArgumentParser(description='cifar-100 dataset continual learning experiment.')

    parser.add_argument('--train_epochs', default=100,
                        help='main training epochs between tasks')
    parser.add_argument('--device', default='cpu',
                        help='execution device: if you have gpu execution would be short ')
    parser.add_argument('--zero_threshold', default=0.3,
                        help='classifier wights near zero elements should be zero -thr<weights<thr')
    parser.add_argument('--test_freq', default=5,
                        help='Apply test on the model default 5')
    parser.add_argument('--batch_size', default=128,
                        help='batch size of training')
    parser.add_argument('--learning_rate', default=0.1,
                        help='learning rate of training')
    parser.add_argument('--milestone', default=[25, 50, 75],
                        help='learning rate of training')
    parser.add_argument('--test_accuracy_goal', default=80.0,
                        help='each task accuracy should be reach this')

    opt = parser.parse_args()
    print(opt.train_epochs)
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18()
    # model.load_state_dict(torch.load('./resnet18_cifar100.pt').state_dict())
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    model.to(opt.device)
    print(model.fc)
    train_loader, test_loader = get_split_cifar100(task_id=None, start_class=0, end_class=100, batch_size=opt.batch_size,
                                                   shuffle=True)
    # test_accuracy = test(model, test_loader, opt)
    # weight = model.fc.weight.data.detach().cpu()
    # plt.imshow(weight)
    # plt.title('Pretrained Linear accuracy: %.02f'%test_accuracy)
    # plt.savefig('100_orig.png', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1,
    #             metadata=None)
    # plt.show()


    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone,
                                                     gamma=0.2)  # learning rate decay
    train(model, train_loader, test_loader, train_scheduler, optimizer, loss_function, mask=None, opt=opt)
    goal_accuracy = test(model, test_loader, opt)
    torch.save(model.state_dict(), './resnet18_cifar100.pt')
    print('Continual learning test accuracy goal: ', goal_accuracy, ' Let\'s begin...')
    print(len(list(model.children())))
    # ct = 0
    # for child in model.children():
    #     ct += 1
    #     if ct < 6:
    #         for param in child.parameters():
    #             param.requires_grad = False
    # added_weigths = torch.zeros(100, 512)
    # added_weigths_cleared = torch.zeros(100, 512)
    # for i in range(1,21):
    #     print("Task No:%i"%i)
    #     model.cpu()
    #     model.fc = nn.Linear(model.fc.in_features, i*5)
    #     model.to(opt.device)
    #     train_loader, test_loader = get_split_cifar100(task_id=i, start_class=None, end_class=None,
    #                                                    batch_size=opt.batch_size, shuffle=True)
    #     optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    #     opt.train_epochs = 2
    #     opt.test_freq = 2
    #     # summary(model, input_size=(3, 32, 32))
    #     train(model, train_loader, test_loader, train_scheduler, optimizer, loss_function, mask=None, opt=opt)
    #     test_accuracy = test(model, test_loader, opt)
    #     weight = model.fc.weight.data.detach().cpu()
    #     added_weigths[(i-1)*5:i*5, :] = weight[(i-1)*5:i*5, :]
    #
    #     model, mask = update_mask(model, opt, old_mask=None)
    #     train(model, train_loader, test_loader, train_scheduler, optimizer, loss_function, mask=mask, opt=opt)
    #     test_accuracy = test(model, test_loader, opt)
    #     weight = model.fc.weight.data.detach().cpu()
    #     added_weigths_cleared[(i-1)*5:i*5, :] = weight[(i-1)*5:i*5, :]
    #     del mask
    # model.cpu()
    # model.fc = nn.Linear(model.fc.in_features, 100)
    # model.fc.weight.data = added_weigths
    # model.to(opt.device)
    # train_loader, test_loader = get_split_cifar100(task_id=None, start_class=0, end_class=100,
    #                                                batch_size=opt.batch_size, shuffle=True)
    # test_accuracy = test(model, test_loader, opt)
    # plt.title('Task by task added Linear accuracy: %.02f'%test_accuracy)
    # plt.imshow(added_weigths)
    #
    # plt.savefig('added_wiegths.png', dpi=None, facecolor='w', edgecolor='w',
    #             orientation='portrait', format=None,
    #             transparent=False, bbox_inches=None, pad_inches=0.1,
    #             metadata=None)
    # plt.show()
    # torch.save(added_weigths, 'added_weigths.pt')
    # # model_calib = fc_calib(model, opt, mode=0)
    # #
    # # model.cpu()
    # # model.fc = nn.Linear(model.fc.in_features, 100)
    # # model.fc.weight.data = added_weigths_cleared
    # # loss_function = MSE()
    # # train_loader, test_loader = get_split_cifar100(task_id=None, start_class=0, end_class=100,
    # #                                                batch_size=opt.batch_size, shuffle=True)
    # # optimizer = torch.optim.SGD(model_calib.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=5e-4)
    # # train_calib(model_calib, train_loader, test_loader, optimizer, loss_function, opt)
    # #
    # # plt.title('Task by task added and Cleared Linear accuracy: %.02f'%test_accuracy)
    # # plt.imshow(added_weigths_cleared)
    # # plt.savefig('added_weigths_cleared.png', dpi=None, facecolor='w', edgecolor='w',
    # #             orientation='portrait', format=None,
    # #             transparent=False, bbox_inches=None, pad_inches=0.1,
    # #             metadata=None)
    # # plt.show()
if __name__ == "__main__":
    main()
