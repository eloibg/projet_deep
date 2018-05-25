import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable
from deeplib.history import History
from deeplib.data import train_valid_loaders
import time


def validate(model, val_loader, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):
        (inputs, targets), inputs_len = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output = model(inputs, inputs_len)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).data[0])
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())
    return confusion_matrix(true, pred), accuracy_score(true, pred), sum(val_loss) / len(val_loss)


def train(model, dataset, n_epoch, batch_size, learning_rate, use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)

    train_loader, val_loader = train_valid_loaders(dataset, batch_size=batch_size)
    for i in range(n_epoch):
        scheduler.step()
        t = time.time()
        model.train()
        for j, batch in enumerate(train_loader):
            (inputs, targets), inputs_len = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            inputs = Variable(inputs)
            targets = Variable(targets)
            optimizer.zero_grad()
            output = model(inputs, inputs_len)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

        print('Time: ' + str(time.time()-t))
        train_cm, train_acc, train_loss = validate(model, train_loader, use_gpu)
        val_cm, val_acc, val_loss = validate(model, val_loader, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.4f} - Val loss: {:.4f}'.format(i,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
        print('Train confusion matrix')
        print(train_cm)
        print('Valid confusion matrix')
        print(val_cm)
        with open('resultats.txt', 'a') as f:
            f.write('Epoch' + str(i))
            f.write('\n')
            f.write('Train confusion matrix')
            f.write('\n')
            f.write(str(train_cm))
            f.write('\n')
            f.write('Valid confusion matrix')
            f.write('\n')
            f.write(str(val_cm))
            f.write('\n')
    return history