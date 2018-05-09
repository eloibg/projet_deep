import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.autograd import Variable
from deeplib.history import History
from deeplib.data import train_valid_loaders


def validate(model, val_loader, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    criterion = nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output, hidden = model(inputs, None)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).data[0])
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())
    #print("Precision: ")
    #print(precision_score(true, pred))
    #print("Recall: ")
    #print(recall_score(true, pred))
    return confusion_matrix(true, pred), accuracy_score(true, pred), sum(val_loss) / len(val_loss)


def train(model, dataset, n_epoch, batch_size, learning_rate, use_gpu=False):
    history = History()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_loader, val_loader = train_valid_loaders(dataset, batch_size=batch_size)

    for i in range(n_epoch):
        model.train()
        for j, batch in enumerate(train_loader):

            inputs, targets = batch
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            inputs = Variable(inputs)
            targets = Variable(targets)
            optimizer.zero_grad()
            output, hidden = model(inputs, None)

            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

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
    return history