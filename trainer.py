from dataloader import OneClassDatasetCIFAR10
from torch.utils.data import DataLoader
from model import Net
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

def main():
    DATA_DIR = 'datasets\\CIFAR10'
    EPOCHS = 200
    BATCH_SIZE = 512
    REAL_CLASS = 2
    VAL_EACH = 10

    training_set = OneClassDatasetCIFAR10(DATA_DIR, real_class=REAL_CLASS, train=True)
    val_set = OneClassDatasetCIFAR10(DATA_DIR, real_class=REAL_CLASS, train=False, vis=False)
    train_loader = DataLoader(training_set, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()

    model = Net(4)
    model.double()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch_idx in range(EPOCHS):
        pbar = tqdm(train_loader)
        for image_batch, label_batch in pbar:
            pbar.set_description("Epoch: %s" % str(epoch_idx))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            logits_batch = model(image_batch)
            loss = criterion(logits_batch, label_batch)
            loss.backward()
            optimizer.step()


        if epoch_idx != 0 and epoch_idx % VAL_EACH == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(val_loader)
                for image_batch, label_batch in pbar:
                    pbar.set_description("Epoch: %s" % str(epoch_idx))
                    logits_batch = model(image_batch)
                    _, predicted = torch.max(logits_batch.data, 1)
                    total += label_batch.size(0)
                    correct += (predicted == label_batch).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))


main()