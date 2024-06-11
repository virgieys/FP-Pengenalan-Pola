import cv2 as cv
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import tim as timm

class Data(Dataset):
    def __init__(self, f="./dataset/"):
        self.features, self.labels = [], []

        src_dir = './dataset'
        folders = ['bleached_corals', 'healthy_corals']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(folders)}

        # Iterasi melalui setiap folder dan hitung jumlah gambar
        for folder in folders:
            # Tentukan direktori folder
            folder_path = os.path.join(src_dir, folder)

            # Iterasi melalui file di direktori folder
            for file_name in os.listdir(folder_path):
                file = os.path.join(folder_path, file_name)
                img = cv.imread(file)
                # Periksa apakah gambar berhasil dibaca
                if img is None:
                    print(f"Error: Unable to read image from path: {file}")
                    return None
                img = cv.resize(img, (150, 150))

                img = (img - np.min(img)) / np.ptp(img)

                self.features.append(img)
                self.labels.append(self.class_to_idx[folder])

    def __getitem__(self, item):
        feature, label = self.features[item], self.labels[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.features)

class ConvNet(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.num_features
        self.model.fc = nn.Linear(n_features, n_class)

    def forward(self, x):
        return self.model(x)


def main():
    BATCH_SIZE = 8
    EPOCHS = 10

    # Membuat objek dataset
    dataset = Data()

    # Membagi dataset menjadi set pelatihan dan pengujian
    train_set, test_set = train_test_split(dataset, test_size=0.3, random_state=42)

    # Membuat DataLoader untuk set pelatihan dan pengujian
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Membuat model, optimizer, dan loss function
    model = ConvNet(model_arch="resnet18", n_class=2)  # Ubah dengan arsitektur model yang Anda inginkan
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Iterasi untuk setiap epoch
    for epoch in range(EPOCHS):
        model.train()  # Memastikan model dalam mode pelatihan
        running_loss = 0.0
        correct = 0
        total = 0

        # Melatih model menggunakan data pelatihan
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            # Menukar dimensi gambar
            inputs = inputs.permute(0, 3, 1, 2)  # Ubah dari [batch_size, height, width, channels] menjadi [batch_size, channels, height, width]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:  # Tampilkan setiap 100 batch
                print('[%d, %5d] loss: %.3f, accuracy: %.2f%%' % (epoch + 1, i + 1, running_loss / 100, 100 * correct / total))
                running_loss = 0.0
                correct = 0
                total = 0

        # Menguji model pada dataset uji setiap epoch
        model.eval()  # Memastikan model dalam mode evaluasi
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.permute(0, 3, 1, 2)  # Menukar dimensi gambar
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        print('Epoch %d, Accuracy of the network on the test images: %.2f%%' % (epoch + 1, 100 * test_correct / test_total))

    print('Finished Training')

if __name__ == "__main__":
    main()
