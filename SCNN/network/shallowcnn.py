import torch
import torch.nn as nn


BATCH_NORM_MOMENTUM = 0.01
BATCH_NORM_EPS = 1e-5


class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2, multi_gpu=False, **kwargs):
        self.num_classes = num_classes
        # self._model = self.create(**kwargs)
        self.LossFunction = None
        # if multi_gpu:
        #     self._model = nn.DataParallel(self._model)
        super().__init__()
        # self.conv = nn.Sequential(nn.Conv2d(in_channels=1,
        #                                   out_channels=10,
        #                                   kernel_size=(1, 3),
        #                                   stride=1),
        #                           nn.ReLU()
        #                           )
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=20,
                                          kernel_size=(9, 3),
                                          stride=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=20,
                                            out_channels=20,
                                            kernel_size=(5, 1),
                                            stride=1),
                                  nn.ReLU()
                                  )
        # self.fc = nn.Sequential(nn.Linear(540, 100),
        #                         nn.ReLU(),
        #                         nn.Linear(100, 50),
        #                         nn.ReLU(),
        #                         nn.Linear(50, self.num_classes))
        self.fc = nn.Sequential(nn.Linear(360, 50),
                                nn.ReLU(),
                                nn.Linear(50, self.num_classes))
        self.softmax = nn.Softmax()

    def forward(self, x):
        conv_output = self.conv(x)
        conv_output = conv_output.view(conv_output.size(0), -1)
        # fc_output = self.fc(conv_output)
        # sm_output = self.softmax(fc_output)
        return self.fc(conv_output)

    def predict(self, x):
        conv_output = self.conv(x)
        conv_output = conv_output.view(conv_output.size(0), -1)
        fc_output = self.fc(conv_output)
        return self.softmax(fc_output)

    def create(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        info = "Architecture:\n" + str(self.model)
        return info

    def __repr__(self):
        return self.__str__()

    def train(self, train_data_loader, device, optimizer, criterion):
        for images, labels in train_data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = self.forward(images)
            labels = labels.long()
            labels = labels.squeeze()
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            print(loss.data)

    def load_weights(self, model_saved="logs/model.dat", strict=True):
        self._model.load_state_dict(torch.load(model_saved), strict=strict)

    def test(self, test_data_loader, device):
        predict_label = []
        # truth_labels = []
        for images in test_data_loader:
            images = images.to(device)
            results = torch.max(self.predict(images), dim=1)
            results_label = results.indices
            # labels = labels.long()
            # labels = labels.t()
            # truth_labels.extend(labels)
            predict_label.extend(results_label)
            # tmp = labels == results_label
            # positive += tmp.sum()
        return predict_label

    @property
    def model(self):
        return self._model