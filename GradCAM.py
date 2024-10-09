import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torchsummary import summary
import copy

# Define argument class for storing parameters
class Args:
    src_root = ''
    img_size = 224
    batch_size = 1
    num_workers = 0
    pretrained_model_path = ''
    n_imgs = 4

args = Args()

# Model definition
class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()
        self.class_num = class_num

        backbone = torchvision.models.resnet18(pretrained=False)
        backbone_layers = list(backbone.children())
        self.fc_inputs = backbone.fc.in_features

        # Extracting layers from backbone
        self.conv1 = backbone_layers[0]
        self.bn1 = backbone_layers[1]
        self.relu = backbone_layers[2]
        self.maxpool = backbone_layers[3]
        self.layer1 = backbone_layers[4]
        self.layer2 = backbone_layers[5]
        self.layer3 = backbone_layers[6]
        self.layer4 = backbone_layers[7]
        self.avgpool = backbone_layers[8]
        self.fc = nn.Linear(self.fc_inputs, self.class_num)
        self.gradients = None

    def forward(self, x, reg_hook=True):
        # Forward pass through ResNet layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if reg_hook:
            x.register_hook(self.activations_hook)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        # Get activations from last convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def get_data_transforms():
    """Defines transformations for Train and Test sets"""
    return {
        'Train': transforms.Compose([
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(9 / 10, 10 / 9)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(23)], p=0.8),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
            transforms.ToTensor()
        ]),
        'Test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
    }

def load_datasets(data_dir, data_transforms):
    """Loads Train and Test datasets with ImageFolder"""
    return {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ['Train', 'Test']}

def get_dataloaders(image_datasets, batch_size, num_workers):
    """Creates DataLoader for datasets"""
    return {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
            for x in ['Train', 'Test']}

def load_model(path, num_classes):
    """Loads a pretrained model and its weights"""
    model = ResNet(num_classes)
    trained_model = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(trained_model.state_dict())
    return model

def get_grad_cam(net, img):
    """Performs Grad-CAM for a given image and network"""
    net.eval()
    pred = net(img, True)
    _, preds = torch.max(pred, 1)
    
    pred[:, pred.argmax(dim=1)].backward()
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = net.get_activations(img).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)
    return heatmap, preds

def superimpose_heatmap(heatmap, img):
    """Superimposes heatmap on the image"""
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    superimposed_img = torch.Tensor(cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)) * 0.002 + img[0].cpu().permute(1, 2, 0)
    return superimposed_img

def generate_grad_cam_images(model, dataloaders, n_imgs, img_size):
    """Generates Grad-CAM images for a batch of n_imgs"""
    imgs = torch.Tensor(2, n_imgs, 3, img_size, img_size)
    labels = []
    preds = []

    it = iter(dataloaders['Test'])
    for i in range(n_imgs):
        img, label = next(it)
        heatmap, pred = get_grad_cam(model, img)
        imgs[0][i] = img[0]
        imgs[1][i] = superimpose_heatmap(heatmap, img).permute(2, 0, 1)
        labels.append(label.item())
        preds.append(pred.item())

    return imgs, labels, preds

def display_images(imgs, labels, preds, img_size):
    """Displays images and Grad-CAM heatmaps side by side"""
    for i in range(len(labels)):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(f'True label: {labels[i]} - Prediction: {preds[i]}')
        ax1.imshow(imgs[0][i].permute(1, 2, 0), interpolation='nearest')
        ax2.imshow(imgs[1][i].permute(1, 2, 0), interpolation='nearest')

# Main execution starts here
if __name__ == "__main__":
    data_transforms = get_data_transforms()
    data_dir = ' '
    image_datasets = load_datasets(data_dir, data_transforms)
    dataloaders = get_dataloaders(image_datasets, args.batch_size, args.num_workers)

    model = load_model(args.pretrained_model_path, num_classes=4)

    imgs, labels, preds = generate_grad_cam_images(model, dataloaders, args.n_imgs, args.img_size)
    display_images(imgs, labels, preds, args.img_size)
