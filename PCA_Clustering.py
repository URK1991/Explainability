import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision import datasets, models, transforms
import os
import torch.nn.functional as nnf

#loading the model and setting to eval mode
model= torch.load('__.pth',map_location=torch.device('cpu'))
model.eval()

feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

def get_transforms(split):
    print('getting data transforms')
    data_transforms = {
    'Train': [
       # transforms.Grayscale(),
        transforms.RandomResizedCrop(size=(224, 224),scale=(0.9, 1.0),ratio=(9 / 10, 10 / 9)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(23)], p=0.8),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.125)], p=0.8),
        transforms.ToTensor()
        ],
    'Test': [
       # transforms.Grayscale(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ]
    }
    return transforms.Compose(data_transforms[split])

data_dir = ' '
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),get_transforms(x)) for x in ['Train', 'Test']}
dataloaders = {};
dataloaders['Train'] = torch.utils.data.DataLoader(image_datasets['Train'], batch_size=32,
                                             shuffle=True, num_workers=4)

dataloaders['Test'] = torch.utils.data.DataLoader(image_datasets['Test'], batch_size=32,
                                             shuffle=False, num_workers=4)
dataset_sizes = {x: len(image_datasets[x]) for x in ['Train', 'Test']}        
class_names = image_datasets['Train'].classes
print(class_names)

#we are mainly interested in the test images and the predictions
test_imgs = torch.zeros((0, 3, 224, 224), dtype=torch.float32)
test_predictions = []
test_targets = []
test_embeddings = torch.zeros((0, 512), dtype=torch.float32)

for x,y in dataloaders['Test']:
    embeddings = feature_extractor(x).squeeze() 
    out = model(x)
    _, preds = torch.max(out, 1)
    test_predictions.extend(preds.detach().cpu().tolist())
    test_targets.extend(y.detach().cpu().tolist())
    test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)
    test_imgs = torch.cat((test_imgs, x.detach().cpu()), 0)
  
test_imgs = np.array(test_imgs)
test_embeddings = np.array(test_embeddings)
test_targets = np.array(test_targets)
test_predictions = np.array(test_predictions)

#Twp PCA components are extracted for 2D visualization
pca = PCA(n_components=2)
pca.fit(test_embeddings)
pca_proj = pca.transform(test_embeddings)

cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 4 #four classes

for lab in range(num_categories):
    indices = test_predictions==lab
    ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()

