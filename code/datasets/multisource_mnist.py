"""
This script contains the multi-source MNIST data set (MSMNIST). 
The data set builds up on the MNIST data set [http://yann.lecun.com/exdb/mnist/] 
and utilizes the PyTorch MNIST data set [https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html].
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MSMNIST(Dataset):
  def __init__(self, 
               train=True,
               label_summation=False,
               split_image=True, 
               frac_data_noise1=0,
               frac_data_noise2=0,               
               frac_label_noise1=0, 
               frac_label_noise2=0, 
               shuffle1=False,
               shuffle2=False
               ):
    """
    Initializes object of class MSMNIST

    Parameters:
        train                 bool      Training data set
        label_summation       bool      Label summation case
        split_image           bool      Split images vertically
        frac_data_noise1      float     Fraction of images in data source 1 with data noise [0,1]
        frac_data_noise2      float     Fraction of images in data source 1 with data noise [0,1]
        frac_label_noise1     float     Fraction of images in data source 1 with label noise [0,1]
        frac_label_noise2     float     Fraction of images in data source 1 with label noise [0,1]
        shuffle1              bool      Shuffle pixels of data source 1
        shuffle2              bool      Shuffle pixels of data source 2

    Output:
        Instance of class MSMNIST
    """

    self.split_image = split_image
    self.train = train

    self.label_summation = label_summation
    
    self.data_noise1 = frac_data_noise1
    self.data_noise2 = frac_data_noise2
    self.label_noise1 = frac_label_noise1
    self.label_noise2 = frac_label_noise2

    self.shuffle1 = shuffle1
    self.shuffle2 = shuffle2

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    self.mnist_dataset1 = datasets.MNIST(root='./data', train=train, transform=transform, download=True)
    self.mnist_dataset2 = datasets.MNIST(root='./data', train=train, transform=transform, download=True)

    self.idx1_list = torch.arange(len(self))
    self.idx2_list = torch.arange(len(self))

    self.orig_labels1 = self.mnist_dataset1.targets.clone()
    self.orig_labels2 = self.mnist_dataset2.targets.clone()
    self.is_data_noise1 = torch.zeros_like(self.orig_labels1).bool()
    self.is_data_noise2 = torch.zeros_like(self.orig_labels1).bool()
    self.is_label_noise1 = torch.zeros_like(self.orig_labels1).bool()
    self.is_label_noise2 = torch.zeros_like(self.orig_labels1).bool()

    # split images
    if split_image:
      self.mnist_dataset1.data = self.mnist_dataset1.data[:,:,:14]
      self.mnist_dataset2.data = self.mnist_dataset2.data[:,:,14:]

    # shuffle (all) images
    if self.shuffle1:
      j,k = self.mnist_dataset1[0].shape
      for img_id in range(len(self.mnist_dataset1)):
        self.mnist_dataset1.data[img_id] = self.mnist_dataset1.data[img_id].reshape([-1])[torch.randperm(j*k)].reshape([j,k])

    if self.shuffle2:
      j,k = self.mnist_dataset2[0].shape
      for img_id in range(len(self.mnist_dataset2)):
        self.mnist_dataset2.data[img_id] = self.mnist_dataset2.data[img_id].reshape([-1])[torch.randperm(j*k)].reshape([j,k])

    # apply data noise (to fraction of samples)
    if self.data_noise1 > 0:
      j,k = self.mnist_dataset1[0].shape
      noisy_ids1 = torch.randperm(len(self.mnist_dataset1))[:int(self.data_noise1*len(self.mnist_dataset1))]
      for img_id in noisy_ids1:
        self.mnist_dataset1.data[img_id] = torch.randint_like(self.mnist_dataset1.data[img_id], 256)
        self.data_noise1[img_id] = True

    if self.data_noise2 > 0:
      j,k = self.mnist_dataset2[0].shape
      noisy_ids2 = torch.randperm(len(self.mnist_dataset2))[:int(self.data_noise2*len(self.mnist_dataset2))]
      for img_id in noisy_ids2:    
        self.mnist_dataset2.data[img_id] = torch.randint_like(self.mnist_dataset2.data[img_id], 256)
        self.data_noise2[img_id] = True

    # apply label noise (to fraction of samples)
    if self.label_noise1 > 0:
      noisy_ids1 = torch.randperm(len(self.mnist_dataset1))[:int(self.label_noise1*len(self.mnist_dataset1))]
      shuffled_labels_idx = torch.randperm(len(noisy_ids1))

      for i in range(len(noisy_ids1)):
        old_id = noisy_ids1[i]
        new_id = noisy_ids1[shuffled_labels_idx[i]]
        if old_id == new_id and i+1 < len(shuffled_labels_idx):
          new_id = noisy_ids1[shuffled_labels_idx[i+1]]
          shuffled_labels_idx[i+1] = shuffled_labels_idx[i] 
        
        self.idx1_list[old_id] = new_id
        self.label_noise1[old_id] = True

    if self.label_noise2 > 0:
      noisy_ids2 = torch.randperm(len(self.mnist_dataset2))[:int(self.label_noise2*len(self.mnist_dataset2))]
      shuffled_labels_idx = torch.randperm(len(noisy_ids2))

      for i in range(len(noisy_ids2)):
        old_id = noisy_ids2[i]
        new_id = noisy_ids2[shuffled_labels_idx[i]]
        if old_id == new_id and i+1 < len(shuffled_labels_idx):
          new_id = noisy_ids2[shuffled_labels_idx[i+1]]
          shuffled_labels_idx[i+1] = shuffled_labels_idx[i] 
        
        self.idx2_list[old_id] = new_id
        self.label_noise2[old_id] = True

  def __getitem__(self, idx):
    """
    Returns a dictionary containing the images and further information about the sample with the gvien id. 

    Parameters:
      idx       int       Id of the sample to be returned (random sample for "label_summation" case)

    Output:

      Dictionary with the following entries:
        img1         : torch.Tensor       image for data source 1  
        img2         : torch.Tensor       image for data source 2
        label        : torch.Tensor       label

        orig_id1     : int                original id of data source 1 image
        orig_id2     : int                original id of data source 2 image
        orig_label1  : torch.Tensor       original label of data source 1
        orig_label2  : torch.Tensor       original label of data source 2
        data_noise1  : bool               data source 1 affected by data noise
        data_noise2  : bool               data source 2 affected by data noise
        label_noise1 : bool               data source 1 affected by label noise
        label_noise2 : bool               data source 2 affected by label noise
    """

    if self.label_summation and self.train:
      idx1 = torch.randint([1,])[0]
      idx2 = torch.randint([1,])[0]    
    else:
      idx1 = self.idx1_list[idx]
      idx2 = self.idx2_list[idx]

    img1, label1 = self.mnist_dataset1[idx1]
    img2, label2 = self.mnist_dataset2[idx2]

    data_noise1 = self.is_data_noise1[idx1]
    data_noise2 = self.is_data_noise2[idx2]

    label_noise1 = self.is_label_noise1[idx1]
    label_noise2 = self.is_label_noise2[idx2]

    orig_label1 = self.orig_labels1[idx1]
    orig_label2 = self.orig_labels2[idx2]

    if self.label_summation:
      label = (label1 + label2) % 10
    else:
      assert label1 == label2
      label = label1

    return {"img1": img1, 
            "img2": img2, 
            "label": label, 

            "orig_id1": idx1, 
            "orig_id2": idx2, 
            "orig_label1": orig_label1,
            "orig_label2": orig_label2,
            "data_noise1": data_noise1,
            "data_noise2": data_noise2,
            "label_noise1": label_noise1,
            "label_noise2": label_noise2}

  def __len__(self):
    """
    Returns the number of samples in this data set. 
    """
    assert len(self.mnist_dataset1) == len(self.mnist_dataset2)
    return len(self.mnist_dataset1)
  
  def plot_imgs(self, axs, num_examples, fontsize=12):
    plot_idx = torch.randperm(len(self))

    for row_id in range(num_examples[0]):
      for col_id in range(num_examples[1]):

        sample = self[plot_idx[row_id*num_examples[1]+col_id]]

        img = torch.cat([sample["img1"], torch.ones([1,28,5]), sample["img2"]], -1)[0]
        axs[row_id,col_id].matshow(img, cmap="gray")
        axs[row_id,col_id].axis("off")
        axs[row_id,col_id].set_title(f"Label {sample['label']}", fontsize=fontsize)
