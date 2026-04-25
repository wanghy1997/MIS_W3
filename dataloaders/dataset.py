import torch
import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import h5py
from dataloaders.transforms import RandomBrightnessContrast, RandomGaussianNoise


class BTCV_ws(Dataset):
    """ BTCV Dataset """
    def __init__(self, image_list, base_dir=None, patch_size=None):
        self._base_dir = base_dir
        self.image_list = image_list

        self.spatial_transform = RandomCrop(patch_size)

        self.randcolor = RandomBrightnessContrast(
            brightness_limit=0.5, contrast_limit=0.5, prob=0.8
        )
        self.randblur = RandomGaussianNoise(
            sigma=[0.1, 1.0], apply_prob=0.2
        )

        print("Total {} samples for training".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = self._base_dir + '/btcv_h5/{}.h5'.format(image_name)

        with h5py.File(image_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}

        # 1. 先做共享空间增强，保证 image 和 label 对齐
        sample = self.spatial_transform(sample)
        image, label = sample['image'], sample['label']

        # 2. weak image 直接使用裁剪后的 image
        image_weak = image.copy()

        # 3. strong image 在 weak 的基础上做强增强
        image_strong = image.copy()
        image_strong = self.randblur(self.randcolor(image_strong))

        # 4. 转 tensor
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8)).long()

        sample = {
            'image': image_weak,          # 常规增强后的 image（这里就是裁剪后的）
            'label': label,               # 对应 label
            'strong_aug': image_strong, # strong 后的 image
        }

        return sample


class MMWHS(Dataset):
    """ 
    适配 TwoStreamBatchSampler 的心脏数据集加载器 
    逻辑：将有标签和无标签数据按顺序合并，由 Sampler 控制读取顺序
    """
    def __init__(self, full_list, base_dir='', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.full_list = full_list # 这里的 full_list 是 labeled_list + unlabeled_list
        
        self.all_images = []
        self.all_labels = []

        # print(f"Starting to load {len(full_list)} samples into memory...")
        
        for name in self.full_list:
            # 这里的 name 假设是 '1001', '1002' 等数字字符串
            image_path = os.path.join(self._base_dir, 'npy', f'ct_train_{name}_image.npy')
            label_path = os.path.join(self._base_dir, 'npy', f'ct_train_{name}_label.npy')
            
            if not os.path.exists(image_path):
                print(f"Error: {image_path} not found!")
                continue
                
            # 加载数据 (保持 0-1 float32)
            image = np.load(image_path)
            # 加载标签 (uint8)
            label = np.load(label_path).astype(np.uint8)
            
            # 预处理维度: (D, H, W) -> (1, D, H, W)
            # 注意：如果你的 RandomCrop 不支持带通道的数据，请检查 transform 逻辑
            # image = image[np.newaxis, ...]
            
            self.all_images.append(image)
            self.all_labels.append(label)

        print(f"Successfully loaded {len(self.all_images)} samples!")

    def __len__(self):
        # 必须返回列表的总长度，Sampler 会基于这个长度生成索引
        return len(self.all_images)

    def __getitem__(self, idx):
        # 这里的 idx 由 TwoStreamBatchSampler 提供
        image = self.all_images[idx]
        label = self.all_labels[idx]
        
        sample = {
            'image': image, 
            'label': label.astype(np.int64) # 转换成 Long 供 Loss 函数使用
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    

    
class FLARE(Dataset):
    """ AMOS Dataset """
    def __init__(self, image_list, base_dir='', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list

        self.images = []
        self.laebls = []
        for i in range(len(self.image_list)):
            image_name = self.image_list[i] 
            image_path = self._base_dir +  '/npy/' + '{}_image.npy'.format(image_name)
            label_path = self._base_dir +  '/npy/' + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
    
            self.images.append(image)
            self.laebls.append(label)
            print('Loading ', image_path)
            
        print("Loaded! Total {} samples for training".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):        
        image = self.images[idx]
        label = self.laebls[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample


class FLARE_fast(Dataset):
    """ FLARE Dataset """
    def __init__(self, labeled_list, unlabeled_list, base_dir='', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.labeled_list = labeled_list
        self.unlabeled_list = unlabeled_list
        
        self.images_l = []
        self.laebls_l = []
        for i in range(len(self.labeled_list)):
            image_name = self.labeled_list[i] 
            image_path = self._base_dir + '/npy/' + '{}_image.npy'.format(image_name)
            label_path = self._base_dir + '/npy/' + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
            self.images_l.append(image)
            self.laebls_l.append(label)
            # print('Loading {:2d}-th labeled sample from {}'.format(i, image_path))

        self.images_u = []
        self.laebls_u = []
        for i in range(len(self.unlabeled_list)):
            image_name = self.unlabeled_list[i] 
            image_path = self._base_dir +  '/npy/' + '{}_image.npy'.format(image_name)
            label_path = self._base_dir +  '/npy/' + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
            self.images_u.append(image)
            self.laebls_u.append(label)
            # print('Loading {:2d}-th unlabeled sample from {}'.format(i, image_path))
            
        print("Loaded! Total {} samples for training".format(len(labeled_list + unlabeled_list)))
        print(len(self.images_l), len(self.laebls_l), len(self.images_u), len(self.laebls_u))


    def __len__(self):
        return (len(self.unlabeled_list)*4)

    def __getitem__(self, idx):
        if idx < len(self.unlabeled_list)*2: # labeled data
            idx = idx % len(self.labeled_list)
            image = self.images_l[idx]
            label = self.laebls_l[idx]
        else:                              # unlabeled data
            idx = idx % len(self.unlabeled_list)
            image = self.images_u[idx]
            label = self.laebls_u[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class AMOS(Dataset):
    """ AMOS Dataset """
    def __init__(self, image_list, base_dir='', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list

        self.images = []
        self.laebls = []
        for i in range(len(self.image_list)):
            image_name = self.image_list[i] 
            image_path = self._base_dir +  '/npy/' + '{}_image.npy'.format(image_name)
            label_path = self._base_dir +  '/npy/' + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
    
            self.images.append(image)
            self.laebls.append(label)
            print('Loading ', image_path)
            
        print("Loaded! Total {} samples for training".format(len(self.image_list)))



    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):        
        image = self.images[idx]
        label = self.laebls[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)        
        return sample



class AMOS_fast(Dataset):
    """ AMOS Dataset """
    def __init__(self, labeled_list, unlabeled_list, base_dir='', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.labeled_list = labeled_list
        self.unlabeled_list = unlabeled_list
        
        self.images_l = []
        self.laebls_l = []
        for i in range(len(self.labeled_list)):
            image_name = self.labeled_list[i] 
            image_path = self._base_dir + '/npy/' + '{}_image.npy'.format(image_name)
            label_path = self._base_dir + '/npy/' + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
            self.images_l.append(image)
            self.laebls_l.append(label)
            # print('Loading {:2d}-th labeled sample from {}'.format(i, image_path))

        self.images_u = []
        self.laebls_u = []
        for i in range(len(self.unlabeled_list)):
            image_name = self.unlabeled_list[i] 
            image_path = self._base_dir +  '/npy/' + '{}_image.npy'.format(image_name)
            label_path = self._base_dir +  '/npy/' + '{}_label.npy'.format(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            image = image.clip(min=-125, max=275)
            image = (image + 125) / 400
            self.images_u.append(image)
            self.laebls_u.append(label)
            # print('Loading {:2d}-th unlabeled sample from {}'.format(i, image_path))
            
        print("Loaded! Total {} samples for training".format(len(labeled_list + unlabeled_list)))
        print(len(self.images_l), len(self.laebls_l), len(self.images_u), len(self.laebls_u))


    def __len__(self):
        return (len(self.unlabeled_list)*4)

    def __getitem__(self, idx):
        if idx < len(self.unlabeled_list)*2: # labeled data
            idx = idx % len(self.labeled_list)
            image = self.images_l[idx]
            label = self.laebls_l[idx]
        else:                              # unlabeled data
            idx = idx % len(self.unlabeled_list)
            image = self.images_u[idx]
            label = self.laebls_u[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class BTCV(Dataset):
    """ Synapse Dataset """
    def __init__(self, image_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = image_list

        print("Total {} samples for training".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # ex: self._base_dir: '../data/MACT_h5'
        image_path = self._base_dir + '/btcv_h5/{}.h5'.format(image_name)
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MACT(Dataset):
    """ Multi-organ Abdominal CT Reference Standard Segmentations Dataset """
    def __init__(self, image_list, base_dir=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.image_list = ['{:0>4}'.format(i + 1) for i in image_list]

        print("Total {} samples for training".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # ex: self._base_dir: '../data/MACT_h5'
        image_path = self._base_dir + '/{}.h5'.format(image_name)
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

        
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, is_2d=False):
        self.is_2d = is_2d

    def __call__(self, sample):
        # image, label: For AHNet 2D to 3D,
        # 3D: WxHxD -> 1xWxHxD, 96x96x96 -> 1x96x96x96
        # 2D: WxHxD -> CxWxh, 224x224x3 -> 3x224x224
        image, label = sample['image'], sample['label']

        if self.is_2d:
            image = image.transpose(2, 0, 1).astype(np.float32)
            label = label.transpose(2, 0, 1)[1, :, :]
        else:
            # image = image.transpose(1, 0, 2)
            # label = label.transpose(1, 0, 2)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class Synapse_fast_npy(Dataset):
    """ Synapse Dataset """

    def __init__(self, labeled_list, unlabeled_list, base_dir='', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.labeled_list = labeled_list
        self.unlabeled_list = unlabeled_list
        print('self._base_dir', self._base_dir)
        self.images_l = []
        self.laebls_l = []
        for i in range(len(self.labeled_list)):
            image_name = self.labeled_list[i]
            print('image_name', image_name)
            image_path = self._base_dir + '/npy/{}_image.npy'.format(image_name)  # 0001_image.npy
            label_path = self._base_dir + '/npy/{}_label.npy'.format(image_name)
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                raise ValueError(image_name)
            image = np.load(image_path)
            label = np.load(label_path)

            self.images_l.append(image)
            self.laebls_l.append(label)
            print('Loading {:2d}-th labeled sample from {}'.format(i, image_path))

        self.images_u = []
        self.laebls_u = []
        for i in range(len(self.unlabeled_list)):
            image_name = self.unlabeled_list[i]
            image_path = self._base_dir + '/npy/{}_image.npy'.format(image_name)  # 0001_image.npy
            label_path = self._base_dir + '/npy/{}_label.npy'.format(image_name)
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                raise ValueError(image_name)
            image = np.load(image_path)
            label = np.load(label_path)
            self.images_u.append(image)
            self.laebls_u.append(label)
            print('Loading {:2d}-th unlabeled sample from {}'.format(i, image_path))

        print("Loaded! Total {} samples for training".format(len(labeled_list + unlabeled_list)))
        print(len(self.images_l), len(self.laebls_l), len(self.images_u), len(self.laebls_u))

    def __len__(self):
        return (len(self.unlabeled_list) * 4)

    def __getitem__(self, idx):
        if idx < len(self.unlabeled_list) * 2:  # labeled data
            idx = idx % len(self.labeled_list)
            image = self.images_l[idx]
            label = self.laebls_l[idx]
        else:  # unlabeled data
            idx = idx % len(self.unlabeled_list)
            image = self.images_u[idx]
            label = self.laebls_u[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
