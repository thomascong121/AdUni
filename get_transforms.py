from torchvision import transforms, datasets
from util import GaussianBlur
from randaugment import rand_augment_transform
from autoaug import ImageNetPolicy, CIFAR10Policy, Cutout
from PIL import Image
import numpy as np
import cv2


def get_transform(opt):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if opt.aug_plus:
        augmentation = [
            transforms.RandomResizedCrop(opt.size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(opt.size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_cifar = [
        transforms.RandomCrop(opt.size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize,
    ]

    augmentation_imagenet = [
        transforms.RandomResizedCrop(opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        normalize
    ]

    augmentation_regular_sim = [
        transforms.RandomResizedCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_regular = [
        transforms.RandomResizedCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ]
    
    augmentation_sim_noscale = [
        transforms.RandomResizedCrop(size=opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
        ]),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_kaggle = [
        transforms.Resize((opt.size, opt.size)),
#         transforms.RandomResizedCrop(opt.size),
        transforms.RandomAffine(
            degrees=(-180, 180),
            scale=(0.8, 1.2),
            shear=(0.8, 1.2),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.2),
                               contrast=(0.2),
                               hue=(0.1),
                               saturation=(0.2),
                               ),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_sim02 = [
        transforms.RandomResizedCrop(opt.size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(opt.size * 0.45),
                     img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(opt.size, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        # transforms.RandomResizedCrop(opt.size, scale=(0.08, 1.)),
        transforms.Resize((opt.size, opt.size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    if opt.aug == 'regular_regular':  # 73.06
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif opt.aug == 'mocov2_mocov2':  # not work
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif opt.aug == 'imagnet_sim':
        transform_train = [transforms.Compose(augmentation_imagenet), transforms.Compose(augmentation_regular_sim)]
    elif opt.aug == 'sim_sim':
        transform_train = [transforms.Compose(augmentation_sim_noscale), transforms.Compose(augmentation_sim_noscale)]
    elif opt.aug == 'cifar':
        transform_train = [transforms.Compose(augmentation_cifar), transforms.Compose(augmentation_sim_cifar)]
    elif opt.aug == 'kaggle':
        transform_train = [transforms.Compose(augmentation_kaggle), transforms.Compose(augmentation_kaggle)]
    elif opt.aug == 'randcls_sim':  # 79.50
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim_cifar)]
    elif opt.aug == 'randclsstack_sim':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim_cifar)]
    elif opt.aug == 'randclsstack_sim02':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim02)]
    else:
        raise Exception("Augmentation type not specified")
    print('===> Use Augmentation for training: ', opt.aug)

    return transform_train


class crop_image_from_gray(object):
    def __call__(self, pic, tol=7):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        img = np.asarray(pic)
        if img.ndim == 2:
            mask = img > tol
            return Image.fromarray(img[np.ix_(mask.any(1), mask.any(0))])
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if (check_shape == 0):  # image is too dark so that we crop out everything,
                return Image.fromarray(img)  # return original image
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'