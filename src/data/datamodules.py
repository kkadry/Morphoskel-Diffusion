# -*- coding: utf-8 -*-
import torchio as tio
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader


# Seed datamodule that returns a constant seed volume
class AnatomicSeedDataModule:
    def __init__(self, seed_folder: str, seed_name: str, batch_size: int):
        """
        Initialize the AnatomicSeedDataModule.

        Args:
            seed_folder (str): The folder containing the seed file.
            seed_name (str): The name of the seed file.
            batch_size (int): The batch size for data loading.
        """
        self.seed_path = os.path.join(seed_folder, seed_name)
        self.seed = torch.load(self.seed_path)
        # Repeating the seed according to batch size
        self.seed = self.seed[0:1].repeat(batch_size, 1, 1, 1, 1)

    def __iter__(self):
        # Wrap the seed in the same structure as __next__
        return iter([{"label": {"data": self.seed}}])

    def __next__(self):
        return {"label": {"data": next(iter(self))}}

    def train_dataloader(self):
        return self

    def val_dataloader(self):
        return self


# Datamodule
class AnatomicDataModule(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        data_dir: str,
        label_dir: str,
        image_dir: str,
        train_val_ratio: float,
        samples_per_volume: int,
        max_queue_length: int,
        patch_size: tuple,
        num_workers: int,
        resol: float,
        zcrop: int,
        xycrop: int,
        droplast: bool,
        aug_check: bool,
        aug_scale: float,
        **kwargs,
    ):
        """
        Initialize the AnatomicDataModule.

        Args:
            batch_size (int): The batch size for data loading.
            data_dir (str): The root directory for the dataset.
            label_dir (str): The directory containing label files.
            image_dir (str): The directory containing image files.
            train_val_ratio (float): The ratio of training to validation data.
            samples_per_volume (int): Number of samples to extract per volume.
            max_queue_length (int): Maximum queue length for data loading.
            patch_size (tuple): The size of patches to extract.
            num_workers (int): Number of worker processes for data loading.
            resol (float): Resolution factor for the data.
            zcrop (int): Number of slices to crop in z-direction.
            xycrop (int): Number of pixels to crop in x and y directions.
            droplast (bool): Whether to drop the last incomplete batch.
            aug_check (bool): Whether to perform augmentation check.
            aug_scale (float): Scale factor for augmentations.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.label_dir = Path(os.path.join(self.data_dir, label_dir))
        self.image_dir = (
            Path(os.path.join(self.data_dir, image_dir)) if image_dir else None
        )
        self.patients = os.listdir(self.label_dir)

        self.train_val_ratio = train_val_ratio
        self.samples_per_volume = samples_per_volume
        self.max_queue_length = max_queue_length
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.resol = resol
        self.zcrop = zcrop
        self.xycrop = xycrop
        self.droplast = droplast
        self.aug_check = aug_check
        self.aug_scale = aug_scale

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        """
        Prepare the data by creating subjects from label and image files.
        """
        label_training_paths = sorted(self.label_dir.glob("*.nii.gz"))
        if self.image_dir is not None:
            image_training_paths = sorted(self.image_dir.glob("*.nii.gz"))
        else:
            image_training_paths = label_training_paths
        self.subjects = []
        for label_path, image_path, patient in zip(
            label_training_paths, image_training_paths, self.patients
        ):
            # Creating a subject for each patient
            subject = tio.Subject(label=tio.LabelMap(label_path), name=patient)
            # Adding the image to the subject if it exists
            if image_path is not None:
                subject.add_image(tio.ScalarImage(image_path), "image")
            # Adding the subject to the list of subjects
            self.subjects.append(subject)

    def get_preprocessing_transform(self):
        """
        Get the preprocessing transform.

        Returns:
            tio.Transform: The preprocessing transform.
        """
        raise NotImplementedError

    def get_augmentation_transform(self):
        """
        Get the augmentation transform.

        Returns:
            tio.Transform: The augmentation transform.
        """
        raise NotImplementedError

    def setup(self):
        """
        Set up the datasets and data loaders.
        """
        # Calculate the number of subjects for training and validation
        num_subjects = len(self.subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))

        # Split subjects into training and validation sets
        train_subjects = self.subjects[:num_train_subjects]
        val_subjects = self.subjects[num_train_subjects:]

        # Get preprocessing and augmentation transforms
        self.preprocess = self.get_preprocessing_transform()
        self.augment = self.get_augmentation_transform()

        # Combine preprocessing and augmentation if aug_check is True
        if self.aug_check:
            self.transform = tio.Compose([self.preprocess, self.augment])
        else:
            self.transform = self.preprocess

        # Create training dataset
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)

        # Set up patch-based sampling for training
        self.sampler = tio.data.UniformSampler(patch_size=self.patch_size)
        self.patches_train_set = tio.Queue(
            subjects_dataset=self.train_set,
            max_length=self.max_queue_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=0,
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        # Create validation dataset
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.transform)

        # Set up patch-based sampling for validation
        self.patches_validation_set = tio.Queue(
            subjects_dataset=self.val_set,
            max_length=self.max_queue_length,
            samples_per_volume=self.samples_per_volume,
            sampler=self.sampler,
            num_workers=self.num_workers,
            shuffle_subjects=False,
            shuffle_patches=False,
        )

    def train_dataloader(self):
        """
        Get the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        return DataLoader(
            self.patches_train_set,
            self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.droplast,
        )

    def val_dataloader(self):
        """
        Get the validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        return DataLoader(
            self.patches_validation_set,
            self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.droplast,
        )


class CoronaryDataModule(AnatomicDataModule):
    def get_preprocessing_transform(self):
        """
        Get the preprocessing transform for coronary data.

        Returns:
            tio.Transform: The preprocessing transform.
        """
        preprocess = tio.Compose(
            [
                tio.transforms.Resize((self.resol[0], self.resol[1], self.resol[2])),
                tio.RemapLabels({4: 3}),
            ]
        )

        return preprocess

    def get_augmentation_transform(self):
        """
        Get the augmentation transform for coronary data.

        Returns:
            tio.Transform: The augmentation transform.
        """
        augment = tio.Compose(
            [
                tio.transforms.RandomAffine(
                    scales=(
                        0.5 * self.aug_scale,
                        0.5 * self.aug_scale,
                        0.5 * self.aug_scale,
                    ),
                    degrees=(180 * self.aug_scale, 180 * self.aug_scale, 180),
                ),
                tio.OneHot(),
            ]
        )
        return augment
