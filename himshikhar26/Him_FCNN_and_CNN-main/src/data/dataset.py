"""MNIST Dataset class for loading and preprocessing images"""
import os
import cv2
import torch
from glob import glob
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """
    Custom MNIST Dataset loader that loads images from directory structure
    
    Directory structure expected:
        path_dir/
            0/
                image1.jpg
                image2.jpg
            1/
                image1.jpg
            ...
    """
    
    def __init__(
        self,
        num_labels: int,
        img_pre_process=None,
        path_dir: str = "MNIST/Images/train",
        file_ext: str = ".jpg"
    ) -> None:
        """
        Initialize MNIST Dataset
        
        Args:
            num_labels: Number of classes (10 for MNIST digits 0-9)
            img_pre_process: Optional transforms.Compose object for preprocessing
            path_dir: Path to directory containing labeled subdirectories
            file_ext: File extension to search for (default: .jpg)
        """
        super().__init__()
        self.path_dir = path_dir
        self.num_labels = num_labels
        
        # Default preprocessing: normalize with MNIST mean and std
        self.img_pre_process = img_pre_process if img_pre_process is not None else \
            transforms.Compose([
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        if not self.verify_dir():
            raise FileNotFoundError(
                f"Unable to find {self.num_labels} labels in {self.path_dir}"
            )
        
        # Get all image paths
        self.img_path = glob(os.path.join(self.path_dir, f"**/*{file_ext}"), recursive=True)
        
        # Create label mapping
        self.label = {label: tag for tag, label in enumerate(os.listdir(self.path_dir))}

    def verify_dir(self) -> bool:
        """Verify that directory contains expected number of label subdirectories"""
        path_contents = os.listdir(self.path_dir)
        return len(path_contents) == self.num_labels
    
    def __len__(self):
        """Return total number of images"""
        return len(self.img_path)
    
    def load_image(self, path: str) -> Tensor:
        """
        Load and preprocess an image
        
        Args:
            path: Path to image file
            
        Returns:
            Preprocessed image tensor of shape (1, H, W)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Unable to find image at path {path}")
        
        # Read image and convert to grayscale
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).to(torch.float32)
        
        # Add channel dimension
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        
        # Apply preprocessing
        return self.img_pre_process(img_tensor)

    def __getitem__(self, idx):
        """
        Get image and label at index
        
        Args:
            idx: Index of sample
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        path = self.img_path[idx]
        img_tensor = self.load_image(path)
        
        # Extract label from directory name (second to last component of path)
        label = self.label[path.split("/")[-2]]
        label = torch.tensor(label).to(torch.long)
        
        return img_tensor, label
