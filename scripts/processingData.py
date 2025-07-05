import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Subset, WeightedRandomSampler, DataLoader
from torchvision import transforms

def process_data(path,val=0.15,random_seed=123):

    base_ds = ImageFolder(root=path, transform=None) # cargamos el conjunto de datos entero, sin transformaciones
    dataset_size = len(base_ds)
    val_size     = int(val* dataset_size) # guardaremos por defecto un 15% de los datos para validación 
    train_size   = dataset_size - val_size

    generator = torch.Generator().manual_seed(random_seed) # fijamos una semilla aleatoria para reproducibilidad 
    train_ds_none, val_ds_none = random_split(
        base_ds,
        [train_size, val_size],
        generator=generator
    ) # obtenemos subcojuntos del dataset pero sin transformaciones

    # nos quedamos solo con los índices
    train_indices = train_ds_none.indices
    val_indices = val_ds_none.indices

    # Transformaciones para el conjunto de entrenamiento (con Data Augmentation)

    # valores de normalización de los canales RGB (los comunes en redes pre-entrenadas de ImageNet)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Transformaciones para el conjunto de validación
    
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Obtención de los subconjuntos de datos con sus respectivas transformaciones
    train_ds = Subset(
        ImageFolder(root=path, transform=train_transforms),
        train_indices
    )
    val_ds = Subset(
        ImageFolder(root=path, transform=val_transforms),
        val_indices
    )

    # Definición del sampler para balancear los datos de entrenamiento
    train_labels   = [base_ds.targets[i] for i in train_indices]
    class_counts   = np.bincount(train_labels)
    class_weights  = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Creación de los DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=sampler,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader