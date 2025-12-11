import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

DATA_PATH = "/home/elisaveloso/reinforcement-learning/project_1/data"

def get_client_loader(cid: int, num_clients: int, batch_size: int = 32):
    # Transformações básicas
    transform = transforms.Compose([
        transforms.Resize((120, 160)), # Reduzir tamanho para economizar memória
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Carregar dataset
    dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    
    # Garantir reprodutibilidade no split
    generator = torch.Generator().manual_seed(42)
    
    # Dividir dados
    total_size = len(dataset)
    part_size = total_size // num_clients
    lengths = [part_size] * num_clients
    if sum(lengths) < total_size:
        lengths[-1] += total_size - sum(lengths)
        
    partitions = random_split(dataset, lengths, generator=generator)
    
    # Pegar a partição do cliente
    my_partition = partitions[cid]
    
    # Dividir em treino e validação
    len_train = int(len(my_partition) * 0.9)
    len_val = len(my_partition) - len_train
    
    # Split local também com seed fixa para consistência (opcional)
    train_subset, val_subset = random_split(my_partition, [len_train, len_val], generator=generator)
    
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_subset, batch_size=batch_size, num_workers=0)
    
    return trainloader, valloader, dataset.classes

# Função antiga para compatibilidade (não usada mais no main refatorado)
def load_data(num_clients: int, batch_size: int = 32):
    # ... (pode ser removida ou mantida)
    pass
