import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchxrayvision as xrv
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms

# ------------------- CONFIGURACIÓN -------------------
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 30
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = True if DEVICE.type == "cuda" else False

print(f"Dispositivo: {DEVICE}")

train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)
])

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Normal', 'Neumonia', 'Tuberculosis']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar: {img_path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)  # 3 canales
        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    train_dataset = ChestXrayDataset(TRAIN_DIR, transform=train_transforms)
    val_dataset = ChestXrayDataset(VAL_DIR, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Entrenamiento: {len(train_dataset)} imágenes")
    print(f"Validación: {len(val_dataset)} imágenes")

    # Pesos de clase ajustados
    all_labels = train_dataset.labels
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights[0] = class_weights[0] * 1.8  
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Pesos de clase ajustados: {class_weights}")

    model_path = "models/backup_model.pth"
    if os.path.exists(model_path):
        print(f"Cargando modelo desde {model_path}...")
        # Crear arquitectura vacía
        model = xrv.models.DenseNet(in_channels=3, weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
        print("✅ Modelo cargado desde backup.")
    else:
        print("No se encontró backup. Creando modelo desde cero...")

        model = xrv.models.DenseNet(in_channels=3, weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )

    model.to(DEVICE)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    print("--- FASE 1: Reentrenando cabeza (capas congeladas) ---")
    for epoch in range(EPOCHS_PHASE1):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Fase1 Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={correct/total:.4f}")

    # ------------------- FASE 2: FINE-TUNING (descongelar todo) -------------------
    print("--- FASE 2: Fine-Tuning (descongelar capas) ---")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 5

    for epoch in range(EPOCHS_PHASE2):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Fase2 Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        scheduler.step(val_acc)
        print(f"Epoch {EPOCHS_PHASE1 + epoch + 1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model_pytorch.pth")
            print(f" Nuevo mejor modelo guardado con Acc: {best_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f" Early stopping en época {EPOCHS_PHASE1 + epoch + 1}")
                break

    # GUARDAR MODELO FINAL -------------------
    if os.path.exists("models/best_model_pytorch.pth"):
        model.load_state_dict(torch.load("models/best_model_pytorch.pth", map_location=DEVICE))
        print(" Cargado el mejor modelo encontrado.")
    else:
        print(" No se encontró best_model, guardando el último.")

    torch.save(model.state_dict(), "models/densenet121_finetuned_pytorch.pth")
    print(" Modelo fine-tuned guardado en 'models/densenet121_finetuned_pytorch.pth'")
    print(f"Mejor precisión en validación: {best_acc:.4f} ({best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()