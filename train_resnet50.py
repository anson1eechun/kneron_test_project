import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import time

# ==========================================
# 設定區
# ==========================================
# 資料集路徑 (請確認您的資料夾名稱是 data)
data_dir = 'data'
# 設定運算裝置 (有 GPU 用 GPU，沒有用 CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用運算裝置: {device}")

# ==========================================
# 1. 資料預處理與載入
# ==========================================
print("正在載入圖片資料...")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"資料載入完成。類別: {class_names}")
print(f"訓練集數量: {dataset_sizes['train']}, 驗證集數量: {dataset_sizes['val']}")

# ==========================================
# 2. 定義訓練函數
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 每個 epoch 都有訓練和驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 設定為訓練模式
            else:
                model.eval()   # 設定為評估模式

            running_loss = 0.0
            running_corrects = 0

            # 批次讀取資料
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向傳播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 後向傳播與優化 (只在訓練階段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 紀錄最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n訓練完成，花費時間: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳驗證準確度: {best_acc:4f}')

    # 載入最佳權重
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 3. 設定模型 (ResNet50)
# ==========================================
print("\n正在下載並設定 ResNet50 模型...")

# 修改點 A: 使用 resnet50 (配合 Part-04 講義)
# 使用新的 weights 參數（適用於 torchvision >= 0.13.0）
# 如果您的版本較舊，可以改回 pretrained=True
try:
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
except AttributeError:
    # 向後兼容舊版本
    model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features

# 修改點 B: 設定輸出類別為 2 (螞蟻/蜜蜂)
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# 這裡我們只微調最後一層 (fc)，這樣訓練速度比較快
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# ==========================================
# 4. 開始訓練
# ==========================================
print("開始訓練 (這可能需要幾分鐘)...")
# 為了示範快速完成，我們只訓練 5 個 Epoch (講義通常建議更多，但 5 就足夠產生模型了)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=5)

# ==========================================
# 5. 匯出 ONNX (Part-04 的核心目標)
# ==========================================
print("\n[Start] Exporting to ONNX...")

# 切換到推論模式
model_ft.eval() 

# 建立虛擬輸入 (Batch size=1, RGB 3通道, 224x224)
dummy_input = torch.randn(1, 3, 224, 224, device=device)
onnx_file_name = "ants_bees.onnx"

# 匯出
torch.onnx.export(
    model_ft,
    dummy_input,
    onnx_file_name,
    verbose=False,
    input_names=['input'],   # 輸入節點命名為 input
    output_names=['output'], # 輸出節點命名為 output
    opset_version=11         # 建議使用 opset 11
)

print(f"匯出成功！檔案已儲存為: {onnx_file_name}")
print("請繼續進行 Part-05 的 Docker 轉換步驟。")