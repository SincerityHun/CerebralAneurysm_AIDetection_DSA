from model import UNet,device
import torch
from tqdm import tqdm
from data_loader import dataloader
from dataset import SatelliteDataset,DataLoader
from data_loader import transform
import numpy as np
from utils import rle_encode,submission


if __name__ == "__main__":
    # 1. Train
    # model 초기화
    model = UNet().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(10):  # 10 에폭 동안 학습합니다.
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')

    # 2. Test
    test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 3. Inference
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)
    
    # 4. Submit
    submission('mask_rle',result)