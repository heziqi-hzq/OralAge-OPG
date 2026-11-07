import os
import random
import cv2
import torch
import gc
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import torch.nn.functional as F

# Global configuration

CONFIG = {
    # Paths
    "IMAGE_DIR": "./data/images",
    "CSV_PATH": "./data/labels.csv",
    "OUTPUT_ROOT": "./outputs",

    # Training parameters
    "BATCH_SIZE": 8,
    "EPOCHS": 50,
    "LR": 1e-4,
    "SEED": 42,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # Age bins
    "START_AGE": 10,
    "NUM_AGE_CLASSES": 86,   # 86 integer-age bins: 10 to 95
}

IMAGE_DIR = CONFIG["IMAGE_DIR"]
CSV_PATH = CONFIG["CSV_PATH"]
BASE_OUTPUT_DIR = CONFIG["OUTPUT_ROOT"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]
EPOCHS = CONFIG["EPOCHS"]
LR = CONFIG["LR"]
SEED = CONFIG["SEED"]
DEVICE = CONFIG["DEVICE"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Dataset and preprocessing

class OPGDataset(Dataset):
    def __init__(self, df, img_dir, is_train=False):
        self.df = df
        self.img_dir = img_dir
        self.is_train = is_train
        self.rs, self.cs, self.fs = (1600, 3000), (1350, 2600), (512, 512)
    def __len__(self): return len(self.df)
    def process_image(self, img):
        h, w = img.shape[:2]
        scale = min(self.rs[0]/h, self.rs[1]/w)
        nh, nw = int(h*scale), int(w*scale)
        res = cv2.resize(img, (nw, nh))
        canvas = np.ones(self.rs, dtype=np.uint8) * 255
        canvas[(self.rs[0]-nh)//2:(self.rs[0]-nh)//2+nh, (self.rs[1]-nw)//2:(self.rs[1]-nw)//2+nw] = res
        top, left = (self.rs[0]-self.cs[0])//2, (self.rs[1]-self.cs[1])//2
        img_fin = cv2.resize(canvas[top:top+self.cs[0], left:left+self.cs[1]], self.fs)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_fin)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{str(row['image_name']).zfill(4)}.jpg")
        img = cv2.imread(img_path, 0)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = self.process_image(img)
        img_t = torch.from_numpy(img).float().unsqueeze(0)/255.0
        # Random horizontal flip with probability 0.5 during training.
        if self.is_train and random.random() > 0.5:
            img_t = torch.flip(img_t, dims=[2])
        return img_t, torch.tensor(row['age'], dtype=torch.float32)


# Loss

class MeanVarianceLoss(nn.Module):
    def __init__(self, lambda_1=0.2, lambda_2=0.05, start_age=10):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age

    def forward(self, input, target):
        class_dim = input.shape[-1]
        if len(input.shape) == 1:
            input = input.unsqueeze(0)

        m = nn.Softmax(dim=1)
        p = m(input)

        device = input.device
        a = torch.arange(class_dim, dtype=torch.float32, device=device)
        age_values = a + self.start_age

        mean = torch.squeeze((p * age_values).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        b = (age_values[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        target_closest_int = torch.round(target).long()
        target_idx = target_closest_int - self.start_age
        target_idx = torch.clamp(target_idx, 0, class_dim - 1)

        ce_loss = F.cross_entropy(input.view(-1, class_dim), target_idx)
        loss = ce_loss + (self.lambda_1 * mean_loss) + (self.lambda_2 * variance_loss)

        return loss

# OralAge-OPG model

def get_model():
    import timm
    m = timm.create_model('inception_resnet_v2', pretrained=True, in_chans=1, num_classes=0)
    in_dim = m.num_features

    regressor = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, CONFIG["NUM_AGE_CLASSES"])
    )

    return nn.Sequential(m, nn.Flatten(1), regressor)


# Five-fold training and validation

def train():
    arch = "inception_resnet_v2"
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)

    set_seed(SEED)
    df = pd.read_csv(CSV_PATH)
    df['image_name'] = df['image_name'].astype(str).str.zfill(4)

    df['age_group'] = pd.qcut(df['age'], q=5, labels=False, duplicates='drop')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_details = []

    print("Starting OralAge-OPG five-fold training...")

    for fold, (t_idx, v_idx) in enumerate(skf.split(df, df['age_group'])):

        # Apply augmentation only to training images.
        train_loader = DataLoader(OPGDataset(df.iloc[t_idx], IMAGE_DIR, is_train=True), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(OPGDataset(df.iloc[v_idx], IMAGE_DIR, is_train=False), batch_size=BATCH_SIZE)

        model = get_model().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = MeanVarianceLoss(start_age=CONFIG["START_AGE"])

        best_mae = float('inf')
        r_at_best = 0.0
        history = []

        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0
            for imgs, ages in train_loader:
                imgs, ages = imgs.to(DEVICE), ages.to(DEVICE)
                optimizer.zero_grad()

                output = model(imgs)
                loss = criterion(output, ages)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            ps, ts = [], []
            with torch.no_grad():
                for imgs, ages in val_loader:
                    imgs = imgs.to(DEVICE)
                    output = model(imgs)

                    p_softmax = F.softmax(output, dim=1)
                    a = torch.arange(CONFIG["NUM_AGE_CLASSES"], dtype=torch.float32, device=DEVICE)
                    pred_age = (p_softmax * (a + CONFIG["START_AGE"])).sum(dim=1)
                    ps.append(pred_age.cpu())

                    ts.append(ages)

            ps, ts = torch.cat(ps).numpy().flatten(), torch.cat(ts).numpy()
            mae = np.mean(np.abs(ps - ts))

            if mae < best_mae:
                best_mae = mae
                try:
                    r_at_best = pearsonr(ps, ts)[0]
                except:
                    r_at_best = 0.0

                save_path = os.path.join(BASE_OUTPUT_DIR, f"best_weight_{arch}_fold{fold+1}.pth")
                torch.save(model.state_dict(), save_path)

            history.append([epoch + 1, train_loss / len(train_loader), mae])

        history_df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'val_mae'])
        history_df.to_csv(os.path.join(BASE_OUTPUT_DIR, f"history_{arch}_fold{fold+1}.csv"), index=False)

        print(f"Fold {fold+1} complete | Best MAE: {best_mae:.3f} | Pearson r: {r_at_best:.3f}")
        fold_details.append({"Fold": fold+1, "MAE": best_mae, "Pearson_R": r_at_best})

    details_df = pd.DataFrame(fold_details)
    details_df.to_csv(os.path.join(BASE_OUTPUT_DIR, f"detail_{arch}_5fold.csv"), index=False)

    res = {
        "Model": arch,
        "Mean_MAE": details_df['MAE'].mean(),
        "Std_MAE": details_df['MAE'].std(),
        "Mean_R": details_df['Pearson_R'].mean()
    }
    pd.DataFrame([res]).to_csv(os.path.join(BASE_OUTPUT_DIR, f"summary_{arch}.csv"), index=False)

    print(f"Training complete. Results and weights saved to: {BASE_OUTPUT_DIR}")

    del model, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return res

if __name__ == "__main__":
    result = train()
