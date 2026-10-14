import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, ACSIncome, BasicProblem


class NumpyDataset(Dataset):
    def __init__(self, X, y, g):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.g = torch.tensor(g, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.g[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Define custom task with SEX as group (regression: no binarization)
ACSIncomeSex = BasicProblem(
    features=ACSIncome.features,
    target='PINCP',
    target_transform=lambda x: x,
    group='SEX',
    preprocess=lambda x: x,
    postprocess=lambda x: x,
)

# Define regression task with RAC1P as group (no binarization)
ACSIncomeRegression = BasicProblem(
    features=ACSIncome.features,
    target='PINCP',
    target_transform=lambda x: x,
    group='RAC1P',
    preprocess=lambda x: x,
    postprocess=lambda x: x,
)


def load_acs_income(states=("CA",), year=2018, group_by="race", features=None, employed_only=False):
    """
    Load ACS Income dataset.

    Args:
        states: tuple of state codes
        year: survey year
        group_by: str
            - "race": Use RAC1P (race) as group (9 groups)
            - "sex": Use SEX (gender) as group (2 groups: 1=Male, 2=Female)
            - "race_binary": Use RAC1P binarized as White (1) vs Non-White (2-9) (2 groups)
        features: list of str or None
            - If None, use all default features
            - If list, select only specified features from ['AGEP', 'COW', 'SCHL', 'MAR',
              'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
            - Example: ['AGEP', 'SCHL', 'OCCP', 'WKHP'] for core income predictors
        employed_only: bool
            - If True, filter to full-time year-round workers:
              ESR == 1 (civilian employed, at work),
              WKHP >= 35 (35+ hours/week),
              WKW == 1 (50-52 weeks/year)
    """
    data_source = ACSDataSource(
        survey_year=str(year),
        horizon="1-Year",
        survey="person",
    )
    acs_data = data_source.get_data(states=list(states), download=True)

    # Handle 2024 data compatibility: RELP renamed to RELSHIPP
    year_int = int(year) if isinstance(year, str) else year
    if year_int >= 2024:
        if 'RELSHIPP' in acs_data.columns and 'RELP' not in acs_data.columns:
            acs_data['RELP'] = acs_data['RELSHIPP']

    # Also handle case where RELP is missing for any reason
    if 'RELP' not in acs_data.columns and 'RELSHIPP' in acs_data.columns:
        acs_data['RELP'] = acs_data['RELSHIPP']

    # Use appropriate task (regression: no target binarization)
    if group_by == "sex":
        task = ACSIncomeSex
    else:
        task = ACSIncomeRegression

    # Drop rows with NaN in any relevant column before extracting numpy arrays
    relevant_cols = list(task.features) + [task.target, task.group]
    n_before = len(acs_data)
    acs_data = acs_data.dropna(subset=[c for c in relevant_cols if c in acs_data.columns])
    n_removed = n_before - len(acs_data)
    if n_removed > 0:
        print(f"Removed {n_removed} samples with NaN values")

    if employed_only:
        n_before_esr = len(acs_data)
        acs_data = acs_data[
            (acs_data['ESR'] == 1) &
            (acs_data['WKHP'] >= 35) &
            (acs_data['WKW'] == 1)    # WKW 1 = 50-52 weeks/year
        ]
        print(f"Filtered to full-time year-round employed (ESR==1, WKHP>=35, WKW==1): {n_before_esr} -> {len(acs_data)}")

    X, y, group = task.df_to_numpy(acs_data)

    # Remove non-positive income and apply log transform
    pos_mask = y > 0
    if not pos_mask.all():
        n_nonpos = int((~pos_mask).sum())
        print(f"Removed {n_nonpos} samples with non-positive income")
        X, y, group = X[pos_mask], y[pos_mask], group[pos_mask]
    y = np.log(y)

    # Create derived binary features before subsetting
    # MARC: married (MAR==1) vs not (MAR!=1)
    if 'MAR' in task.features:
        mar_idx = task.features.index('MAR')
        mar_binary = (X[:, mar_idx] == 1).astype(np.float64)
    else:
        mar_binary = None

    # RELPC: householder (RELP==0) vs not (RELP!=0)
    if 'RELP' in task.features:
        relp_idx = task.features.index('RELP')
        relp_binary = (X[:, relp_idx] == 0).astype(np.float64)
    else:
        relp_binary = None

    # Select feature subset if specified
    if features is not None:
        all_features = task.features

        # Map derived feature names to their arrays
        derived = {}
        if mar_binary is not None:
            derived['MARC'] = mar_binary
        if relp_binary is not None:
            derived['RELPC'] = relp_binary

        # Build feature matrix by checking both task.features and raw data
        X_cols = []
        selected_features = []

        for f in features:
            if f in derived:
                # Derived binary feature
                X_cols.append(derived[f].reshape(-1, 1))
                selected_features.append(f)
            elif f in all_features:
                # Feature is in task.features, extract from X
                idx = all_features.index(f)
                X_cols.append(X[:, idx:idx+1])
                selected_features.append(f)
            elif f in acs_data.columns:
                # Feature exists in raw data but not in task.features
                # This handles renamed features like RELP -> RELSHIPP in 2024
                X_cols.append(acs_data[f].to_numpy().reshape(-1, 1))
                selected_features.append(f)
            else:
                # Feature not found anywhere
                print(f"Warning: '{f}' not found in task.features or raw data, skipping")

        if X_cols:
            X = np.hstack(X_cols)
        else:
            X = np.array([]).reshape(len(y), 0)

        print(f"Selected {len(selected_features)} features: {selected_features}")

    # Binarize race if requested
    if group_by == "race_binary":
        # RAC1P: 1 = White alone, 2-9 = Non-White
        group = np.where(group == 1, 0, 1)  # 0 = White, 1 = Non-White

    unique_groups, group_ids = np.unique(group, return_inverse=True)
    num_groups = len(unique_groups)
    print(f"Loaded {len(y)} samples, {X.shape[1]} features, {num_groups} groups (group_by={group_by})")
    return X, y.astype(np.float32), group_ids, num_groups, unique_groups


def make_splits(X, y, g, test_size=0.2, val_size=0.2, seed=0):
    X_train, X_temp, y_train, y_temp, g_train, g_temp = train_test_split(
        X, y, g, test_size=(val_size + test_size),
        random_state=seed
    )
    rel_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, g_val, g_test = train_test_split(
        X_temp, y_temp, g_temp,
        test_size=rel_test_size,
        random_state=seed,
    )
    return (X_train, y_train, g_train,
            X_val, y_val, g_val,
            X_test, y_test, g_test)


def make_dataloaders(X_train, y_train, g_train,
                     X_val, y_val, g_val,
                     batch_size=1024):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_ds = NumpyDataset(X_train_scaled, y_train, g_train)
    val_ds = NumpyDataset(X_val_scaled, y_val, g_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    return train_loader, val_loader, scaler


def evaluate(model, loader, num_groups, device="cpu"):
    """Evaluate model with RMSE (root mean squared error) per group."""
    model.eval()

    total_se = 0.0
    total_n = 0
    group_se = [0.0 for _ in range(num_groups)]
    group_counts = [0 for _ in range(num_groups)]

    with torch.no_grad():
        for X, y, g in loader:
            X = X.to(device)
            y = y.to(device)
            g = g.to(device)
            preds = model(X)
            if preds.dim() > 1:
                preds = preds.squeeze(-1)
            se = (preds - y) ** 2

            total_se += se.sum().item()
            total_n += len(y)

            for gid in range(num_groups):
                mask = (g == gid)
                if mask.any():
                    group_se[gid] += se[mask].sum().item()
                    group_counts[gid] += mask.sum().item()

    mean_rmse = np.sqrt(total_se / max(total_n, 1))
    group_rmse = []
    for gid in range(num_groups):
        if group_counts[gid] > 0:
            group_rmse.append(np.sqrt(group_se[gid] / group_counts[gid]))
        else:
            group_rmse.append(float("nan"))

    worst_group_rmse = np.nanmax(group_rmse) if len(group_rmse) > 0 else 0.0
    return mean_rmse, group_rmse, worst_group_rmse


def compute_baseline_losses(y_train, g_train, num_groups):
    """
    Compute baseline RMSE for constant predictor f_0(x) = mean(y),
    where mean(y) is the overall training mean.
    """
    baseline_losses = np.zeros(num_groups, dtype=np.float32)

    mu_0 = float(y_train.mean())

    print(f"Overall mean income = {mu_0:.2f}")
    print(f"Baseline predictor f_0(x) = {mu_0:.2f} (constant)")

    for gid in range(num_groups):
        mask = (g_train == gid)
        y_g = y_train[mask]

        if len(y_g) == 0:
            baseline_losses[gid] = float("nan")
            continue

        mse = float(np.mean((y_g - mu_0) ** 2))
        baseline_losses[gid] = np.sqrt(mse)

    return baseline_losses
