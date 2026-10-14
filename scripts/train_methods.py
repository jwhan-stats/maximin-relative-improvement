import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from folktables_utils import NumpyDataset, evaluate


def get_model(model_type, input_dim, **kwargs):
    """
    Factory function to create models.

    model_type: str
        - "mlp": Multi-layer perceptron (2 hidden layers)
        - "linear": Single linear layer (equivalent to logistic regression)
        - "small_mlp": Smaller MLP (1 hidden layer)
    """
    if model_type == "mlp":
        hidden_dim = kwargs.get("hidden_dim", 128)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    elif model_type == "small_mlp":
        hidden_dim = kwargs.get("hidden_dim", 64)
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    elif model_type in ("linear", "logistic", "regression"):
        # Linear model
        return nn.Linear(input_dim, 1)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def prepare_data(X_train, y_train, g_train, X_val, y_val, g_val,
                 batch_size=2048, scale=True):
    """
    Prepare dataloaders with optional scaling.
    """
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        scaler = None

    train_ds = NumpyDataset(X_train_scaled, y_train, g_train)
    val_ds = NumpyDataset(X_val_scaled, y_val, g_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, scaler, X_train_scaled, X_val_scaled


def _adaptive_eta_from_losses(base_step, losses, eps=1e-8):
    """
    Scale exponentiated-gradient step size based on the spread of group losses.
    Keeps eta small when groups are already balanced and grows it when
    worst-group loss is noticeably larger.
    """
    if losses is None or len(losses) == 0:
        return base_step
    arr = np.asarray(losses, dtype=np.float32)
    spread = float(np.nanmax(arr) - np.nanmin(arr))
    mean_mag = float(np.nanmean(np.abs(arr)) + eps)
    factor = spread / mean_mag
    return float(np.clip(base_step * max(factor, 0.25), base_step * 0.25, base_step * 4.0))


def _adaptive_eta_from_regrets(base_step, regrets, eps=1e-8):
    """
    Normalize regret magnitudes so eta remains stable even when regrets
    are close to zero (common for logistic regression).
    """
    if regrets is None or len(regrets) == 0:
        return base_step
    arr = np.asarray(regrets, dtype=np.float32)
    max_abs = float(np.nanmax(np.abs(arr)))
    if max_abs < eps:
        return base_step * 0.5
    normalized = np.abs(arr) / (max_abs + eps)
    scale = float(np.nanmean(normalized))
    factor = 0.5 + scale  # push harder when normalized regrets differ
    return float(np.clip(base_step * factor, base_step * 0.25, base_step * 4.0))


def _adaptive_eta_from_ri(base_step, rel_improvements):
    """
    Adapt eta for relative-improvement updates using the standard deviation
    of valid RI scores. When groups have similar RI, updates stay mild;
    when one group lags the variance increases and eta grows.
    """
    if rel_improvements is None or len(rel_improvements) == 0:
        return base_step
    arr = np.asarray(rel_improvements, dtype=np.float32)
    if arr.size == 0:
        return base_step
    std = float(np.nanstd(arr))
    factor = 0.5 + std
    return float(np.clip(base_step * factor, base_step * 0.25, base_step * 4.0))


def train_group_dro(model, train_loader, val_loader, num_groups,
                    device="cpu", lr=1e-3, epochs=5, verbose=True, warm_start_epochs=0,
                    group_weight_step_size=0.01, save_best=True,
                    patience=50, min_delta=1e-4):
    """
    Train with Group DRO using exponentiated gradient on group weights.

    Implements the algorithm from Sagawa et al. (2020):
    - Maintains global per-group weights
    - Updates model with weighted ERM each epoch
    - Updates group weights via exponentiated gradient based on group losses

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (used for best model selection)
        num_groups: Number of groups
        device: Device to train on
        lr: Learning rate for model parameters
        epochs: Total number of epochs
        verbose: Whether to print progress
        warm_start_epochs: Number of epochs to pre-train with ERM (included in total epochs)
        group_weight_step_size: Step size for group weight updates (eta in paper)
        save_best: Whether to save and return best validation model
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to reset patience counter

    Returns:
        model: Trained model (best validation if save_best=True, else final)
    """
    model.to(device)

    # Ensure warm_start_epochs doesn't exceed total epochs
    warm_start_epochs = min(warm_start_epochs, epochs)
    dro_epochs = epochs - warm_start_epochs

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    # For saving best model
    best_val_worst = float('inf')
    best_model_state = None

    # ========== Phase 1: Warm start with ERM ==========
    if warm_start_epochs > 0:
        if verbose:
            print(f"Phase 1: Warm-start with ERM for {warm_start_epochs} epochs...")

        for epoch in range(1, warm_start_epochs + 1):
            model.train()
            for X, y, g in train_loader:
                X, y = X.to(device), y.to(device)

                logits = model(X).squeeze(-1)
                loss = loss_fn(logits, y).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose and (epoch % max(1, warm_start_epochs // 5) == 0 or epoch == warm_start_epochs):
                train_mean, _, train_worst = evaluate(model, train_loader, num_groups, device)
                val_mean, _, val_worst = evaluate(model, val_loader, num_groups, device)
                print(f"  Epoch {epoch}/{warm_start_epochs}: "
                      f"train={train_mean:.4f}(worst={train_worst:.4f}), "
                      f"val={val_mean:.4f}(worst={val_worst:.4f})")

                # Track best model
                if save_best and val_worst < best_val_worst:
                    best_val_worst = val_worst
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ========== Phase 2: Group DRO with reweighting ==========
    if dro_epochs > 0:
        if verbose:
            print(f"\nPhase 2: Group DRO with exponentiated gradient for {dro_epochs} epochs...")
            print(f"  Group weight step size: {group_weight_step_size}")
            print(f"  Early stopping: patience={patience}, min_delta={min_delta}")

        # Initialize group weights (uniform)
        group_weights = np.ones(num_groups, dtype=np.float32) / num_groups

        # Early stopping variables
        epochs_without_improvement = 0
        best_metric = float('inf')  # For DRO: minimize worst-group loss

        for epoch in range(1, dro_epochs + 1):
            global_epoch = warm_start_epochs + epoch

            # ===== Mini-batch training with current group weights =====
            model.train()
            for X, y, g in train_loader:
                X, y, g = X.to(device), y.to(device), g.to(device)

                logits = model(X).squeeze(-1)
                per_ex_loss = loss_fn(logits, y)

                # Compute weighted loss using current global group weights
                # L = sum_g w_g * (1/n_g) * sum_{i in g} loss_i
                weighted_loss = 0.0
                for gid in range(num_groups):
                    mask = (g == gid)
                    if mask.any().item():
                        group_loss = per_ex_loss[mask].mean()
                        weighted_loss = weighted_loss + group_weights[gid] * group_loss

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            # ===== Compute full dataset group losses =====
            model.eval()
            with torch.no_grad():
                all_losses = []
                all_groups = []

                for X, y, g in train_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X).squeeze(-1)
                    per_ex_loss = loss_fn(logits, y)
                    all_losses.append(per_ex_loss)
                    all_groups.append(g.to(device))

                all_losses = torch.cat(all_losses)
                all_groups = torch.cat(all_groups)

                # Compute group losses
                group_losses = np.zeros(num_groups, dtype=np.float32)
                for gid in range(num_groups):
                    mask = (all_groups == gid)
                    if mask.any().item():
                        group_losses[gid] = all_losses[mask].mean().item()

            # ===== Update group weights via adaptive exponentiated gradient =====
            # Normalize to unit scale for task-agnostic updates
            loss_scale = np.abs(group_losses).mean() + 1e-8
            normalized_losses = group_losses / loss_scale
            centered_normalized = normalized_losses - normalized_losses.mean()
            adaptive_eta = _adaptive_eta_from_losses(group_weight_step_size, normalized_losses)
            group_weights = group_weights * np.exp(adaptive_eta * centered_normalized)
            group_weights = group_weights / group_weights.sum()

            # ===== Evaluation and logging =====
            train_mean, train_group, train_worst = evaluate(model, train_loader, num_groups, device)
            val_mean, val_group, val_worst = evaluate(model, val_loader, num_groups, device)

            if verbose:
                print(f"  Epoch {global_epoch}/{epochs}: "
                      f"train={train_mean:.4f}(worst={train_worst:.4f}), "
                      f"val={val_mean:.4f}(worst={val_worst:.4f})")

                # Print detailed per-group information for all group counts
                train_losses_str = ", ".join([f"G{g}={train_group[g]:.4f}" for g in range(num_groups)])
                val_losses_str = ", ".join([f"G{g}={val_group[g]:.4f}" for g in range(num_groups)])
                weights_str = ", ".join([f"G{g}={group_weights[g]:.4f}" for g in range(num_groups)])
                print(f"    Train losses: {train_losses_str}")
                print(f"    Val losses:   {val_losses_str}")
                print(f"    Weights: {weights_str}")

            # Track best model
            if save_best and val_worst < best_val_worst:
                best_val_worst = val_worst
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Early stopping check
            current_metric = val_worst  # DRO objective: worst-group loss
            if best_metric - current_metric > min_delta:
                best_metric = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\n  Early stopping triggered at epoch {global_epoch}")
                    print(f"  No improvement for {patience} epochs (best={best_metric:.4f})")
                break

    # ========== Restore best model if requested ==========
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model with val worst-group loss: {best_val_worst:.4f}")

    return model


def train_group_oracles(X_train, y_train, g_train, num_groups, model_fn=None,
                        device="cpu", lr=1e-3, epochs=10, batch_size=2048, verbose=False,
                        closed_form=True):
    """
    Train separate models for each group to estimate oracle losses (RMSE).

    Args:
        closed_form: If True, use sklearn LinearRegression (exact OLS solution).
                     If False, use gradient descent with model_fn.
        model_fn: callable that returns a fresh model given input_dim (only used when closed_form=False)
    """
    oracle_losses = np.zeros(num_groups, dtype=np.float32)

    if closed_form:
        for gid in range(num_groups):
            mask = (g_train == gid)
            Xg = X_train[mask]
            yg = y_train[mask]

            if verbose:
                print(f"Oracle for group {gid} (closed-form OLS), n={len(yg)}")

            reg = LinearRegression().fit(Xg, yg)
            preds = reg.predict(Xg)
            rmse = float(np.sqrt(np.mean((preds - yg) ** 2)))
            oracle_losses[gid] = rmse

            if verbose:
                print(f"  Oracle RMSE_{gid}* = {rmse:.4f}")

        return oracle_losses

    # Gradient descent fallback for non-linear models
    assert model_fn is not None, "model_fn required when closed_form=False"
    loss_fn = nn.MSELoss()
    input_dim = X_train.shape[1]

    for gid in range(num_groups):
        mask = (g_train == gid)
        Xg = X_train[mask]
        yg = y_train[mask]

        if verbose:
            print(f"Training oracle for group {gid}, n={len(yg)}")

        ds = NumpyDataset(Xg, yg, np.full_like(yg, gid))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        model = model_fn(input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            model.train()
            for Xb, yb, _ in loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb).squeeze(-1)
                loss = loss_fn(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            eval_loss = 0.0
            eval_n = 0
            with torch.no_grad():
                for Xb, yb, _ in loader:
                    Xb = Xb.to(device)
                    yb = yb.to(device)
                    logits = model(Xb).squeeze(-1)
                    loss = loss_fn(logits, yb)
                    eval_loss += loss.item() * len(yb)
                    eval_n += len(yb)

            current_loss = eval_loss / max(eval_n, 1)
            if current_loss < best_loss:
                best_loss = current_loss

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs):
                print(f"  Epoch {epoch}/{epochs}: rmse={np.sqrt(current_loss):.4f} (best={np.sqrt(best_loss):.4f})")

        oracle_losses[gid] = np.sqrt(best_loss)

        if verbose:
            print(f"  Oracle RMSE_{gid}* = {oracle_losses[gid]:.4f}")

    return oracle_losses



def train_minimax_regret(model, train_loader, val_loader, num_groups, oracle_losses,
                         device="cpu", lr=1e-3, epochs=5, verbose=True, warm_start_epochs=0,
                         group_weight_step_size=0.01, save_best=True,
                         patience=50, min_delta=1e-4):
    """
    Train with Minimax Regret using exponentiated gradient on group weights.

    Similar to Group DRO but weights are based on regrets (L_g - L_g*) instead of losses.
    Regrets are automatically normalized to unit scale for task-agnostic optimization.

    Args:
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to reset patience counter
    """
    model.to(device)

    warm_start_epochs = min(warm_start_epochs, epochs)
    regret_epochs = epochs - warm_start_epochs

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    best_val_worst_regret = float('inf')
    best_model_state = None

    # ========== Phase 1: Warm start with ERM ==========
    if warm_start_epochs > 0:
        if verbose:
            print(f"Phase 1: Warm-start with ERM for {warm_start_epochs} epochs...")

        for epoch in range(1, warm_start_epochs + 1):
            model.train()
            for X, y, g in train_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X).squeeze(-1)
                loss = loss_fn(logits, y).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose and (epoch % max(1, warm_start_epochs // 5) == 0 or epoch == warm_start_epochs):
                train_mean, train_group, train_worst = evaluate(model, train_loader, num_groups, device)
                val_mean, val_group, val_worst = evaluate(model, val_loader, num_groups, device)

                train_regrets = [train_group[g] - oracle_losses[g] for g in range(num_groups)]
                val_regrets = [val_group[g] - oracle_losses[g] for g in range(num_groups)]

                print(f"  Epoch {epoch}/{warm_start_epochs}: "
                      f"train={train_mean:.4f}(regret={max(train_regrets):.4f}), "
                      f"val={val_mean:.4f}(regret={max(val_regrets):.4f})")

                if save_best and max(val_regrets) < best_val_worst_regret:
                    best_val_worst_regret = max(val_regrets)
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ========== Phase 2: Regret optimization with reweighting ==========
    if regret_epochs > 0:
        if verbose:
            print(f"\nPhase 2: Minimax Regret with exponentiated gradient for {regret_epochs} epochs...")
            print(f"  Group weight step size: {group_weight_step_size}")
            print(f"  Early stopping: patience={patience}, min_delta={min_delta}")

        group_weights = np.ones(num_groups, dtype=np.float32) / num_groups
        oracle_losses_t = torch.tensor(oracle_losses, dtype=torch.float32, device=device)

        # Early stopping variables
        epochs_without_improvement = 0
        best_metric = float('inf')  # For Regret: minimize worst-group regret

        for epoch in range(1, regret_epochs + 1):
            global_epoch = warm_start_epochs + epoch

            # ===== Mini-batch training with current group weights =====
            model.train()
            for X, y, g in train_loader:
                X, y, g = X.to(device), y.to(device), g.to(device)
                logits = model(X).squeeze(-1)
                per_ex_loss = loss_fn(logits, y)

                # Weighted regret loss
                weighted_loss = 0.0
                for gid in range(num_groups):
                    mask = (g == gid)
                    if mask.any().item():
                        group_loss = per_ex_loss[mask].mean()
                        regret = group_loss - oracle_losses_t[gid]
                        weighted_loss = weighted_loss + group_weights[gid] * regret

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            # ===== Compute full dataset group regrets =====
            model.eval()
            with torch.no_grad():
                all_losses = []
                all_groups = []

                for X, y, g in train_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X).squeeze(-1)
                    per_ex_loss = loss_fn(logits, y)
                    all_losses.append(per_ex_loss)
                    all_groups.append(g.to(device))

                all_losses = torch.cat(all_losses)
                all_groups = torch.cat(all_groups)

                group_regrets = np.zeros(num_groups, dtype=np.float32)
                for gid in range(num_groups):
                    mask = (all_groups == gid)
                    if mask.any().item():
                        group_loss = all_losses[mask].mean().item()
                        group_regrets[gid] = group_loss - oracle_losses[gid]

            # ===== Update group weights via adaptive exponentiated gradient =====
            # Normalize regrets to unit scale for task-agnostic updates
            regret_scale_factor = np.abs(group_regrets).mean() + 1e-8
            normalized_regrets = group_regrets / regret_scale_factor
            centered_normalized = normalized_regrets - normalized_regrets.mean()
            adaptive_eta = _adaptive_eta_from_regrets(group_weight_step_size, normalized_regrets)
            group_weights = group_weights * np.exp(adaptive_eta * centered_normalized)
            group_weights = group_weights / group_weights.sum()

            # ===== Evaluation and logging =====
            train_mean, train_group, train_worst = evaluate(model, train_loader, num_groups, device)
            val_mean, val_group, val_worst = evaluate(model, val_loader, num_groups, device)

            train_regrets = [train_group[g] - oracle_losses[g] for g in range(num_groups)]
            val_regrets = [val_group[g] - oracle_losses[g] for g in range(num_groups)]

            if verbose:
                print(f"  Epoch {global_epoch}/{epochs}: "
                      f"train={train_mean:.4f}(regret={max(train_regrets):.4f}), "
                      f"val={val_mean:.4f}(regret={max(val_regrets):.4f})")

                # Print detailed per-group information for all group counts
                train_regrets_str = ", ".join([f"G{g}={train_regrets[g]:.4f}" for g in range(num_groups)])
                val_regrets_str = ", ".join([f"G{g}={val_regrets[g]:.4f}" for g in range(num_groups)])
                weights_str = ", ".join([f"G{g}={group_weights[g]:.4f}" for g in range(num_groups)])
                print(f"    Train regrets: {train_regrets_str}")
                print(f"    Val regrets:   {val_regrets_str}")
                print(f"    Weights: {weights_str}")

            if save_best and max(val_regrets) < best_val_worst_regret:
                best_val_worst_regret = max(val_regrets)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Early stopping check
            current_metric = max(val_regrets)  # Regret objective: worst-group regret
            if best_metric - current_metric > min_delta:
                best_metric = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\n  Early stopping triggered at epoch {global_epoch}")
                    print(f"  No improvement for {patience} epochs (best={best_metric:.4f})")
                break

    # ========== Restore best model if requested ==========
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model with val worst regret: {best_val_worst_regret:.4f}")

    return model



def train_maximin_relative_improvement(model, train_loader, val_loader, num_groups,
                                        baseline_losses, oracle_losses,
                                        device="cpu", lr=1e-3, epochs=5,
                                        min_gap=1e-3, verbose=True, warm_start_epochs=0,
                                        group_weight_step_size=0.01, save_best=True,
                                        patience=50, min_delta=1e-4):
    """
    Train with Maximin Relative Improvement using exponentiated gradient on group weights.

    Weights are based on relative improvements: (baseline_g - L_g) / (baseline_g - oracle_g).

    Args:
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum improvement to reset patience counter
    """
    model.to(device)

    warm_start_epochs = min(warm_start_epochs, epochs)
    ri_epochs = epochs - warm_start_epochs

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    gaps = baseline_losses - oracle_losses
    valid_groups = gaps >= min_gap

    if not valid_groups.any():
        print("WARNING: No groups have sufficient gap for optimization!")
        valid_groups = np.ones(num_groups, dtype=bool)

    best_val_worst_ri = float('-inf')
    best_model_state = None

    # ========== Phase 1: Warm start with ERM ==========
    if warm_start_epochs > 0:
        if verbose:
            print(f"Phase 1: Warm-start with ERM for {warm_start_epochs} epochs...")

        for epoch in range(1, warm_start_epochs + 1):
            model.train()
            for X, y, g in train_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X).squeeze(-1)
                loss = loss_fn(logits, y).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if verbose and (epoch % max(1, warm_start_epochs // 5) == 0 or epoch == warm_start_epochs):
                train_mean, train_group, train_worst = evaluate(model, train_loader, num_groups, device)
                val_mean, val_group, val_worst = evaluate(model, val_loader, num_groups, device)

                train_ri = [(baseline_losses[g] - train_group[g]) / gaps[g] if valid_groups[g] else float('nan')
                           for g in range(num_groups)]
                val_ri = [(baseline_losses[g] - val_group[g]) / gaps[g] if valid_groups[g] else float('nan')
                         for g in range(num_groups)]

                print(f"  Epoch {epoch}/{warm_start_epochs}: "
                      f"train={train_mean:.4f}(RI={np.nanmin(train_ri):.4f}), "
                      f"val={val_mean:.4f}(RI={np.nanmin(val_ri):.4f})")

                if save_best and np.nanmin(val_ri) > best_val_worst_ri:
                    best_val_worst_ri = np.nanmin(val_ri)
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ========== Phase 2: RI optimization with reweighting ==========
    if ri_epochs > 0:
        if verbose:
            print(f"\nPhase 2: Maximin Relative Improvement with exponentiated gradient for {ri_epochs} epochs...")
            print(f"  Optimizing over {valid_groups.sum()}/{num_groups} groups (gap >= {min_gap})")
            print(f"  Group weight step size: {group_weight_step_size}")
            print(f"  Early stopping: patience={patience}, min_delta={min_delta}")

        group_weights = np.ones(num_groups, dtype=np.float32) / num_groups
        baseline_t = torch.tensor(baseline_losses, dtype=torch.float32, device=device)
        oracle_t = torch.tensor(oracle_losses, dtype=torch.float32, device=device)
        gaps_t = baseline_t - oracle_t

        # Early stopping variables
        epochs_without_improvement = 0
        best_metric = float('-inf')  # For RI: maximize worst-group relative improvement

        for epoch in range(1, ri_epochs + 1):
            global_epoch = warm_start_epochs + epoch

            # ===== Mini-batch training with current group weights =====
            model.train()
            for X, y, g in train_loader:
                X, y, g = X.to(device), y.to(device), g.to(device)
                logits = model(X).squeeze(-1)
                per_ex_loss = loss_fn(logits, y)

                # Weighted relative improvement loss (negative because we maximize)
                weighted_loss = 0.0
                for gid in range(num_groups):
                    if not valid_groups[gid]:
                        continue
                    mask = (g == gid)
                    if mask.any().item():
                        group_loss = per_ex_loss[mask].mean()
                        rel_imp = (baseline_t[gid] - group_loss) / gaps_t[gid]
                        weighted_loss = weighted_loss - group_weights[gid] * rel_imp  # Negative for maximization

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

            # ===== Compute full dataset group relative improvements =====
            model.eval()
            with torch.no_grad():
                all_losses = []
                all_groups = []

                for X, y, g in train_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model(X).squeeze(-1)
                    per_ex_loss = loss_fn(logits, y)
                    all_losses.append(per_ex_loss)
                    all_groups.append(g.to(device))

                all_losses = torch.cat(all_losses)
                all_groups = torch.cat(all_groups)

                group_rel_imps = np.zeros(num_groups, dtype=np.float32)
                for gid in range(num_groups):
                    if valid_groups[gid]:
                        mask = (all_groups == gid)
                        if mask.any().item():
                            group_loss = all_losses[mask].mean().item()
                            group_rel_imps[gid] = (baseline_losses[gid] - group_loss) / gaps[gid]

            # ===== Update group weights via exponentiated gradient =====
            # Use NEGATIVE rel_imp because we want to give more weight to groups with LOWER rel_imp
            # Normalize RI values to make updates more stable (mean-centered)
            valid_ri = group_rel_imps[valid_groups]
            if len(valid_ri) > 0 and valid_ri.std() > 1e-8:
                ri_mean = valid_ri.mean()
                centered_ri = group_rel_imps - ri_mean
                adaptive_eta = _adaptive_eta_from_ri(group_weight_step_size, valid_ri)
                group_weights = group_weights * np.exp(-adaptive_eta * centered_ri)
            else:
                # Fallback to original method if all RI are equal
                adaptive_eta = _adaptive_eta_from_ri(group_weight_step_size, group_rel_imps)
                group_weights = group_weights * np.exp(-adaptive_eta * group_rel_imps)
            group_weights = group_weights / group_weights.sum()

            # ===== Evaluation and logging =====
            train_mean, train_group, train_worst = evaluate(model, train_loader, num_groups, device)
            val_mean, val_group, val_worst = evaluate(model, val_loader, num_groups, device)

            train_ri = [(baseline_losses[g] - train_group[g]) / gaps[g] if valid_groups[g] else float('nan')
                       for g in range(num_groups)]
            val_ri = [(baseline_losses[g] - val_group[g]) / gaps[g] if valid_groups[g] else float('nan')
                     for g in range(num_groups)]

            if verbose:
                print(f"  Epoch {global_epoch}/{epochs}: "
                      f"train={train_mean:.4f}(RI={np.nanmin(train_ri):.4f}), "
                      f"val={val_mean:.4f}(RI={np.nanmin(val_ri):.4f})")

                # Print detailed per-group information for all group counts
                train_ri_str = ", ".join([f"G{g}={train_ri[g]:.4f}" if not np.isnan(train_ri[g]) else f"G{g}=N/A" for g in range(num_groups)])
                val_ri_str = ", ".join([f"G{g}={val_ri[g]:.4f}" if not np.isnan(val_ri[g]) else f"G{g}=N/A" for g in range(num_groups)])
                weights_str = ", ".join([f"G{g}={group_weights[g]:.4f}" for g in range(num_groups)])
                print(f"    Train RI: {train_ri_str}")
                print(f"    Val RI:   {val_ri_str}")
                print(f"    Weights: {weights_str}")

            if save_best and np.nanmin(val_ri) > best_val_worst_ri:
                best_val_worst_ri = np.nanmin(val_ri)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Early stopping check
            current_metric = np.nanmin(val_ri)  # RI objective: maximize worst-group RI
            if current_metric - best_metric > min_delta:
                best_metric = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\n  Early stopping triggered at epoch {global_epoch}")
                    print(f"  No improvement for {patience} epochs (best={best_metric:.4f})")
                break

    # ========== Restore best model if requested ==========
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"\nRestored best model with val worst relative improvement: {best_val_worst_ri:.4f}")

    return model


def compute_baseline_losses(y_train, g_train, num_groups):
    """
    Compute baseline RMSE for constant predictor f_0(x) = mean(y).
    """
    baseline_losses = np.zeros(num_groups, dtype=np.float32)

    mu_0 = float(y_train.mean())

    for gid in range(num_groups):
        mask = (g_train == gid)
        y_g = y_train[mask]

        if len(y_g) == 0:
            baseline_losses[gid] = float("nan")
            continue

        mse = float(np.mean((y_g - mu_0) ** 2))
        baseline_losses[gid] = np.sqrt(mse)

    return baseline_losses


def solve_weighted_groups_closed_form(X, y, g, num_groups, lambda_weight):
    """
    Closed-form weighted least squares for group-weighted objective:
        min_β  (1-λ)/n_0 * ||X_0 β - y_0||² + λ/n_1 * ||X_1 β - y_1||²

    Equivalent to WLS: β* = (X^T W X)^{-1} X^T W y
    where w_i = λ_g / n_g for sample i in group g.

    Args:
        X: (n, d) numpy array (already scaled)
        y: (n,) numpy array
        g: (n,) numpy array of group labels (integers)
        num_groups: number of groups
        lambda_weight: float in [0, 1]. For 2 groups: group weights are (1-λ, λ).

    Returns:
        coef: (d,) coefficients
        intercept: float
        group_rmse: list of per-group RMSE
    """
    n = len(y)

    # Build per-sample weights
    if num_groups == 2:
        group_lambdas = np.array([1.0 - lambda_weight, lambda_weight], dtype=np.float64)
    else:
        group_lambdas = np.ones(num_groups, dtype=np.float64) / num_groups

    group_counts = np.zeros(num_groups, dtype=np.float64)
    for gid in range(num_groups):
        group_counts[gid] = max((g == gid).sum(), 1)

    w = np.zeros(n, dtype=np.float64)
    for gid in range(num_groups):
        mask = (g == gid)
        w[mask] = group_lambdas[gid] / group_counts[gid]

    # Add intercept column
    X_aug = np.column_stack([X, np.ones(n, dtype=np.float64)])

    # WLS: β = (X^T W X)^{-1} X^T W y
    W = w  # (n,) vector, use broadcasting
    XtW = X_aug.T * W[np.newaxis, :]  # (d+1, n)
    XtWX = XtW @ X_aug  # (d+1, d+1)
    XtWy = XtW @ y  # (d+1,)

    # Solve via Cholesky or lstsq for numerical stability
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)

    coef = beta[:-1]
    intercept = beta[-1]

    # Compute per-group RMSE
    preds = X @ coef + intercept
    group_rmse = np.zeros(num_groups, dtype=np.float32)
    for gid in range(num_groups):
        mask = (g == gid)
        if mask.any():
            group_rmse[gid] = float(np.sqrt(np.mean((preds[mask] - y[mask]) ** 2)))

    return coef, intercept, group_rmse


def train_weighted_groups(model, train_loader, num_groups, lambda_weight,
                          device="cpu", lr=1e-3, epochs=30, verbose=False,
                          warm_start_model=None):
    """
    Train with weighted group objective: (1-λ) * L_g0 + λ * L_g1
    Gradient descent fallback for non-linear models.
    For linear models, use solve_weighted_groups_closed_form() instead.
    """
    model.to(device)

    if warm_start_model is not None:
        model.load_state_dict(warm_start_model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X, y, g in train_loader:
            X = X.to(device)
            y = y.to(device)
            g = g.to(device)

            logits = model(X).squeeze(-1)
            per_ex_loss = loss_fn(logits, y)

            group_losses = []
            for gid in range(num_groups):
                mask = (g == gid)
                if mask.any():
                    group_losses.append(per_ex_loss[mask].mean())
                else:
                    group_losses.append(torch.tensor(0.0, device=device))

            if num_groups == 2:
                weighted_loss = (1 - lambda_weight) * group_losses[0] + lambda_weight * group_losses[1]
            else:
                weighted_loss = sum(group_losses) / len(group_losses)

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            n_batches += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  λ={lambda_weight:.3f}, Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

    return model
