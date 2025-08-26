import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Callable, List
from accelerate import Accelerator
from sklearn.linear_model import LinearRegression


class eval_mode:
    def __init__(self, *models, no_grad=False):
        self.models = models
        self.no_grad = no_grad
        self.no_grad_context = torch.no_grad()

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)
        if self.no_grad:
            self.no_grad_context.__enter__()

    def __exit__(self, *args):
        if self.no_grad:
            self.no_grad_context.__exit__(*args)
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def embed_trajectory_dataset(
    model,
    dataset,
    obs_only=True,
    device=None,
    embed_goal=False,
):
    if type(model) is nn.parallel.DistributedDataParallel:
        return embed_trajectory_dataset_ddp(
            model,
            dataset,
            obs_only=obs_only,
            device=device,
            embed_goal=embed_goal,
        )
    else:
        result = []
        accelerator = Accelerator()
        device = device or accelerator.device  # result device
        with eval_mode(model, no_grad=True):
            for i in range(len(dataset)):
                obs, *rest = dataset[i]
                obs = obs.to(accelerator.device)
                obs_enc = model(obs).to(device)
                if obs_only:
                    result.append(obs_enc)
                else:
                    if embed_goal:
                        # assuming goal comes last
                        goal = rest[-1]
                        rest = rest[:-1]
                        goal = goal.to(accelerator.device)
                        goal_enc = model(goal).to(device)
                        rest.append(goal_enc)
                    rest = [x.to(device) for x in rest]
                    result.append((obs_enc, *rest))
        return result


def embed_trajectory_dataset_ddp(
    model: nn.Module,
    dataset,
    obs_only=True,
    device=None,
    embed_goal=False,
):
    assert type(model) is nn.parallel.DistributedDataParallel, "Model must be DDP"
    embeddings = []
    accelerator = Accelerator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
    )
    dataloader = accelerator.prepare(dataloader)
    # get the max trajectory length, so that we can pad tensors for DDP gather
    max_T = max(dataset.get_seq_length(i) for i in range(len(dataset)))
    with eval_mode(model, no_grad=True):
        for obs, *rest in dataloader:
            obs = obs.to(accelerator.device)  # obs shape 1 T V C H W
            obs_enc = model(obs)
            obs_enc = pad_to_length(obs_enc, max_T, dim=1)
            obs_enc = accelerator.gather_for_metrics(obs_enc)
            if obs_only:
                embeddings.append(obs_enc)
            else:
                if embed_goal:
                    # assuming goal comes last
                    goal = rest[-1]
                    rest = rest[:-1]
                    goal = goal.to(accelerator.device)
                    goal_enc = model(goal)
                    rest.append(goal_enc)
                rest = [x.to(accelerator.device) for x in rest]
                rest = [pad_to_length(x, max_T, dim=1) for x in rest]
                rest = [accelerator.gather_for_metrics(x) for x in rest]
                embeddings.append((obs_enc, *rest))

    device = device or accelerator.device
    # unpad the tensors
    result = []
    if obs_only:
        embeddings = torch.cat(embeddings, dim=0)
        assert len(embeddings) == len(dataset)
    else:
        embeddings = [torch.cat(x, dim=0) for x in zip(*embeddings)]
        assert len(embeddings[0]) == len(dataset)
    for i in range(len(dataset)):
        T = dataset.get_seq_length(i)
        if obs_only:
            result.append(embeddings[i, :T].to(device))
        else:
            result.append([x[i, :T].to(device) for x in embeddings])
    return result


def pad_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """
    Pad tensor x to length along dim, adding zeros at the end.
    """
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    pad = torch.zeros(
        *x.shape[:dim],
        pad_size,
        *x.shape[dim + 1 :],
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=dim)


def repeat_start_to_length(x: torch.Tensor, length: int, dim: int = 0):
    """
    Pad tensor x to length along dim, repeating the first value at the start.
    """
    pad_size = length - x.shape[dim]
    if pad_size <= 0:
        return x
    first_frame = x.index_select(dim, torch.tensor(0, device=x.device))
    repeat_shape = [1] * len(x.shape)
    repeat_shape[dim] = pad_size
    pad = first_frame.repeat(*repeat_shape)
    return torch.cat([pad, x], dim=dim)


def nn_lookup(
    query: torch.Tensor,
    pool: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    pairwise_query = query.repeat_interleave(len(pool), dim=0)
    pairwise_pool = pool.repeat((len(query), 1))
    dist = metric(pairwise_query, pairwise_pool)
    nn_dist, nn_idx = dist.view(len(query), len(pool)).sort(dim=1)
    return nn_dist, nn_idx


def batch_knn(
    query: torch.Tensor,
    pool: torch.Tensor,
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    k: int,
    batch_size: int,
):
    """
    Return the k nearest neighbors of query in pool using metric.
    Input:
        query: Tensor[N, D] of query points
        pool: Tensor[M, D] of pool points
        metric: Callable[[Tensor[N, D], Tensor[M, D]], Tensor[N, M]] distance function
        k: int number of neighbors to return
        batch_size: int batch size for computation. Batched over query.
    Output: (distances, indices)
        distances: Tensor[N, k] of distances to the k nearest neighbors
        indices: Tensor[N, k] of indices of the k nearest neighbors
    """
    nn_dists = []
    nn_idxs = []
    for i in range(0, len(query), batch_size):
        batch = query[i : i + batch_size].to(pool.device)
        nn_dist, nn_idx = nn_lookup(batch, pool, metric)
        nn_dists.append(nn_dist[:, :k])
        nn_idxs.append(nn_idx[:, :k])
    return torch.cat(nn_dists), torch.cat(nn_idxs)


def linear_probe_with_trajectory_split(
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: List[int],
    val_idx: List[int],
):
    X_train = torch.cat([X[i] for i in train_idx]).cpu().numpy()
    y_train = torch.cat([y[i] for i in train_idx]).cpu().numpy()
    X_val = torch.cat([X[i] for i in val_idx]).cpu().numpy()
    y_val = torch.cat([y[i] for i in val_idx]).cpu().numpy()

    X_all = torch.cat(X).cpu().numpy()
    y_all = torch.cat(y).cpu().numpy()

    m = LinearRegression()
    # all -> train
    m.fit(X_all, y_all)
    linear_probe_mse_train_all = np.mean((m.predict(X_train) - y_train) ** 2).item()
    # all -> val
    linear_probe_mse_val_all = np.mean((m.predict(X_val) - y_val) ** 2).item()
    return {
        "linear_probe_mse_train_all": linear_probe_mse_train_all,
        "linear_probe_mse_val_all": linear_probe_mse_val_all,
    }


def mse(a: torch.Tensor, b: torch.Tensor):
    return ((a - b) ** 2).mean(dim=1)


def mahalanobis(a, b, VI):
    u = a - b
    v = u @ VI  # (V^{-1} @ (a - b).T).T
    return (u * v).sum(dim=-1).sqrt()  # sqrt of dot product for each row


class OLS:
    """
    OLS in torch
    NOTE: discrepancy with sklearn's LinearRegression when ill-conditioned; reverting to sklearn for now
    """

    def __init__(self, bias=True, fallback_to_cpu=True):
        self.bias = bias
        self.w = None
        self.fallback_to_cpu = fallback_to_cpu

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the model
        """
        if self.bias:
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        self.w = torch.linalg.lstsq(X, y).solution
        if torch.isnan(self.w).any():
            cond = torch.linalg.cond(X)
            rank = torch.linalg.matrix_rank(X)
            msg = f"NaNs in OLS solution. Input shape: {X.shape}, cond: {cond}, rank: {rank}"
            if not self.fallback_to_cpu:
                raise ValueError(msg)
            logging.warn(f"{msg}; Falling back to CPU with gelss driver.")
            self.w = torch.linalg.lstsq(X.cpu(), y.cpu(), driver="gelss").solution
            self.w = self.w.to(X.device)
        return self

    def predict(self, X: torch.Tensor):
        """
        Predict the output
        """
        if self.w is None:
            raise ValueError("Model not fitted")
        if self.bias:
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
        return X @ self.w


class SGDClassifier:
    def __init__(self, lr=1e-4, max_iter=1000, tol=1e-3, batch_size=2048):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        n_samples, input_dim = X.shape
        n_classes = y.max().item() + 1
        self.linear = nn.Linear(input_dim, n_classes).to(X.device)
        optimizer = torch.optim.AdamW(
            self.linear.parameters(), lr=self.lr, weight_decay=0.0
        )
        criterion = nn.CrossEntropyLoss()
        for j in range(self.max_iter):
            total_loss = 0
            n_batches = 0
            indices = torch.randperm(n_samples).to(X.device)
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch_X, batch_y = X[batch_indices], y[batch_indices]
                optimizer.zero_grad()
                logits = self.linear(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            avg_loss = total_loss / n_batches
            if avg_loss < self.tol:
                break
        if j + 1 < self.max_iter:
            logging.info(f"Converged at epoch {j+1}.")
        else:
            logging.info(f"Max iter reached. Final loss {avg_loss}")
        return self

    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            return torch.argmax(self.linear(X), dim=1)

    def score(self, X: torch.Tensor, y: torch.Tensor):
        return (self.predict(X) == y).float().mean().item()
