"""
MTGR Training Script

This script trains the MTGR model with:
- Listwise ranking loss (first candidate is positive)
- Evaluation metrics: NDCG@K, Hit Rate@K, MRR
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
"""

import argparse
import logging
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchrec.modules.embedding_configs import EmbeddingConfig, DataType

from generative_recommenders.dlrm_v3.datasets.beauty_mtgr import MTGRBeautyDataset
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, DlrmHSTU
from generative_recommenders.modules.multitask_module import MultitaskTaskType, TaskConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mtgr_config(dataset: MTGRBeautyDataset, args: argparse.Namespace) -> DlrmHSTUConfig:
    """Create MTGR configuration."""
    config = DlrmHSTUConfig(
        # MTGR-specific settings
        use_mtgr_tokenization=True,
        mtgr_user_feature_names=dataset._uih_keys[:3],  # First 3 are user features
        mtgr_seq_feature_names=dataset._uih_keys[3:],   # Rest are sequence features
        mtgr_cross_feature_names=[dataset._candidates_keys[0]],  # brand_count
        mtgr_candidate_feature_names=dataset._candidates_keys[1:],  # Item features

        # Model architecture
        hstu_embedding_table_dim=args.embedding_dim,
        hstu_transducer_embedding_dim=args.d_model,
        hstu_num_heads=args.num_heads,
        hstu_attn_linear_dim=args.d_model,
        hstu_attn_qk_dim=args.d_model // args.num_heads,
        mtgr_num_layers=args.num_layers,

        # Required fields (not used in MTGR mode)
        uih_post_id_feature_name='seq_item_id',
        uih_action_time_feature_name='',
        uih_weight_feature_name='',
        candidates_weight_feature_name='',
        candidates_watchtime_feature_name='',
        candidates_querytime_feature_name='',
    )

    # Add multitask config
    config.multitask_configs = [
        TaskConfig(
            task_name="recommendation",
            task_weight=1.0,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        )
    ]

    return config


def create_embedding_tables(config: DlrmHSTUConfig, dataset: MTGRBeautyDataset) -> Dict[str, EmbeddingConfig]:
    """Create embedding tables for all MTGR features."""
    tables = {}

    # Get vocab sizes from dataset
    max_user_id = max(int(user['user_id']) for user in dataset.users)
    max_item_id = max(dataset.all_item_ids)

    logger.info(f"Embedding vocab sizes: user_id={max_user_id + 1}, item_id={max_item_id + 1}")

    # User features
    for feature_name in config.mtgr_user_feature_names:
        if feature_name == 'user_id':
            num_emb = max_user_id + 1
        elif 'rating' in feature_name:
            num_emb = 51  # 0-50 (ratings * 10)
        elif 'review' in feature_name:
            num_emb = 5000
        else:
            num_emb = 1000

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    # Sequence features
    for feature_name in config.mtgr_seq_feature_names:
        if 'item_id' in feature_name or 'item' in feature_name:
            num_emb = max_item_id + 1
        elif 'category' in feature_name or 'brand' in feature_name:
            num_emb = 10000
        elif 'price' in feature_name:
            num_emb = 100000
        else:
            num_emb = 10000

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    # Cross features
    for feature_name in config.mtgr_cross_feature_names:
        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=200,  # Brand count
            data_type=DataType.FP32,
        )

    # Candidate features
    for feature_name in config.mtgr_candidate_feature_names:
        if 'item_id' in feature_name or 'item' in feature_name:
            num_emb = max_item_id + 1
        elif 'category' in feature_name or 'brand' in feature_name:
            num_emb = 10000
        elif 'price' in feature_name:
            num_emb = 100000
        else:
            num_emb = 10000

        tables[feature_name] = EmbeddingConfig(
            name=feature_name,
            embedding_dim=config.hstu_embedding_table_dim,
            num_embeddings=num_emb,
            data_type=DataType.FP32,
        )

    return tables


class MTGRRankingLoss(nn.Module):
    """
    MTGR Ranking Loss (Listwise Cross-Entropy).

    Assumes first candidate is positive, rest are negatives.
    Uses softmax cross-entropy to maximize the score of the positive item.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        num_candidates: int,
    ) -> torch.Tensor:
        """
        Compute listwise ranking loss.

        Args:
            user_emb: [batch * num_candidates, d_model] - User representations
            item_emb: [batch * num_candidates, d_model] - Item representations
            num_candidates: Number of candidates per sample (1 positive + K-1 negatives)

        Returns:
            loss: Scalar loss value
        """
        batch_size = user_emb.shape[0] // num_candidates

        # Reshape to [batch, num_candidates, d_model]
        user_emb = user_emb.view(batch_size, num_candidates, -1)
        item_emb = item_emb.view(batch_size, num_candidates, -1)

        # Compute scores: dot product between user and item embeddings
        # [batch, num_candidates, d_model] * [batch, num_candidates, d_model]
        # → [batch, num_candidates]
        scores = (user_emb * item_emb).sum(dim=-1) / self.temperature

        # Create labels: first candidate (index 0) is positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=scores.device)

        # Softmax cross-entropy loss
        # This encourages the positive item (index 0) to have the highest score
        loss = nn.functional.cross_entropy(scores, labels)

        return loss


def compute_metrics(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    num_candidates: int,
    k_list: list = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute ranking metrics: NDCG@K, Hit Rate@K, MRR.

    Args:
        user_emb: [batch * num_candidates, d_model]
        item_emb: [batch * num_candidates, d_model]
        num_candidates: Number of candidates per sample
        k_list: List of K values for metrics

    Returns:
        metrics: Dictionary of metric values
    """
    batch_size = user_emb.shape[0] // num_candidates

    # Reshape
    user_emb = user_emb.view(batch_size, num_candidates, -1)
    item_emb = item_emb.view(batch_size, num_candidates, -1)

    # Compute scores
    scores = (user_emb * item_emb).sum(dim=-1)  # [batch, num_candidates]

    # Rank candidates (argsort descending)
    rankings = torch.argsort(scores, dim=1, descending=True)  # [batch, num_candidates]

    # Positive item is at index 0
    positive_idx = torch.zeros(batch_size, dtype=torch.long, device=rankings.device)

    # Compute metrics
    metrics = {}

    # MRR (Mean Reciprocal Rank)
    positive_ranks = (rankings == positive_idx.unsqueeze(1)).nonzero(as_tuple=True)[1]
    reciprocal_ranks = 1.0 / (positive_ranks.float() + 1.0)
    metrics['MRR'] = reciprocal_ranks.mean().item()

    for k in k_list:
        # Hit Rate @ K
        hits = (rankings[:, :k] == positive_idx.unsqueeze(1)).any(dim=1).float()
        metrics[f'Hit@{k}'] = hits.mean().item()

        # NDCG @ K (simplified: binary relevance)
        # DCG: 1 / log2(rank + 2) if positive in top-k, else 0
        # IDCG: 1 / log2(1 + 2) = 1.0 (perfect ranking)
        positive_in_topk = (rankings[:, :k] == positive_idx.unsqueeze(1)).any(dim=1)
        ranks_in_topk = (rankings[:, :k] == positive_idx.unsqueeze(1)).nonzero(as_tuple=True)[1]

        dcg = torch.zeros(batch_size, device=rankings.device)
        dcg[positive_in_topk] = 1.0 / torch.log2(ranks_in_topk.float() + 2.0)

        idcg = 1.0  # Perfect ranking (positive at rank 0)
        ndcg = dcg / idcg

        metrics[f'NDCG@{k}'] = ndcg.mean().item()

    return metrics


def train_epoch(
    model: DlrmHSTU,
    dataloader: DataLoader,
    criterion: MTGRRankingLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_candidates: int,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch.to(device)

        # Forward pass
        user_emb, item_emb, aux_losses, mt_preds, mt_labels, mt_weights = model(
            batch.uih_features_kjt,
            batch.candidates_features_kjt
        )

        # Compute ranking loss
        loss = criterion(user_emb, item_emb, num_candidates)

        # Add auxiliary losses if any
        for aux_name, aux_loss in aux_losses.items():
            loss = loss + aux_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

            # TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)

    avg_loss = total_loss / num_batches
    return {'loss': avg_loss}


def validate(
    model: DlrmHSTU,
    dataloader: DataLoader,
    criterion: MTGRRankingLoss,
    device: torch.device,
    num_candidates: int,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_metrics = {
        'MRR': 0.0,
        'Hit@1': 0.0, 'Hit@5': 0.0, 'Hit@10': 0.0,
        'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch.to(device)

            # Forward pass
            user_emb, item_emb, aux_losses, mt_preds, mt_labels, mt_weights = model(
                batch.uih_features_kjt,
                batch.candidates_features_kjt
            )

            # Compute loss
            loss = criterion(user_emb, item_emb, num_candidates)
            for aux_name, aux_loss in aux_losses.items():
                loss = loss + aux_loss

            total_loss += loss.item()

            # Compute metrics
            batch_metrics = compute_metrics(user_emb, item_emb, num_candidates)
            for metric_name, metric_value in batch_metrics.items():
                all_metrics[metric_name] += metric_value

            num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    for metric_name in all_metrics:
        all_metrics[metric_name] /= num_batches

    all_metrics['loss'] = avg_loss
    return all_metrics


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = MTGRBeautyDataset(
        data_file=args.train_data,
        item_features_file=args.item_features,
        num_negatives=args.num_negatives,
        max_seq_len=args.max_seq_len,
        is_inference=False,
    )

    val_dataset = MTGRBeautyDataset(
        data_file=args.val_data,
        item_features_file=args.item_features,
        num_negatives=args.num_negatives,
        max_seq_len=args.max_seq_len,
        is_inference=True,  # No random slicing for validation
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    logger.info("Creating MTGR model...")
    config = create_mtgr_config(train_dataset, args)
    embedding_tables = create_embedding_tables(config, train_dataset)

    model = DlrmHSTU(
        hstu_configs=config,
        embedding_tables=embedding_tables,
        is_inference=False,
    )

    # Move to device (use to_empty for meta device, then reset params)
    model = model.to_empty(device=device)
    model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create loss, optimizer, scheduler
    criterion = MTGRRankingLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Training loop
    logger.info("Starting training...")
    best_ndcg = 0.0
    num_candidates = 1 + args.num_negatives

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*80}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, num_candidates, epoch, writer
        )
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, num_candidates
        )
        logger.info(f"Validation Metrics:")
        logger.info(f"  Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  MRR: {val_metrics['MRR']:.4f}")
        logger.info(f"  Hit@1: {val_metrics['Hit@1']:.4f} | Hit@5: {val_metrics['Hit@5']:.4f} | Hit@10: {val_metrics['Hit@10']:.4f}")
        logger.info(f"  NDCG@1: {val_metrics['NDCG@1']:.4f} | NDCG@5: {val_metrics['NDCG@5']:.4f} | NDCG@10: {val_metrics['NDCG@10']:.4f}")

        # TensorBoard
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/MRR', val_metrics['MRR'], epoch)
        writer.add_scalar('Val/NDCG@5', val_metrics['NDCG@5'], epoch)
        writer.add_scalar('Val/Hit@5', val_metrics['Hit@5'], epoch)

        # Save best model
        if val_metrics['NDCG@5'] > best_ndcg:
            best_ndcg = val_metrics['NDCG@5']
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (NDCG@5={best_ndcg:.4f}) to {checkpoint_path}")

        # Learning rate scheduling
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': val_metrics,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    logger.info("\n" + "="*80)
    logger.info("Training complete!")
    logger.info(f"Best NDCG@5: {best_ndcg:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MTGR model")

    # Data
    parser.add_argument("--train-data", type=str, default="data/Beauty_train.pkl")
    parser.add_argument("--val-data", type=str, default="data/Beauty_train.pkl")  # TODO: split train/val
    parser.add_argument("--item-features", type=str, default="data/Beauty_item_feature_map.pkl")

    # Model
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=100)

    # System
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save-every", type=int, default=5)

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    main(args)
