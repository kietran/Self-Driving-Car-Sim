from dataset import Driving_Dataset
from src.model import CNN
import torch
import argparse
import os
from torchvision.transforms import v2
import numpy as np
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser =argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default='dataset/augmented_driving_log.csv')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3)
    parser.add_argument('--checkpoint_path', '-c', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_load', '-z', type=str, default=None)
    parser.add_argument('--tensorboard_dir', '-t', type=str, default='tensorboard')
    args = parser.parse_args()
    return args

def train(args):
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.isdir(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)

    writer = SummaryWriter(args.tensorboard_dir)
    transform = v2.Compose([
        v2.ToImage(),
        # v2.Resize((66, 200), antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # learning_rate_schedule = {"0": 0.1, "25": 3e-3}
    
    # Train test split
    train_ratio = 0.8

    train_dataset = Driving_Dataset(
        dataset_path=args.dataset_path, 
        transform=transform, 
        is_train=True,
        train_ratio=train_ratio, 
        random_seed=42)
    val_dataset = Driving_Dataset(
        dataset_path=args.dataset_path, 
        transform=transform, 
        is_train=False,
        train_ratio=train_ratio, 
        random_seed=42)

    print(f'Number of train images: {len(train_dataset)}')
    print(f'Number of valid images: {len(val_dataset)}')

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN()
    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criteria = nn.MSELoss()

    # Load checkpoint if requested
    if args.checkpoint_load and os.path.isfile(args.checkpoint_load):
        checkpoint = torch.load(args.checkpoint)
        model = model.load_state_dict(checkpoint['model_params'])
        optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        best_loss = np.inf

    num_iters = len(train_dataloader)
    # Train and validation
    model.train()
    for epoch in range(start_epoch, args.epochs):
        # if str(epoch) in learning_rate_schedule.keys():
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = learning_rate_schedule[str(epoch)]

        progress_bar = tqdm(train_dataloader, colour='cyan')
        train_losses = []
        for iter, (images, steerings) in enumerate(progress_bar):
            images = images.to(device)
            steerings = steerings.to(device)
            
            # Forward
            prediction = model(images)
            loss = criteria(prediction, steerings)
            train_losses.append(loss.item())
            loss_val = np.mean(train_losses)
            progress_bar.set_description(f"Train: Epoch {epoch}/{args.epochs}. Loss: {loss_val}")
            writer.add_scalar('Train/Loss', loss_val, epoch*num_iters+iter)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        
        val_losses = []
        for iter, (images, steerings) in enumerate(val_dataloader):
            with torch.inference_mode():
                images = images.to(device)
                steerings = steerings.to(device)

                prediction = model(images)
                loss = criteria(prediction, steerings)
                val_losses.append(loss.item())

        loss_val = np.mean(val_losses)
        writer.add_scalar('Val/Loss', loss_val, epoch)
        print(f"Val: Epoch {epoch}/{args.epochs}. Loss: {loss_val}")

        # Save checkpoint
        checkpoint = {
            'model_params': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, 'last.pt'))
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'best.pt'))

if __name__ == "__main__":
    args = get_args()
    train(args)