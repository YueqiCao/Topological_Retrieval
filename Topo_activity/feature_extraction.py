import torch
import torchvision
from data.dataloader import VideoDataset
from tqdm import tqdm
from utils import get_dataloader


def run_feature_extraction(args):
    # Get Dataloaders
    train_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                 args.class_idx_path, window=args.window, mode='training')
    train_loader = get_dataloader(train_dataset,False)

    val_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                 args.class_idx_path, window=args.window, mode='validation')
    val_loader =  get_dataloader(val_dataset,False)

    # Get Feature Extractor
    feature_extractor = torchvision.models.resnet18(pretrained=True, progress=True)
    feature_extractor.to(args.device)
    feature_extractor.eval()


    with torch.no_grad():
        to_output_train = {}
        for (i, (frames,label,videoid)) in tqdm(enumerate(train_loader),total=len(train_dataset)):
            frames = frames.to(args.device).view(-1,224,224)
            print(frames.shape)
            pred = feature_extractor(frames)
            print(pred.shape)
            pred = torch.mean(pred,dim=0)
            print(pred.shape)
            a=k
            to_output_train[i] = { 'features' : pred.data.cpu().numpy(),
                             'label' : label,
                             'videoid':videoid

            }
        torch.save(to_output_train,'./train_features_window_10.pt')

        to_output_val = {}
        for (i, (frames,label,videoid)) in tqdm(enumerate(val_loader),total=len(val_dataset)):
            frames = frames.to(args.device).view(-1,224,224)
            pred = feature_extractor(frames)
            to_output_val[i] = { 'features' : pred.data.cpu().numpy(),
                             'label' : label,
                             'videoid':videoid

            }
        torch.save(to_output_val,'./val_features_window_10.pt')
