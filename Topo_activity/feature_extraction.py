import torch
import torchvision
from data.dataloader import VideoDataset
from tqdm import tqdm
from utils import get_dataloader


def run_feature_extraction(args):
    # Get Dataloaders

    args.bath_size =1
    train_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                 args.class_idx_path, args= args,window=args.window, mode='training')
    train_loader = get_dataloader(train_dataset,False,workers=args.workers,batch_size=args.bath_size)

    val_dataset = VideoDataset(args.json_path, args.video_path, args.csv_path,
                                 args.class_idx_path,args= args, window=args.window, mode='validation')
    val_loader =  get_dataloader(val_dataset,False,workers=args.workers,batch_size=args.bath_size)

    # Get Feature Extractor
    feature_extractor = torchvision.models.resnet34(pretrained=True, progress=True)
    feature_extractor.fc = torch.nn.Identity()
    feature_extractor.to(args.device)
    feature_extractor.eval()


    with torch.no_grad():
        to_output_train = {}
        for (i, (frames,target, label, videoid)) in tqdm(enumerate(train_loader),total=len(train_dataset)):
            frames = frames.to(args.device).float()
            pred = [feature_extractor(frames[:,jj,:,:,:]) for jj in range(frames.shape[1])]
            pred = torch.stack(pred,1).mean(1)
            pred = torch.mean(pred,dim=0)

            to_output_train[i] = { 'features' : pred.data.cpu().numpy(),
                                    'label' : label,
                                   'target': target,
                                     'videoid':videoid

            }
        torch.save(to_output_train,'./experiments/train_features_window_{}_res34.pt'.format(args.window))

        to_output_val = {}
        for (i, (frames, target, label, videoid)) in tqdm(enumerate(val_loader), total=len(val_dataset)):
            frames = frames.to(args.device).float()
            pred = [feature_extractor(frames[:, jj, :, :, :]) for jj in range(frames.shape[1])]
            pred = torch.stack(pred, 1).mean(1)
            pred = torch.mean(pred, dim=0)

            to_output_val[i] = {'features': pred.data.cpu().numpy(),
                                  'label': label,
                                  'target': target,
                                  'videoid': videoid

                                  }
        torch.save(to_output_val,'./experiments/val_features_window_{}_res34.pt'.format(args.window))
