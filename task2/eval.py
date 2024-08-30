import torch
from model import vit_base_patch16, vit_large_patch16, vit_huge_patch14
from utils import create_cacheds_dl, Dataset_memm
from utils import Monai2DTransforms, Resize_cv2
from torchvision import transforms
from utils import  initialize
from model import Models
import cv2
import pandas as pd
from utils import mae_args_parser
from utils import load_last_k

initialize(allow_tf32=False)

args = mae_args_parser()

def train(args):
    merged_df = pd.DataFrame()

    df = pd.read_csv(args.data_dir+'df_task2_val_challenge.csv')
    df['image']=args.data_dir+'data_task2/val/'+df['image']

    NORM = []
    if args.norm and args.in_ch==3: NORM=[[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
    elif args.norm and args.in_ch==1: NORM = [[0.20253482627511976], [0.11396578943414482]]

    transf_list= [Resize_cv2((224, 224),interpolation = cv2.INTER_AREA),  
                                    transforms.ConvertImageDtype(torch.float32)]
    if args.norm: transf_list.append(transforms.Normalize(*NORM))
    test_transform = transforms.Compose(transf_list)#,
    test_transform = Monai2DTransforms(test_transform)

    ds_test_scan = Dataset_memm(df, grayscale=args.in_ch==1,challange_eval=True)

    testloader = create_cacheds_dl(ds_test_scan, test_transform, shuffle=True, num_workers=8,  cache_rate=0.0)
    
    for fold in range(3):
        save_dir = args.save_dir+'/fold_'+str(fold)

        if "vit" in args.backbone:
            model = Models[args.backbone](in_chans=args.in_ch, num_classes=args.n_cl, global_pool=args.global_pool)
        else:
            model  = Models[args.backbone](args.in_ch, args.n_cl)

        model.cuda()

        val_preds = []
        val_cases = []
        device = torch.device("cuda" if torch.cuda.is_available() 
                                    else "cpu")
        saved_data = load_last_k(save_dir, device)

        if saved_data:  
            # Extract data
            msg = model.load_state_dict(saved_data, strict=True)
            assert set(msg.missing_keys) == set()

        with torch.no_grad():
            model.eval()
            for j, inp_dict in enumerate(testloader,0):

                inputs = inp_dict["image"]

                inputs = inputs.cuda()

                outs_vals = model.forward(inputs)
                
                predicted =  outs_vals.argmax(dim=1, keepdim=True)
                val_preds += predicted.data.reshape(-1).cpu()
                val_cases += inp_dict['case'].data.cpu()

        df_pred = pd.DataFrame({'case':[int(x) for x in val_cases], 'predictio_n'+str(fold):[int(x) for x in val_preds]})
        if merged_df.empty:
            merged_df = df_pred
        else:
            merged_df = pd.merge(merged_df, df_pred, on='case', how='outer')
    
    merged_df['prediction'] = merged_df[['prediction_0','prediction_1','prediction_2']].mode(axis=1)[0] # Get the mode of the predictions for majority voting, it will be refined later
    merged_df.to_csv(args.save_dir+'/predictions.csv', index=False)

def main():
    train(args)

if __name__ == "__main__":
    main()
