from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import argparse
import os
from train import BertDataset
from eval import evaluate
from model.contrast import ContrastModel
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--name', type=str, default='nyt-past01', help='Name of checkpoint. Commonly as DATASET-NAME.')
parser.add_argument('--extra', default='_macro', choices=['_micro', '_micro'],help='An extra string in the name of checkpoint.')
args = parser.parse_args()

if __name__ == '__main__':

    checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(args.extra)),
                            map_location='cpu')
    batch_size = args.batch
    device = args.device
    extra = args.extra
    args = checkpoint['args'] if checkpoint['args'] is not None else args
    data_path = os.path.join('data', args.data)

    if not hasattr(args, 'graph'):
        args.graph = True
    print(args)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
    label_dict = {i: tokenizer.decode(v, skip_special_tokens=True) for i, v in label_dict.items()}
    num_class = len(label_dict)

    dataset = BertDataset(device=device, pad_idx=tokenizer.pad_token_id, data_path=data_path)
    model = ContrastModel.from_pretrained('bert-base-uncased', num_labels=num_class,
                                          contrast_loss=args.contrast, graph=args.graph,
                                          layer=args.layer, data_path=data_path, multi_label=args.multi,
                                          lamb=args.lamb, threshold=args.thre)
    split = torch.load(os.path.join(data_path, 'split.pt'))
    test = Subset(dataset, split['test'])
    test = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model.load_state_dict(checkpoint['param'])
    model.to(device)

    truth = []
    pred = []
    model.eval()


    # test
    pbar = tqdm(test)
    with torch.no_grad():
        for data, label, idx in pbar:
            padding_mask = data != tokenizer.pad_token_id
            output = model(data, padding_mask, return_dict=True, return_pooled_output=True, )

            for l in label:  # 获得样本真实标签
                t = []
                for i in range(l.size(0)):
                    if l[i].item() == 1:
                        t.append(i)
                truth.append(t)
            for i, l in enumerate(output['logits']):
                y_model = torch.sigmoid(l)
                pred.append(y_model.tolist())
    pbar.close()

    #output the result
    scores = evaluate(pred, truth, label_dict, data_path)
    macro_f1 = scores['macro_f1']
    micro_f1 = scores['micro_f1']
    accuracy = scores['accuracy']
    HiP= scores['HiP']
    HiR= scores['HiR']
    HiF= scores['HiF']
    print('macro', macro_f1, 'micro', micro_f1, 'accuracy', accuracy,'\n',
            'HiP',HiP,'HiR',HiR,'HiF',HiF)
