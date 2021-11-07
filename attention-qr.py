import collections
import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)
from processors.metrics import get_predictions
from transformers import XLNetModel, XLNetTokenizer, XLNetConfig
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss
from processors.coqa import CoqaPipeline, Tokenizer, XLNetExampleProcessor, XLNetPredictProcessor, OutputResult
import numpy as np

train_file="coqa-train-v1.0.json"
predict_file="coqa-dev-v1.0.json"
output_directory="XLNet_comb"
pretrained_model="xlnet-base-cased"
max_seq_length = 512
epochs = 1.0
evaluation_batch_size = 14
train_batch_size = 4
lr = 3e-5
MIN_FLOAT = -1e30
MAX_FLOAT = 1e30
top_k = 5
 
class XLNetBaseModel(XLNetModel):
    def __init__(self,config, load_pre = False):
        super(XLNetBaseModel,self).__init__(config)
        self.xlnet = XLNetModel.from_pretrained(pretrained_model, config=config,) if load_pre else XLNetModel(config)
        hidden_size = config.hidden_size
        self.seq_len = max_seq_length
        self.start_project = nn.Linear(hidden_size,1)

        self.end_modelling = nn.Linear(2*hidden_size,hidden_size)
        self.end_norm = nn.LayerNorm(hidden_size)
        self.end_project = nn.Linear(hidden_size,1)

        self.answer_modelling = nn.Linear(2*hidden_size,hidden_size)
        self.answer_drop = nn.Dropout(p=0.1)

        self.unk_project = nn.Linear(hidden_size,1)
        self.yes_project = nn.Linear(hidden_size,1)
        self.no_project = nn.Linear(hidden_size,1)
        self.num_project = nn.Linear(hidden_size,12)
        self.opt_project = nn.Linear(hidden_size,3)

        self.relu = nn.ReLU

    def generate_masked_data(self, input_data, input_mask):
        return input_data * input_mask + MIN_FLOAT * (1 - input_mask)

    def compute_loss(self, label, label_mask, predict, predict_mask, size ,):
        masked_predict = self.generate_masked_data(predict, predict_mask)
        masked_label = label.type(torch.long) * label_mask.type(torch.long)
        masked_label = F.one_hot(masked_label,size)
        loss = -torch.sum(F.log_softmax(masked_predict, dim=1) * masked_label, dim=1)
        return loss

    def forward(self,input_ids,
            input_mask, segment_ids,
            p_mask, cls_index,
            start_positions=None,end_positions=None,
            is_unk = None,is_yes = None,
            is_no = None, number = None, 
            option = None):
        
        predicts = {}
        #****************************************XLNET-BASE***************************
        output_result,_,attentions = self.xlnet(input_ids,
                        token_type_ids=segment_ids,
                        input_mask=input_mask,
                        output_hidden_states = False,
                        output_attentions = True,
                        return_dict = False)
        attentions = list(attentions)
        #****************************************START MODELLIING*********************
        start_result = output_result
        start_result_mask = 1 - p_mask
        start_result = self.start_project(start_result)
        start_result = torch.squeeze(start_result, dim=-1)
        start_result = self.generate_masked_data(start_result, start_result_mask)
        start_prob = torch.softmax(start_result, dim=-1)
        
        start_top_prob, start_top_index = torch.topk(start_prob, k=top_k)
        predicts["start_prob"] = start_top_prob
        predicts["start_index"] = start_top_index

        #****************************************END MODELLIING***********************
        start_index = F.one_hot(start_top_index, self.seq_len)
        feat_result = start_index.type(torch.float)@ output_result
        feat_result = torch.unsqueeze(feat_result, dim=1)
        feat_result = feat_result.repeat(1,self.seq_len,1,1)
        
        end_result = torch.unsqueeze(output_result, dim=-2)
        end_result = end_result.repeat(1,1,top_k,1)
        end_result = torch.cat([end_result, feat_result], dim=-1)
        end_result_mask = torch.unsqueeze(1 - p_mask, dim=1)
        end_result_mask = end_result_mask.repeat(1, top_k,1)
        
        end_result = torch.tanh( self.end_modelling(end_result))
        end_result = self.end_norm(end_result)
        end_result = self.end_project(end_result)
        
        end_result = (torch.squeeze(end_result, dim=-1)).permute(0, 2, 1)
        end_result = self.generate_masked_data(end_result, end_result_mask)
        end_prob = torch.softmax(end_result, dim=-1)
        
        end_top_prob, end_top_index = torch.topk(end_prob, k=top_k)
        predicts["end_prob"] = end_top_prob
        predicts["end_index"] = end_top_index

        #****************************************ANSWER MODELLING*********************
        answer_cls_index = F.one_hot(torch.unsqueeze(cls_index, dim=-1), self.seq_len)
        answer_feat_result = (torch.unsqueeze(start_prob, dim=1)) @ output_result
        answer_output_result = answer_cls_index.type(torch.float)@ output_result
        answer_result = torch.cat([answer_feat_result, answer_output_result], dim=-1)
        answer_result = torch.squeeze(answer_result, dim=1)
        answer_result = torch.tanh( self.answer_modelling(answer_result))
        answer_result = self.answer_drop(answer_result)
        
        #****************************************UNKNOWN MODELLING********************
        unk_result = self.unk_project(answer_result)
        unk_result_mask,_ = torch.max(1 - p_mask, dim=-1)
        unk_result = torch.squeeze(unk_result, dim=-1)
        unk_result = self.generate_masked_data(unk_result, unk_result_mask)
        unk_prob = torch.sigmoid(unk_result)
        predicts["unk_prob"] = unk_prob
        
        #****************************************YES MODELLING************************
        yes_result = self.yes_project(answer_result)
        yes_result_mask,_ = torch.max(1 - p_mask, dim=-1)
        yes_result = torch.squeeze(yes_result, dim=-1)
        yes_result = self.generate_masked_data(yes_result, yes_result_mask)
        yes_prob = torch.sigmoid(yes_result)
        predicts["yes_prob"] = yes_prob
        
        #****************************************NO MODELLING*************************
        no_result = self.no_project(answer_result)
        no_result_mask,_ = torch.max(1 - p_mask, dim=-1)
        no_result = torch.squeeze(no_result, dim=-1)
        no_result = self.generate_masked_data(no_result, no_result_mask)
        no_prob = torch.sigmoid(no_result)
        predicts["no_prob"] = no_prob
        
        #****************************************NUM MODELLING************************
        num_result = self.num_project(answer_result)
        num_result_mask,_ = torch.max(1 - p_mask, dim=-1, keepdims=True)
        num_result = self.generate_masked_data(num_result, num_result_mask)
        num_probs = torch.softmax(num_result, dim=-1)
        predicts["num_probs"] = num_probs
        
        #****************************************OPT MODELLIING***********************
        opt_result = self.opt_project(answer_result)
        opt_result_mask,_ = torch.max(1 - p_mask, dim=-1, keepdims=True)
        opt_result = self.generate_masked_data(opt_result, opt_result_mask)
        opt_probs = torch.softmax(opt_result, dim=-1)
        predicts["opt_probs"] = opt_probs
        
        return attentions
        #return predicts,attentions

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

def Write_attentions(model, tokenizer, device, dataset_type = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    qr_results = [[],[],[],[],[],[],[],[],[],[],[],[]]
    sep_results = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for batch in tqdm(evaluation_dataloader, desc="Attentions"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = { "input_ids": batch[0],
                       "input_mask": batch[1],
                       "segment_ids": batch[2],
                       "cls_index": batch[3],
                       "p_mask": batch[4],}
            index = batch[5]
            result = model(**inputs)
        for i, example_index in enumerate(index):
            eval_feature = features[index[i].item()]
            doc_tok = eval_feature.input_tokens
            r_start,r_end = eval_feature.r_start,eval_feature.r_end
            seps = [i for i,j in enumerate(doc_tok) if j == "<sep>"]
            assert len(seps) == 2
            q_start = seps[0]+1
            q_end = seps[1]
            length =len(np.where(np.array(eval_feature.input_mask) == 0)[0])
            if (r_start,r_end) == (0,0) or r_start == r_end or r_end>=length or q_start >= q_end:
                continue
            attentions = result
            attentions = [output[i].detach().cpu().numpy() for output in attentions]
            for j in range(12):
                qr_results[j].append(attention_qr(attentions[j], -1, r_start,r_end,q_start,q_end, length))
                sep_results[j].append(attention_sep(attentions[j], -1, seps))
    qr_results = np.array(qr_results)
    sep_results = np.array(sep_results)
    
    Mean_qr = np.mean(qr_results,axis = 1)
    STD_qr = np.std(qr_results, axis = 1)
    print('eta QR')
    for i in range(12):
        print(f'{i} &\t {Mean_qr[i]} & \t{STD_qr[i]}\\\\')
    Mean = np.mean(sep_results,axis = 1)
    STD = np.std(sep_results, axis = 1)
    print('p SEP')
    for i in range(12):
        print(f'{i} &\t {Mean[i]} & \t{STD[i]}\\\\')

def attention_qr(attention,head,r_start,r_end,q_start,q_end,length):
    assert head < len(attention)
    if head == -1:
        attention = np.mean(attention, axis = 0)
    else:
        attention = attention[head]
    assert attention.shape == (max_seq_length,max_seq_length)
    su = []
    for i in range(r_start,r_end):
        eta = np.sum(attention[i][q_start:q_end])
        eta = (eta*length) / (q_end - q_start)
        su.append(eta)
    return np.mean(su)

def attention_sep(attention,head,seps):
    assert head < len(attention)
    if head == -1:
        attention = np.mean(attention, axis = 0)
    else:
        attention = attention[head]
    assert attention.shape == (max_seq_length,max_seq_length)
    p = 0
    for i in seps:
        p += np.mean(attention[:,i])
    return p
    
def load_dataset(tokenizer, evaluate=False, dataset_type = None):
    processor = CoqaPipeline()
    examples = processor.get_dev_examples(dataset_type = dataset_type)
    feat_extract = XLNetExampleProcessor(tokenizer)
    features, dataset = feat_extract.convert_examples_to_features(examples, not evaluate)
    return dataset, examples, features

def main(model_dir, dataset_type):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    config = XLNetConfig.from_pretrained(pretrained_model)
    tokenizer = Tokenizer(output_directory)
    model = XLNetBaseModel(config)
    model.load_state_dict(torch.load(os.path.join(output_directory,'tweights.pt')))
    model.to(device)
    for i in dataset_type:
        print(model_dir,i)
        Write_attentions(model, tokenizer, device, dataset_type = i)

if __name__ == "__main__":
    main(model_dir = 'XLNet_base',dataset_type = ['TS'])
    main(model_dir = 'XLNet_combM',dataset_type = ['RG',None,'TS'])
