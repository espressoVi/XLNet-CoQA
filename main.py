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
import getopt,sys

pretrained_model="xlnet-base-cased"
max_seq_length = 512
epochs = 1.0
evaluation_batch_size = 16
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
            option = None,):
        
        predicts = {}
        #****************************************XLNET-BASE***************************
        output_result,_ = self.xlnet(input_ids,
                            token_type_ids=segment_ids,
                            input_mask=input_mask,
                            return_dict = False)
        #****************************************START MODELLIING*********************
        start_result = output_result
        start_result_mask = 1 - p_mask
        start_result = self.start_project(start_result)
        start_result = torch.squeeze(start_result, dim=-1)
        start_result = self.generate_masked_data(start_result, start_result_mask)
        start_prob = torch.softmax(start_result, dim=-1)
        
        if not self.training:
            start_top_prob, start_top_index = torch.topk(start_prob, k=top_k)
            predicts["start_prob"] = start_top_prob
            predicts["start_index"] = start_top_index

        #****************************************END MODELLIING***********************
        if self.training:
            start_index = F.one_hot(torch.unsqueeze(start_positions, dim=-1), self.seq_len)
            feat_result = start_index.type(torch.float) @ output_result
            feat_result = feat_result.repeat(1,self.seq_len,1)
            
            end_result = torch.cat([output_result, feat_result], dim=-1)
            end_result_mask = 1 - p_mask
            
            end_result = torch.tanh( self.end_modelling(end_result))
            end_result = self.end_norm(end_result)
            end_result = self.end_project(end_result)
            
            end_result = torch.squeeze(end_result, dim=-1)
            end_result = self.generate_masked_data(end_result, end_result_mask)
            end_prob = torch.softmax(end_result, dim=-1)
        else:
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
    
        #****************************************LOSSES********************************
        bce = BCEWithLogitsLoss(reduction = "none")
        loss = 0.0
        if self.training:
            start_label = start_positions
            start_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            start_loss = self.compute_loss(start_label, start_label_mask, start_result, start_result_mask, self.seq_len)
            end_label = end_positions
            end_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            end_loss = self.compute_loss(end_label, end_label_mask, end_result, end_result_mask,self.seq_len)
            loss += torch.mean(start_loss + end_loss)
            
            unk_label = is_unk
            unk_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            unk_loss = bce(unk_result,unk_label.type(torch.float) * unk_label_mask.type(torch.float))
            loss += torch.mean(unk_loss)
            
            yes_label = is_yes
            yes_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            yes_loss = bce(yes_result, yes_label.type(torch.float) * yes_label_mask.type(torch.float))
            loss += torch.mean(yes_loss)
            
            no_label = is_no
            no_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            no_loss = bce(no_result, no_label.type(torch.float) * no_label_mask.type(torch.float))
            loss += torch.mean(no_loss)
            
            num_label = number
            num_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            num_loss = self.compute_loss(num_label, num_label_mask, num_result, num_result_mask, 12)
            loss += torch.mean(num_loss)
            
            opt_label = option
            opt_label_mask,_ = torch.max(1 - p_mask, dim=-1)
            opt_loss = self.compute_loss(opt_label, opt_label_mask, opt_result, opt_result_mask,3)
            loss += torch.mean(opt_loss)

            return loss
        return predicts

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(train_dataset, model, tokenizer, device,output_directory):
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    t_total = len(train_dataloader) // 1 * epochs
    optimizer_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],"weight_decay": 0.01,},
                            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_parameters,lr=3e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3000, num_training_steps=t_total)
    if os.path.isfile(os.path.join(pretrained_model, "optimizer.pt")) and os.path.isfile(os.path.join(pretrained_model, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(
            os.path.join(pretrained_model, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(pretrained_model, "scheduler.pt")))
    counter = 1
    epochs_trained = 0
    train_loss, loss = 0.0, 0.0
    model.zero_grad()
    iterator = trange(epochs_trained, int(epochs), desc="Epoch", disable=False)
    for _ in iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { "input_ids": batch[0],
                       "input_mask": batch[1],
                       "segment_ids": batch[2],
                       "p_mask": batch[3],
                       "cls_index": batch[4],
                       "start_positions": batch[5],
                       "end_positions": batch[6],
                       "is_unk": batch[7],
                       "is_yes": batch[8],
                       "is_no": batch[9],
                       "number": batch[10],
                       "option": batch[11]}
            loss = model(**inputs)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description("Loss :%f" % (train_loss/(4*counter)))
            epoch_iterator.refresh()

            if counter % 1000 == 0:
                output_dir = os.path.join(output_directory, "model_weights")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    return train_loss/counter


def Write_predictions(model, tokenizer, device, dataset_type = None,output_directory = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    predict_results = []
    for batch in tqdm(evaluation_dataloader, desc="Evaluating"):
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
        pred_results = [OutputResult(
            unique_id = features[index[i].item()].unique_id,
            unk_prob=result["unk_prob"][i].item(),
            yes_prob=result["yes_prob"][i].item(),
            no_prob=result["no_prob"][i].item(),
            num_probs=result["num_probs"][i].tolist(),
            opt_probs=result["opt_probs"][i].tolist(),
            start_prob=result["start_prob"][i].tolist(),
            start_index=result["start_index"][i].tolist(),
            end_prob=result["end_prob"][i].tolist(),
            end_index=result["end_index"][i].tolist()) for i in range(result["unk_prob"].shape[0])]
        predict_results.extend(pred_results)

    predict_processor = XLNetPredictProcessor(output_dir = output_directory, tokenizer=tokenizer, predict_tag = "normal")
    predict_processor.process(examples, features, predict_results)

def load_dataset(tokenizer, evaluate=False, dataset_type = None):
    input_dir = "data"
    examples = []
    processor = CoqaPipeline()
    proc = processor.get_dev_examples if evaluate else processor.get_train_examples
    assert not evaluate or (len(dataset_type) == 1)
    for datas in dataset_type:
        examples.extend(proc(dataset_type = datas))
    feat_extract = XLNetExampleProcessor(tokenizer)
    features, dataset = feat_extract.convert_examples_to_features(examples, not evaluate)
    if evaluate:
        return dataset, examples, features
    return dataset

def manager(isTraining,dataset_type, output_directory):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    config = XLNetConfig.from_pretrained(pretrained_model)
    if isTraining:
        tokenizer = Tokenizer(pretrained_model)
        model = XLNetBaseModel(config, load_pre = True)
        model.to(device)
        if os.path.exists(output_directory) and os.listdir(output_directory):
            raise ValueError(f"Output directory {output_directory}  already exists, Change output_directory name")
        else:
            os.makedirs(output_directory)
        
        train_dataset = load_dataset(tokenizer, evaluate=False, dataset_type = dataset_type)
        train_loss = train(train_dataset, model, tokenizer, device,output_directory)
        tokenizer.save_pretrained(output_directory)
        torch.save(model.state_dict(), os.path.join(output_directory,'tweights.pt'))
    else:
        model = XLNetBaseModel(config)
        model.load_state_dict(torch.load(os.path.join(output_directory,'tweights.pt')))
        model.to(device)
        tokenizer = Tokenizer(output_directory)
        Write_predictions(model, tokenizer, device, dataset_type = dataset_type, output_directory = output_directory)

def main():
    isTraining,isEval = False, False
    train_dataset_type, eval_dataset_type = [],[]
    output_directory = "XLNet"
    argumentList = sys.argv[1:]
    options = "ht:e:o:"
    long_options = ["help", "train=","eval=", "output="]
    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--Help"):
                print ("""python main.py --train [O|C] --eval [O|TS|RG] --output [directory name]\n
                        --train O for original training C for combined training \n
                        --eval for eval on (O) original (TS) truncated and (RG) for TS-R dataset as defined in paper\n
                        --output [dir_name] is the output directory to write weights and predictions in, 
                        and in case of eval to load weights from.
                        e.g. python main.py --train C --eval RG --output XLNet_comb
                        for combined training followed by eval on RG and writing to ./XLNet_comb""")
                return
     
            elif currentArgument in ("-t", "--train"):
                isTraining = True
                opts = {'O':[None],'C':[None, 'TS','RG']}
                if currentValue in opts:
                    train_dataset_type = opts[currentValue]
                else:
                    print('See "python main.py --help" for usage')
                    return
            elif currentArgument in ("-e", "--eval"):
                opts = {'O':[None],'TS':['TS'], 'RG':['RG']}
                if currentValue in opts:
                    eval_dataset_type = opts[currentValue]
                    isEval = True
                else:
                    print('See "python main.py --help" for usage')
                    return
            elif currentArgument in ("-o", "--output"):
                output_directory = currentValue

    except getopt.error as err:
        print (str(err))

    if isTraining:
        manager(isTraining = True, dataset_type = train_dataset_type, output_directory = output_directory)
    if isEval:
        manager(isTraining = False, dataset_type = eval_dataset_type, output_directory = output_directory)
if __name__ == "__main__":
    main()
