import sys
import collections
import os
import os.path
import json
import numpy as np
import torch
import string
from string import punctuation as punct
import re
from transformers import XLNetTokenizer, XLNetConfig
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.utils.data import TensorDataset
import spacy

train_file = "coqa-train-v1.0.json"
test_file = "coqa-dev-v1.0.json"
MIN_FLOAT = -1e30
MAX_FLOAT = 1e30

class InputExample(object):
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 r_start = None,
                 r_end = None,
                 orig_answer_text=None,
                 start_position=None,
                 answer_type=None,
                 answer_subtype=None,
                 is_skipped=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.r_start = r_start
        self.r_end = r_end
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.answer_type = answer_type
        self.answer_subtype = answer_subtype
        self.is_skipped = is_skipped
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", paragraph_text: [%s]" % (self.paragraph_text)
        if self.start_position >= 0:
            s += ", start_position: %d" % (self.start_position)
            s += ", orig_answer_text: %s" % (self.orig_answer_text)
            s += ", answer_type: %s" % (self.answer_type)
            s += ", answer_subtype: %s" % (self.answer_subtype)
            s += ", is_skipped: %r" % (self.is_skipped)
        return "[{0}]\n".format(s)

class InputFeatures(object):
    def __init__(self,
                 unique_id,
                 qas_id,
                 doc_idx,
                 token2char_raw_start_index,
                 token2char_raw_end_index,
                 token2doc_index,
                 input_ids,
                 input_tokens,
                 input_mask,
                 p_mask,
                 segment_ids,
                 cls_index,
                 para_length,
                 r_start=None,
                 r_end=None,
                 start_position=None,
                 end_position=None,
                 is_unk=None,
                 is_yes=None,
                 is_no=None,
                 number=None,
                 option=None):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.doc_idx = doc_idx
        self.token2char_raw_start_index = token2char_raw_start_index
        self.token2char_raw_end_index = token2char_raw_end_index
        self.token2doc_index = token2doc_index
        self.input_ids = input_ids
        self.input_tokens = input_tokens
        self.input_mask = input_mask
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.para_length = para_length
        self.r_start = r_start
        self.r_end = r_end
        self.start_position = start_position
        self.end_position = end_position
        self.is_unk = is_unk
        self.is_yes = is_yes
        self.is_no = is_no
        self.number = number
        self.option = option

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "unique_id: %s" % (self.unique_id)
        s += ", input_ids: [%s] \n" % (self.input_ids)
        s += ", segment_ids: [%s] \n" % (self.segment_ids)
        s += ", input_mask: [%s] \n" % (self.input_mask)
        s += ", cls_idx: [%s] \n" % (self.cls_index)
        s += ", answer: [%s] \n" % (self.input_ids[self.start_position:self.end_position+1])
        return "[{0}]\n".format(s)


class OutputResult(object):
    def __init__(self,
                 unique_id,
                 unk_prob,
                 yes_prob,
                 no_prob,
                 num_probs,
                 opt_probs,
                 start_prob,
                 start_index,
                 end_prob,
                 end_index):
        self.unique_id = unique_id
        self.unk_prob = unk_prob
        self.yes_prob = yes_prob
        self.no_prob = no_prob
        self.num_probs = num_probs
        self.opt_probs = opt_probs
        self.start_prob = start_prob
        self.start_index = start_index
        self.end_prob = end_prob
        self.end_index = end_index
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        s = f"""unique_id: {self.unique_id} \n unk: {self.unk_prob} 
            \n yes: {self.yes_prob}\n no {self.no_prob} 
            \n sentence: {self.start_index}:{self.end_index}"""
        return s

class CoqaPipeline(object):
    def __init__(self, data_dir = "./data", num_turn = 2):
        self.data_dir = data_dir
        self.num_turn = num_turn
    
    def get_train_examples(self, dataset_type = None):
        data_path = os.path.join(self.data_dir, train_file)
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list,dataset_type = dataset_type)
        example_list = [example for example in example_list if not example.is_skipped]
        return example_list
    
    def get_dev_examples(self, dataset_type = None):
        data_path = os.path.join(self.data_dir, test_file)
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list,dataset_type = dataset_type)
        return example_list
    
    def _read_json(self, data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)["data"]
                return data_list
        else:
            raise FileNotFoundError("data path not found: {0}".format(data_path))

    @staticmethod
    def normalize_answer(s):
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _whitespace_tokenize(self, text):
        word_spans = []
        char_list = []
        for idx, char in enumerate(text):
            if char != ' ':
                char_list.append(idx)
                continue
            
            if char_list:
                word_start = char_list[0]
                word_end = char_list[-1]
                word_text = text[word_start:word_end+1]
                word_spans.append((word_text, word_start, word_end))
                char_list.clear()
        
        if char_list:
            word_start = char_list[0]
            word_end = char_list[-1]
            word_text = text[word_start:word_end+1]
            word_spans.append((word_text, word_start, word_end))
        
        return word_spans
    
    def _char_span_to_word_span(self,
                                char_start,
                                char_end,
                                word_spans):
        word_idx_list = []
        for word_idx, (_, start, end) in enumerate(word_spans):
            if end >= char_start:
                if start <= char_end:
                    word_idx_list.append(word_idx)
                else:
                    break
        
        if word_idx_list:
            word_start = word_idx_list[0]
            word_end = word_idx_list[-1]
        else:
            word_start = -1
            word_end = -1
        
        return word_start, word_end
    
    def _search_best_span(self,
                          context_tokens,
                          answer_tokens):
        best_f1 = 0.0
        best_start, best_end = -1, -1
        search_index = [idx for idx in range(len(context_tokens)) if context_tokens[idx][0] in answer_tokens]
        for i in range(len(search_index)):
            for j in range(i, len(search_index)):
                candidate_tokens = [context_tokens[k][0] for k in range(search_index[i], search_index[j]+1) if context_tokens[k][0]]
                common = collections.Counter(candidate_tokens) & collections.Counter(answer_tokens)
                num_common = sum(common.values())
                if num_common > 0:
                    precision = 1.0 * num_common / len(candidate_tokens)
                    recall = 1.0 * num_common / len(answer_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_start = context_tokens[search_index[i]][1]
                        best_end = context_tokens[search_index[j]][2]
        
        return best_f1, best_start, best_end
    
    def _get_question_text(self,
                           history,
                           question):
        question_tokens = ['<s>'] + question["input_text"].split(' ')
        return " ".join(history + [" ".join(question_tokens)])
    
    def _get_question_history(self,
                              history,
                              question,
                              answer,
                              answer_type,
                              is_skipped,
                              num_turn):
        question_tokens = []
        if answer_type != "unknown":
            question_tokens.extend(['<s>'] + question["input_text"].split(' '))
            question_tokens.extend(['</s>'] + answer["input_text"].split(' '))
        
        question_text = " ".join(question_tokens)
        if question_text:
            history.append(question_text)
        
        if num_turn >= 0 and len(history) > num_turn:
            history = history[-num_turn:]
        
        return history
    
    def _find_answer_span(self,
                          answer_text,
                          rationale_text,
                          rationale_start,
                          rationale_end):
        idx = rationale_text.find(answer_text)
        answer_start = rationale_start + idx
        answer_end = answer_start + len(answer_text) - 1
        
        return answer_start, answer_end
    
    def _match_answer_span(self,
                           answer_text,
                           rationale_start,
                           rationale_end,
                           paragraph_text):
        answer_tokens = self._whitespace_tokenize(answer_text)
        answer_norm_tokens = [self.normalize_answer(token) for token, _, _ in answer_tokens]
        answer_norm_tokens = [norm_token for norm_token in answer_norm_tokens if norm_token]
        
        if not answer_norm_tokens:
            return -1, -1
        
        paragraph_tokens = self._whitespace_tokenize(paragraph_text)
        
        if not (rationale_start == -1 or rationale_end == -1):
            rationale_word_start, rationale_word_end = self._char_span_to_word_span(rationale_start, rationale_end, paragraph_tokens)
            rationale_tokens = paragraph_tokens[rationale_word_start:rationale_word_end+1]
            rationale_norm_tokens = [(self.normalize_answer(token), start, end) for token, start, end in rationale_tokens]
            match_score, answer_start, answer_end = self._search_best_span(rationale_norm_tokens, answer_norm_tokens)
            
            if match_score > 0.0:
                return answer_start, answer_end
        
        paragraph_norm_tokens = [(self.normalize_answer(token), start, end) for token, start, end in paragraph_tokens]
        match_score, answer_start, answer_end = self._search_best_span(paragraph_norm_tokens, answer_norm_tokens)
        
        if match_score > 0.0:
            return answer_start, answer_end
        
        return -1, -1
    
    def _get_answer_span(self, answer, answer_type, paragraph_text):
        input_text = answer["input_text"].strip().lower()
        span_start, span_end = answer["span_start"], answer["span_end"]
        if span_start == -1 or span_end == -1:
            span_text = ""
        else:
            span_text = paragraph_text[span_start:span_end].lower()
        
        if input_text in span_text:
            span_start, span_end = self._find_answer_span(input_text, span_text, span_start, span_end)
        else:
            span_start, span_end = self._match_answer_span(input_text, span_start, span_end, paragraph_text.lower())
        
        if span_start == -1 or span_end == -1:
            answer_text = ""
            is_skipped = (answer_type == "span")
        else:
            answer_text = paragraph_text[span_start:span_end+1]
            is_skipped = False
        
        return answer_text, span_start, span_end, is_skipped
    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s
    def _normalize_answer(self, answer):
        norm_answer = self.normalize_answer(answer)
        
        if norm_answer in ["yes", "yese", "ye", "es"]:
            return "yes"
        
        if norm_answer in ["no", "no not at all", "not", "not at all", "not yet", "not really"]:
            return "no"
        
        return norm_answer
    
    def _get_answer_type(self, question, answer):
        norm_answer = self._normalize_answer(answer["input_text"])
        
        if norm_answer == "unknown" or "bad_turn" in answer:
            return "unknown", None
        
        if norm_answer == "yes":
            return "yes", None
        
        if norm_answer == "no":
            return "no", None
        
        if norm_answer in ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]:
            return "number", norm_answer
        
        norm_question_tokens = self.normalize_answer(question["input_text"]).split(" ")
        if "or" in norm_question_tokens:
            index = norm_question_tokens.index("or")
            if index-1 >= 0 and index+1 < len(norm_question_tokens):
                if norm_answer == norm_question_tokens[index-1]:
                    norm_answer = "option_a"
                elif norm_answer == norm_question_tokens[index+1]:
                    norm_answer = "option_b"
        
        if norm_answer in ["option_a", "option_b"]:
            return "option", norm_answer
        
        return "span", None
    
    def _process_found_answer(self,
                              raw_answer,
                              found_answer):
        raw_answer_tokens = raw_answer.split(' ')
        found_answer_tokens = found_answer.split(' ')
        
        raw_answer_last_token = raw_answer_tokens[-1].lower()
        found_answer_last_token = found_answer_tokens[-1].lower()
        
        if (raw_answer_last_token != found_answer_last_token and
            raw_answer_last_token == found_answer_last_token.rstrip(string.punctuation)):
            found_answer_tokens[-1] = found_answer_tokens[-1].rstrip(string.punctuation)
        
        return ' '.join(found_answer_tokens)
    def process(self, parsed_text):
        output = {'word': [], 'offsets': [], 'sentences': []}

        for token in parsed_text:
            output['word'].append(self._str(token.text))
            output['offsets'].append((token.idx, token.idx + len(token.text)))

        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        return output
    def _get_example(self, data_list,dataset_type = None):
        nlp = spacy.load('en_core_web_sm', parser=False) 
        examples = []
        for cnt,data in tqdm(enumerate(data_list),total = len(data_list),desc = "Preprocessing "):
            data_id = data["id"]
            paragraph_text = data["story"]
            
            questions = sorted(data["questions"], key=lambda x: x["turn_id"])
            answers = sorted(data["answers"], key=lambda x: x["turn_id"])
            
            question_history = []
            qas = list(zip(questions, answers))
            parsed = nlp(paragraph_text)
            nlp_contexts = self.process(parsed)

            for i, (question, answer) in enumerate(qas):
                qas_id = "{0}_{1}".format(data_id, i+1)
                r_start, r_end = answer['span_start'],answer['span_end']

                if dataset_type is not None:
                    if dataset_type == "TS":
                        edge,inc = r_end,True
                    elif dataset_type == "RG":
                        edge,inc = r_start,False
                        r_start,r_end = -1,-1
                    if edge != -1:
                        for m,(i,j) in enumerate(nlp_contexts['offsets']):
                            if i <= edge <= j:
                                edge = m+1 
                                break
                        for (i,j) in nlp_contexts['sentences']:
                            if i <= edge < j:
                                sent = j if inc else i
                                break
                        paragraph_text = str(parsed[:sent])
                        if r_start > len(paragraph_text):
                            continue
                    if len(paragraph_text) == 0:
                        continue

                answer_type, answer_subtype = self._get_answer_type(question, answer)

                if dataset_type == "RG" and answer_type == 'span':
                    gt = answer['input_text']
                    f = paragraph_text.find(gt)
                    if  f == -1:
                        r_start = len(paragraph_text)
                        paragraph_text = paragraph_text + ' ' + gt
                        r_end = len(paragraph_text)-1
                    else:
                        st = (paragraph_text[f-1].isspace()) or (paragraph_text[f-1] in punct) if f!= 0 else True
                        en = (paragraph_text[f+len(gt)] in punct) or (paragraph_text[f+len(gt)].isspace()) if (f+len(gt) < len(paragraph_text)) else True
                        if st and en:
                            r_start,r_end = f,f+len(gt)-1
                        else:
                            continue
                if len(paragraph_text) == 0:
                    continue
                answer_text, span_start, span_end, is_skipped = self._get_answer_span(answer, answer_type, paragraph_text)
                question_text = self._get_question_text(question_history, question)
                question_history = self._get_question_history(question_history, question, answer, answer_type, is_skipped, self.num_turn)
                
                                
                if answer_type not in ["unknown", "yes", "no"] and not is_skipped and answer_text:
                    start_position = span_start
                    orig_answer_text = self._process_found_answer(answer["input_text"], answer_text)
                else:
                    start_position = -1
                    orig_answer_text = ""
                
                example = InputExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    paragraph_text=paragraph_text,
                    r_start = r_start,
                    r_end = r_end,
                    orig_answer_text= orig_answer_text if dataset_type in [None,'TS'] else "unknown",
                    start_position=start_position if dataset_type in [None, 'TS'] else 0,
                    answer_type=answer_type if dataset_type in [None,'TS'] else "unknown",
                    answer_subtype=answer_subtype if dataset_type in [None,'TS'] else None,
                    is_skipped=is_skipped)

                examples.append(example)
        
        return examples

class Tokenizer(object):

    def __init__(self, pretrained_model_or_dir):
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_or_dir)
        self.do_lower_case = self.tokenizer.do_lower_case

    def tokenize(self, text):
        return self.tokenizer._tokenize(text)

    def preprocess_text(self, text):
        return self.tokenizer.preprocess_text(text)

    def tokens_to_ids(self, tokens):
        return [self.tokenizer._convert_token_to_id(token) for token in tokens]

    def save_pretrained(self, output_directory):
        self.tokenizer.save_pretrained(output_directory)
    

class XLNetExampleProcessor(object):
    def __init__(self, tokenizer, max_seq_length = 512, max_query_length = 128, doc_stride = 128, ):

        self.special_vocab_list = ["<unk>", "<s>", "</s>", "<cls>", "<sep>", "<pad>", "<mask>", "<eod>", "<eop>"]
        self.special_vocab_map = {}
        for (i, special_vocab) in enumerate(self.special_vocab_list):
            self.special_vocab_map[special_vocab] = i
        
        self.segment_vocab_list = ["<p>", "<q>", "<cls>", "<pad>"]
        self.segment_vocab_map = {"<p>":0, "<q>":1, "<cls>":1, "<pad>":1}
        
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.tokenizer = tokenizer
        self.unique_id = 1000000000
    
    def _generate_match_mapping(self, para_text, tokenized_para_text, N, M, max_N, max_M):
        def _lcs_match(para_text, tokenized_para_text, N, M, max_N, max_M, max_dist):
            f = np.zeros((max_N, max_M), dtype=np.float32)
            g = {}
            
            for i in range(N):
                for j in range(i - max_dist, i + max_dist):
                    if j >= M or j < 0:
                        continue
                    
                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]
                    
                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]
                    
                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    
                    raw_char = self.tokenizer.preprocess_text(para_text[i])
                    tokenized_char = tokenized_para_text[j]
                    if (raw_char == tokenized_char and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1
            
            return f, g
        
        max_dist = abs(N - M) + 5
        for _ in range(2):
            lcs_matrix, match_mapping = _lcs_match(para_text, tokenized_para_text, N, M, max_N, max_M, max_dist)
            
            if lcs_matrix[N - 1, M - 1] > 0.8 * N:
                break
            
            max_dist *= 2
        
        mismatch = lcs_matrix[N - 1, M - 1] < 0.8 * N
        return match_mapping, mismatch
    
    def _convert_tokenized_index(self, index, pos, M=None, is_start=True):
        """Convert index for tokenized text"""
        if index[pos] is not None:
            return index[pos]
        
        N = len(index)
        rear = pos
        while rear < N - 1 and index[rear] is None:
            rear += 1
        
        front = pos
        while front > 0 and index[front] is None:
            front -= 1
        
        assert index[front] is not None or index[rear] is not None
        
        if index[front] is None:
            if index[rear] >= 1:
                if is_start:
                    return 0
                else:
                    return index[rear] - 1
            
            return index[rear]
        
        if index[rear] is None:
            if M is not None and index[front] < M - 1:
                if is_start:
                    return index[front] + 1
                else:
                    return M - 1
            
            return index[front]
        
        if is_start:
            if index[rear] > index[front] + 1:
                return index[front] + 1
            else:
                return index[rear]
        else:
            if index[rear] > index[front] + 1:
                return index[rear] - 1
            else:
                return index[front]
    
    def _find_max_context(self, doc_spans, token_idx):
        best_doc_score = None
        best_doc_idx = None
        for (doc_idx, doc_span) in enumerate(doc_spans):
            doc_start = doc_span["start"]
            doc_length = doc_span["length"]
            doc_end = doc_start + doc_length - 1
            if token_idx < doc_start or token_idx > doc_end:
                continue
            
            left_context_length = token_idx - doc_start
            right_context_length = doc_end - token_idx
            doc_score = min(left_context_length, right_context_length) + 0.01 * doc_length
            if best_doc_score is None or doc_score > best_doc_score:
                best_doc_score = doc_score
                best_doc_idx = doc_idx
        
        return best_doc_idx
    
    def convert_coqa_example(self, example):
        query_tokens = []
        qa_texts = example.question_text.split('<s>')
        for qa_text in qa_texts:
            qa_text = qa_text.strip()
            if not qa_text:
                continue
            
            query_tokens.append('<s>')
            
            qa_items = qa_text.split('</s>')
            if len(qa_items) < 1:
                continue
            
            q_text = qa_items[0].strip()
            q_tokens = self.tokenizer.tokenize(q_text)
            query_tokens.extend(q_tokens)
            
            if len(qa_items) < 2:
                continue
            
            query_tokens.append('</s>')
            
            a_text = qa_items[1].strip()
            a_tokens = self.tokenizer.tokenize(a_text)
            query_tokens.extend(a_tokens)
        
        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[-self.max_query_length:]
        
        para_text = example.paragraph_text
        para_tokens = self.tokenizer.tokenize(example.paragraph_text)
        
        char2token_index = []
        token2char_start_index = []
        token2char_end_index = []
        char_idx = 0
        for i, token in enumerate(para_tokens):
            char_len = len(token)
            char2token_index.extend([i] * char_len)
            token2char_start_index.append(char_idx)
            char_idx += char_len
            token2char_end_index.append(char_idx - 1)
        
        tokenized_para_text = ''.join(para_tokens).replace('_', ' ')
        
        N, M = len(para_text), len(tokenized_para_text)
        max_N, max_M = 1024, 1024
        if N > max_N or M > max_M:
            max_N = max(N, max_N)
            max_M = max(M, max_M)
        
        match_mapping, mismatch = self._generate_match_mapping(para_text, tokenized_para_text, N, M, max_N, max_M)
        
        raw2tokenized_char_index = [None] * N
        tokenized2raw_char_index = [None] * M
        i, j = N-1, M-1
        while i >= 0 and j >= 0:
            if (i, j) not in match_mapping:
                break
            
            if match_mapping[(i, j)] == 2:
                raw2tokenized_char_index[i] = j
                tokenized2raw_char_index[j] = i
                i, j = i - 1, j - 1
            elif match_mapping[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1
        
        #if all(v is None for v in raw2tokenized_char_index) or mismatch:
            #print('Mismatch') 

        token2char_raw_start_index = []
        token2char_raw_end_index = []
        for idx in range(len(para_tokens)):
            start_pos = token2char_start_index[idx]
            end_pos = token2char_end_index[idx]
            raw_start_pos = self._convert_tokenized_index(tokenized2raw_char_index, start_pos, N, is_start=True)
            raw_end_pos = self._convert_tokenized_index(tokenized2raw_char_index, end_pos, N, is_start=False)
            token2char_raw_start_index.append(raw_start_pos)
            token2char_raw_end_index.append(raw_end_pos)
        #RATIONALE PART 
        marks = list(zip(token2char_raw_start_index,token2char_raw_end_index))
        raw_start,raw_end = example.r_start,example.r_end
        if raw_end >= marks[-1][1]:
            raw_end = marks[-1][1]
        if raw_start >= marks[-1][1]:
            raw_start = -1
        if raw_start != -1 and raw_end != -1:
            r_start_tokenised = [i for i,j in enumerate(marks) if j[0] <= raw_start <= j[1]][0]
            r_end_tokenised = [i for i,j in enumerate(marks) if j[0] <= raw_end <= j[1]][0]
        else:
            r_start_tokenised, r_end_tokenised = 0,0
        assert r_start_tokenised <= r_end_tokenised

        if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
            raw_start_char_pos = example.start_position
            raw_end_char_pos = raw_start_char_pos + len(example.orig_answer_text) - 1
            tokenized_start_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_start_char_pos, is_start=True)
            tokenized_end_char_pos = self._convert_tokenized_index(raw2tokenized_char_index, raw_end_char_pos, is_start=False)
            tokenized_start_token_pos = char2token_index[tokenized_start_char_pos]
            tokenized_end_token_pos = char2token_index[tokenized_end_char_pos]
            assert tokenized_start_token_pos <= tokenized_end_token_pos
        else:
            tokenized_start_token_pos = tokenized_end_token_pos = -1
        
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_para_length = self.max_seq_length - len(query_tokens) - 3
        total_para_length = len(para_tokens)
        
        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        doc_spans = []
        para_start = 0
        while para_start < total_para_length:
            para_length = total_para_length - para_start
            if para_length > max_para_length:
                para_length = max_para_length
            
            doc_spans.append({
                "start": para_start,
                "length": para_length
            })
            
            if para_start + para_length == total_para_length:
                break
            
            para_start += min(para_length, self.doc_stride)
        
        feature_list = []
        for (doc_idx, doc_span) in enumerate(doc_spans):
            input_tokens = []
            segment_ids = []
            p_mask = []
            doc_token2char_raw_start_index = []
            doc_token2char_raw_end_index = []
            doc_token2doc_index = {}
            
            for i in range(doc_span["length"]):
                token_idx = doc_span["start"] + i
                
                doc_token2char_raw_start_index.append(token2char_raw_start_index[token_idx])
                doc_token2char_raw_end_index.append(token2char_raw_end_index[token_idx])
                
                best_doc_idx = self._find_max_context(doc_spans, token_idx)
                doc_token2doc_index[len(input_tokens)] = (best_doc_idx == doc_idx)
                
                input_tokens.append(para_tokens[token_idx])
                segment_ids.append(self.segment_vocab_map["<p>"])
                p_mask.append(0)
            
            doc_para_length = len(input_tokens)
            
            input_tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<p>"])
            p_mask.append(1)
            
            # We put P before Q because during pretraining, B is always shorter than A
            for query_token in query_tokens:
                input_tokens.append(query_token)
                segment_ids.append(self.segment_vocab_map["<q>"])
                p_mask.append(1)

            input_tokens.append("<sep>")
            segment_ids.append(self.segment_vocab_map["<q>"])
            p_mask.append(1)
            
            cls_index = len(input_tokens)
            
            input_tokens.append("<cls>")
            segment_ids.append(self.segment_vocab_map["<cls>"])
            p_mask.append(0)
            input_ids = self.tokenizer.tokens_to_ids(input_tokens)

            # The mask has 0 for real tokens and 1 for padding tokens. Only real tokens are attended to.
            input_mask = [0] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < self.max_seq_length:
                input_ids.append(5) #pad
                input_mask.append(1)
                segment_ids.append(self.segment_vocab_map["<pad>"])
                p_mask.append(1)
            
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(p_mask) == self.max_seq_length
            
            start_position = None
            end_position = None
            is_unk = (example.answer_type == "unknown" or example.is_skipped)
            is_yes = (example.answer_type == "yes")
            is_no = (example.answer_type == "no")
            
            doc_start = doc_span["start"]
            doc_end = doc_start + doc_span["length"] - 1
            if r_start_tokenised >= doc_start and r_end_tokenised <= doc_end:
                r_start_tokenised = r_start_tokenised - doc_start
                r_end_tokenised = r_end_tokenised - doc_start
            if example.answer_type == "number":
                number_list = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
                number = number_list.index(example.answer_subtype) + 1
            else:
                number = 0
            
            if example.answer_type == "option":
                option_list = ["option_a", "option_b"]
                option = option_list.index(example.answer_subtype) + 1
            else:
                option = 0
            
            if example.answer_type not in ["unknown", "yes", "no"] and not example.is_skipped and example.orig_answer_text:
                doc_start = doc_span["start"]
                doc_end = doc_start + doc_span["length"] - 1
                if tokenized_start_token_pos >= doc_start and tokenized_end_token_pos <= doc_end:
                    start_position = tokenized_start_token_pos - doc_start
                    end_position = tokenized_end_token_pos - doc_start
                else:
                    start_position = cls_index
                    end_position = cls_index
                    is_unk = True
            else:
                start_position = cls_index
                end_position = cls_index

            feature = InputFeatures(
                unique_id=self.unique_id,
                qas_id=example.qas_id,
                doc_idx=doc_idx,
                token2char_raw_start_index=doc_token2char_raw_start_index,
                token2char_raw_end_index=doc_token2char_raw_end_index,
                token2doc_index=doc_token2doc_index,
                input_tokens = input_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                p_mask=p_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                para_length=doc_para_length,
                r_start = r_start_tokenised,
                r_end = r_end_tokenised,
                start_position=start_position,
                end_position=end_position,
                is_unk=is_unk,
                is_yes=is_yes,
                is_no=is_no,
                number=number,
                option=option)
            feature_list.append(feature)
            self.unique_id += 1
        
        return feature_list
    
    def convert_examples_to_features(self, examples, is_training):
        threads = cpu_count()
        with Pool(threads) as p:
            examp = list(tqdm(
                p.imap(self.convert_coqa_example, examples), total=len(examples), desc="Extracting Features", ))

        features = [item for sublist in examp for item in sublist]
        for i in range(len(features)):
            features[i].unique_id = 10000000+i 

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_idx = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.long)
         
        if not is_training:
            all_r_start = torch.tensor([f.r_start for f in features], dtype=torch.long)
            all_r_end = torch.tensor([f.r_end for f in features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_cls_idx, all_p_mask, all_example_index,all_r_start, all_r_end)
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            all_unk = torch.tensor([f.is_unk for f in features], dtype=torch.long)
            all_yes = torch.tensor([f.is_yes for f in features], dtype=torch.long)
            all_no = torch.tensor([f.is_no for f in features], dtype=torch.long)
            all_number = torch.tensor([f.number for f in features], dtype=torch.long)
            all_option = torch.tensor([f.option for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_p_mask, all_cls_idx,
                    all_start_positions, all_end_positions, all_unk, all_yes, all_no, all_number, all_option)

        return features, dataset

class XLNetPredictProcessor(object):
    def __init__(self, output_dir, tokenizer, n_best_size = 5, start_n_top = 5, end_n_top = 5, max_answer_length = 16,  predict_tag=None):
        self.n_best_size = n_best_size
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.max_answer_length = max_answer_length
        self.tokenizer = tokenizer
        
        predict_tag = predict_tag if predict_tag else "normal"
        self.output_summary = os.path.join(output_dir, "predict_{0}_sum.json".format(predict_tag))
        self.output_detail = os.path.join(output_dir, "predict_{0}_det.json".format(predict_tag))
    
    def _write_to_json(self, data_list, data_path):
        data_folder = os.path.dirname(data_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        with open(data_path, "w") as file:  
            json.dump(data_list, file, indent=4)
   
    def process(self, examples, features, results):
        qas_id_to_features = {}
        unique_id_to_feature = {}
        for feature in features:
            if feature.qas_id not in qas_id_to_features:
                qas_id_to_features[feature.qas_id] = []
            
            qas_id_to_features[feature.qas_id].append(feature)
            unique_id_to_feature[feature.unique_id] = feature
        
        unique_id_to_result = {}
        for result in results:
            unique_id_to_result[result.unique_id] = result
        
        predict_summary_list = []
        predict_detail_list = []
        num_example = len(examples)
        for (example_idx, example) in enumerate(examples):
            if example_idx % 1000 == 0:
                print('Updating {0}/{1} example with predict'.format(example_idx, num_example))
            
            if example.qas_id not in qas_id_to_features:
                print('No feature found for example: {0}'.format(example.qas_id))
                continue
            
            example_unk_score = MAX_FLOAT
            example_yes_score = MIN_FLOAT
            example_no_score = MIN_FLOAT
            example_num_id = 0
            example_num_score = MIN_FLOAT
            example_num_probs = None
            example_opt_id = 0
            example_opt_score = MIN_FLOAT
            example_opt_probs = None
            
            example_all_predicts = []
            example_features = qas_id_to_features[example.qas_id]
            for example_feature in example_features:
                if example_feature.unique_id not in unique_id_to_result:
                    print('No result found for feature: {0}'.format(example_feature.unique_id))
                    continue
                
                example_result = unique_id_to_result[example_feature.unique_id]
                example_unk_score = min(example_unk_score, float(example_result.unk_prob))
                example_yes_score = max(example_yes_score, float(example_result.yes_prob))
                example_no_score = max(example_no_score, float(example_result.no_prob))
                
                num_probs = [float(num_prob) for num_prob in example_result.num_probs]
                num_id = int(np.argmax(num_probs[1:])) + 1
                num_score = num_probs[num_id]
                if example_num_score < num_score:
                    example_num_id = num_id
                    example_num_score = num_score
                    example_num_probs = num_probs
                
                opt_probs = [float(opt_prob) for opt_prob in example_result.opt_probs]
                opt_id = int(np.argmax(opt_probs[1:])) + 1
                opt_score = opt_probs[opt_id]
                if example_opt_score < opt_score:
                    example_opt_id = opt_id
                    example_opt_score = opt_score
                    example_opt_probs = opt_probs
                
                for i in range(self.start_n_top):
                    start_prob = example_result.start_prob[i]
                    start_index = example_result.start_index[i]
                    
                    for j in range(self.end_n_top):
                        end_prob = example_result.end_prob[i][j]
                        end_index = example_result.end_index[i][j]
                        
                        answer_length = end_index - start_index + 1
                        if end_index < start_index or answer_length > self.max_answer_length:
                            continue
                        
                        if start_index > example_feature.para_length or end_index > example_feature.para_length:
                            continue
                        
                        if start_index not in example_feature.token2doc_index:
                            continue
                        
                        example_all_predicts.append({
                            "unique_id": example_result.unique_id,
                            "start_prob": float(start_prob),
                            "start_index": int(start_index),
                            "end_prob": float(end_prob),
                            "end_index": int(end_index),
                            "predict_score": float(np.log(start_prob) + np.log(end_prob))
                        })
            
            example_all_predicts = sorted(example_all_predicts, key=lambda x: x["predict_score"], reverse=True)
            
            is_visited = set()
            example_top_predicts = []
            for example_predict in example_all_predicts:
                if len(example_top_predicts) >= self.n_best_size:
                    break
                
                example_feature = unique_id_to_feature[example_predict["unique_id"]]
                predict_start = example_feature.token2char_raw_start_index[example_predict["start_index"]]
                if example_predict["end_index"] >= len(example_feature.token2char_raw_end_index):
                    example_predict["end_index"] = len(example_feature.token2char_raw_end_index)-1
                predict_end = example_feature.token2char_raw_end_index[example_predict["end_index"]]
                predict_text = example.paragraph_text[predict_start:predict_end + 1].strip()
                
                if predict_text in is_visited:
                    continue
                
                is_visited.add(predict_text)
                
                example_top_predicts.append({
                    "predict_text": predict_text,
                    "predict_score": example_predict["predict_score"]
                })
            
            if len(example_top_predicts) == 0:
                example_top_predicts.append({
                    "predict_text": "",
                    "predict_score": 0.0
                })
            
            example_best_predict = example_top_predicts[0]
            
            example_question_text = example.question_text.split('<s>')[-1].strip()
            
            predict_summary_list.append({
                "qas_id": example.qas_id,
                "question_text": example_question_text,
                "label_text": example.orig_answer_text,
                "unk_score": example_unk_score,
                "yes_score": example_yes_score,
                "no_score": example_no_score,
                "num_id": example_num_id,
                "num_score": example_num_score,
                "num_probs": example_num_probs,
                "opt_id": example_opt_id,
                "opt_score": example_opt_score,
                "opt_probs": example_opt_probs,
                "predict_text": example_best_predict["predict_text"],
                "predict_score": example_best_predict["predict_score"]
            })
                                          
            predict_detail_list.append({
                "qas_id": example.qas_id,
                "question_text": example_question_text,
                "label_text": example.orig_answer_text,
                "unk_score": example_unk_score,
                "yes_score": example_yes_score,
                "no_score": example_no_score,
                "num_id": example_num_id,
                "num_score": example_num_score,
                "num_probs": example_num_probs,
                "opt_id": example_opt_id,
                "opt_score": example_opt_score,
                "opt_probs": example_opt_probs,
                "best_predict": example_best_predict,
                "top_predicts": example_top_predicts
            })
        
        self._write_to_json(predict_summary_list, self.output_summary)
        self._write_to_json(predict_detail_list, self.output_detail)


