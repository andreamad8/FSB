import pickle
import nltk
import spacy
import benepar
import torch
import json
import os
import subprocess
import numpy as np
from tempfile import TemporaryDirectory
from fairseq.models.bart import BARTModel
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import stopwords 
from nltk.tree import Tree
from nltk.tree import ParentedTree
from benepar.spacy_plugin import BeneparComponent
from collections import defaultdict, Counter
from tqdm import tqdm
from itertools import groupby

class FEQA(object):
    def __init__(self, squad_dir='metric/qa_models/squad1.0', bart_qa_dir='metric/bart_qg/', use_gpu=False):
        self.qg_model = BARTModel.from_pretrained(
            bart_qa_dir,
            checkpoint_file = 'checkpoint_best.pt'
            )

        if use_gpu:
            self.qg_model.cuda()
            self.qg_model.half()
        self.qg_model.eval()

        self.batch_size = 64
        self.beam_size = 1
        self.max_length = 100

        self.nlp = spacy.load('en_core_web_sm')
        self.parser = benepar.Parser("benepar_en2")
        self.stop_words = set(stopwords.words('english'))

        self.squad_cmd = [f'CUDA_VISIBLE_DEVICES=7 python {squad_dir}/run_squad.py',
                          '--model_type bert',
                         f'--model_name_or_path {squad_dir}',
                          '--do_eval',
                          '--overwrite_cache',
                          '--do_lower_case',
                          '--predict_file {}',
                          '--per_gpu_eval_batch_size 32',
                          '--max_seq_length 384',
                          '--doc_stride 128',
                          '--output_dir {}']

        self.squad_cmd = ' '.join(self.squad_cmd)


    def _get_entities(self, output_summary):
        entities = [X.text for X in self.nlp(output_summary).ents]
        return entities


    def _get_masked_phrases(self, output_summary, phrase_types=["NP"]):
        n = 10
        groups = (list(values) for _, values in groupby(output_summary))
        output_summary = "".join("".join(v) for v in groups if len(v) < n)
        masked_phrases = []
        if output_summary== "" or len(output_summary.split(" "))>25: return []
        parse_tree = self.parser.parse(output_summary)
        for subtree in parse_tree.subtrees():
            phrases_list = [(subtree_.leaves(), subtree_.label()) for subtree_ in subtree if type(subtree_) == Tree and subtree_.label() in phrase_types]
            for phrase_tuple in phrases_list:
                phrase = phrase_tuple[0]
                phrase_type = phrase_tuple[1]
                phrase_text = " ".join(phrase)
                if len(phrase) > 0 and phrase_text not in self.stop_words:
                    masked_phrases.append(phrase_text)
        return masked_phrases 


    def _generate_questions(self, summaries, entities=True, phrase_types=["NP"]):
        doc_ids = []
        qa_masks = []
        tokenized_phrases = []

        for id_, summary in tqdm(enumerate(summaries),total=len(summaries)):
            summary = summary.strip()
            all_masked_phrases = []
            if entities:
                all_masked_phrases.extend(self._get_entities(summary))
            all_masked_phrases.extend(self._get_masked_phrases(summary,phrase_types))
            all_masked_phrases = list(set(all_masked_phrases))

            for i, masked_phrase in enumerate(all_masked_phrases):
                tokenized_summary = " ".join(nltk.word_tokenize(summary.lower()))
                tokenized_phrase = " ".join(nltk.word_tokenize(masked_phrase.lower()))

                qa_masks.append(tokenized_summary + " [SEP] " + tokenized_phrase)
                doc_ids.append(str(id_))
                tokenized_phrases.append(tokenized_phrase)

        questions = []
        for i in tqdm(range(0, len(qa_masks), self.batch_size)):
            batch = qa_masks[i:i + self.batch_size]
            hypotheses = self.qg_model.sample(batch, beam=self.beam_size, lenpen=1.0, max_len_b=self.max_length, min_len=1, no_repeat_ngram_size=3)
            questions.extend(hypotheses)


        return doc_ids, questions, tokenized_phrases

    def _convert_to_squad_format(self, gold_answers, questions, doc_ids, documents):
        squad_format = {"data":[]}
        
        id_questions=defaultdict(list)
        id_gold_answers=defaultdict(str)

        for idx in range(0,len(doc_ids)):
            id_questions[doc_ids[idx].strip()].append((questions[idx], gold_answers[idx]))
        
        for idx in id_questions:
            paragraphs = []
            context = documents[int(idx)].strip()

            title = "doc_" + str(idx)
            
            questions_list_input=[]
            for q_id, question in enumerate(id_questions[idx]):

                gold_answer = question[1]
                question_text = question[0]
                answers_input = [{"text": gold_answer, "answer_start": 0}]
                questions_input = {
                                    "question": question_text, 
                                    "answers": answers_input, 
                                    "id": str(idx).strip() + "-" + str(q_id)
                                    }
                questions_list_input.append(questions_input) 
                id_gold_answers[questions_input["id"]] = gold_answer      

            
            paragraphs.append({"context":" ".join(nltk.word_tokenize(context)).lower(),"qas":questions_list_input})
            squad_format["data"].append({"title":title,"paragraphs":paragraphs})

            
        squad_format["version"] = "1.1"

        return id_gold_answers, squad_format


    def _run_squad(self, squad_input):
        with TemporaryDirectory() as tmpdir:
            squad_input_file = os.path.join(tmpdir, 'squad_input.json')
            with open(squad_input_file, 'w') as fout:
                json.dump(squad_input, fout)
            cmd = self.squad_cmd.format(squad_input_file, tmpdir)
            ret = subprocess.check_output(cmd, shell=True)

            with open(os.path.join(tmpdir, 'predictions_.json')) as fin:
                squad_output = json.load(fin)

        return squad_output

    def _compute_f1(self, a_gold, a_pred):
        gold_toks = nltk.word_tokenize(a_gold)
        pred_toks = nltk.word_tokenize(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _compute_f1_list(self, a_gold_list, a_pred_list):
        f1_list=[]
        for a_gold,a_pred in zip(a_gold_list, a_pred_list):
            f1_list.append(self._compute_f1(a_gold,a_pred))
        return np.mean(f1_list)


    def compute_score(self, documents, summaries, aggregate=False):
        #generate questions from summaries
        print("Generating questions...")
        doc_ids, questions, gold_answers = self._generate_questions(summaries)
        print("Getting answers...")
        #run qa system
        gold_answers_dict, squad_format = self._convert_to_squad_format(gold_answers, questions, doc_ids, documents)
        predictions_dict = self._run_squad(squad_format)


        doc_questions=defaultdict(dict)
        print("Computing metrics...")
        for qa_id in gold_answers_dict:
            doc_id, question_id=qa_id.split("-")
            prediction = predictions_dict[qa_id]
            if doc_id in doc_questions:
                doc_questions[doc_id]["preds"].append(prediction)
                doc_questions[doc_id]["gold"].append(gold_answers_dict[qa_id])
            else:
                doc_questions[doc_id]={"preds":[prediction],"gold":[gold_answers_dict[qa_id]]}

        doc_f1 = defaultdict(float)
        
        for idx in range(0,len(documents)):
            idx=str(idx)
            try:
                f1 = self._compute_f1_list(doc_questions[idx]["gold"],doc_questions[idx]["preds"])
                doc_f1[idx] = f1
            except:
                doc_f1[idx] = 0
                
        for id_, summary in enumerate(summaries):
            if str(id_) not in doc_f1:
                doc_f1[str(id_)] = 0
                
        if aggregate:
            return np.mean(list(doc_f1.values()))

        else:
            return [doc_f1[k] for k in sorted(doc_f1.keys())]