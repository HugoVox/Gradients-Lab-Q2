from datasets import load_dataset
import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
from python_on_whales import DockerClient
import gdown
import patoolib
import os

def check_database_folder():
    """Check if database folder exists, if not, download and extract it."""
    data_path = "data"
    if not ('pgdata' in os.listdir(data_path)):
        if not ('pgdata.rar' in os.listdir(data_path)):
            print("Download database data from GDrive...")
            url = 'https://drive.google.com/uc?id=17o-lXn0VWi6W7K9XUuG7qfqssfKlBq5F'
            output = "data\pgdata.rar"
            gdown.download(url, output, quiet=False)
        print("Extracting database data...")
        patoolib.extract_archive("data\pgdata.rar", 
                                 outdir="data")

def init_database():
    """Initialize database from docker compose
    """
    docker = DockerClient(compose_files=["./docker-compose.yml"])
    docker.compose.up(detach=True)

def close_database():
    """Close database in docker compose
    """
    docker = DockerClient(compose_files=["./docker-compose.yml"])
    docker.compose.down()

def import_dataset():
    '''
    Returns the wiki_snippets data and the embedded data from PostgreSQL.

            Returns:
                    wiki (DataDict): The wiki_snippets data.
                    data (list): The embedded data from PostgreSQL.

    '''
    wiki = load_dataset('wiki_snippets', 'wiki40b_en_100_0', split='train')
    conn = psycopg2.connect(database="eli5",
                            host="127.0.0.1",
                            user="admin",
                            password="1234",
                            port="6000")

    cursor = conn.cursor()

    return wiki, conn, cursor

def retrieve_model():
    '''
    Returns the sentence transformer model.

            Returns:
                    sbert_model (SentenceTransformer): The sentence transformer model.

    '''
    sbert_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    return sbert_model

def generate_model(model_id = "eli5_bart_model", backbone = "yjernite/bart_eli5", device = "cuda:0"):
    '''
    Returns the inference model and the tokenizer.

            Parameters:
                            model_id (str): The id of LoRA model. (default is "eli5_bart_model")
                            backbone (str): The backbone of LoRA model. (default is "yjernite/bart_eli5")
                            device (str): The device to use. (default is "cuda:0")

            Returns:
                    inference_model (PeftModel): The inference model.
                    tokenizer (AutoTokenizer): The tokenizer.

    '''
    model = AutoModelForSeq2SeqLM.from_pretrained(
        backbone,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    inference_model = PeftModel.from_pretrained(model=model, model_id=model_id)
    inference_model.print_trainable_parameters()
    return inference_model, tokenizer

def query(question, retrieve_model, conn, cursor, wiki, k=5):
    '''
    Returns the top-k closest contexts based on the question by .

            Parameters:
                            question (str): The question to be answered.
                            retrieve_model (SentenceTransformer): The sentence transformer model.
                            cursor (_Cursor): The cursor of PostgreSQL.
                            wiki (DataDict): The wiki_snippets data.
                            k (int): The number of top results. (default is 5)

            Returns:
                    context (str): The context of the question.

    '''
    q_embed = str(retrieve_model.encode(question).tolist())
    query = f"SELECT id FROM wiki40b ORDER BY data_encoded <=> '{q_embed}' DESC LIMIT {str(k)};"
    cursor.execute(query)
    conn.commit()
    data = cursor.fetchall()
    context = ''.join(map(str, [wiki[x]['passage_text'] for x in data]))
    # print('Top 5 results:', best_ids)
    return context

def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device="cuda:0"):
    '''
    Returns the batch of question and answer for the inference model.

            Parameters:
                            qa_list (list): The list of question and answer.
                            tokenizer (AutoTokenizer): The tokenizer.
                            max_len (int): The maximum length of the question. (default is 64)
                            max_a_len (int): The maximum length of the answer. (default is 360)
                            device (str): The device to use. (default is "cuda:0")
            Returns:
                    model_inputs (dict): The batch of question and answer.
    '''
    q_ls = [q for q, a in qa_list]
    a_ls = [a for q, a in qa_list]
    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks["input_ids"]).to(device),
        torch.LongTensor(q_toks["attention_mask"]).to(device),
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), pad_to_max_length=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks["input_ids"]).to(device),
        torch.LongTensor(a_toks["attention_mask"]).to(device),
    )
    lm_labels = a_ids[:, 1:].contiguous().clone()
    lm_labels[a_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        "input_ids": q_ids,
        "attention_mask": q_mask,
        "decoder_input_ids": a_ids[:, :-1].contiguous(),
        "labels": lm_labels,
    }
    return model_inputs

def generate_answer(
    question_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=64,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device="cuda:0",
):
    '''
    Returns the generated answer based on the question and context.

            Parameters:
                            question_doc (str): The context of the question.
                            qa_s2s_model (PeftModel): The inference model.
                            qa_s2s_tokenizer (AutoTokenizer): The tokenizer.
                            num_answers (int): The number of answers. (default is 1)
                            num_beams (int): The number of beams. (default is None)
                            min_len (int): The minimum length of the answer. (default is 64)
                            max_len (int): The maximum length of the answer. (default is 256)
                            do_sample (bool): Whether to use sampling. (default is False)
                            temp (float): The temperature of sampling. (default is 1.0)
                            top_p (float): The top p of sampling. (default is None)
                            top_k (int): The top k of sampling. (default is None)
                            max_input_length (int): The maximum length of the input. (default is 512)
                            device (str): The device to use. (default is "cuda:0")
            Returns:
                    answers (List): The generated answers.
    '''
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], qa_s2s_tokenizer, max_input_length, device=device,)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = qa_s2s_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=qa_s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
    )
    answers = [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]
    return answers
