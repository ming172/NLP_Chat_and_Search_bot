from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_id = "./ChatLM-mini-Chinese"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).to(device)

def use_ChatLM(input):
    txt = input

    encode_ids = tokenizer([txt])
    input_ids, attention_mask = torch.LongTensor(encode_ids['input_ids']), torch.LongTensor(encode_ids['attention_mask'])

    outs = model.my_generate(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        max_seq_len=256,
        search_type='beam',
    )

    outs_txt = tokenizer.batch_decode(outs.cpu().numpy(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return outs_txt[0]

if __name__=='__main__':
    while True:
        i=input('You: ')
        out=use_ChatLM(i)
        out='Bot: '+out
        print(out)
