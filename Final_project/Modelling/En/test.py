from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inp = ['jen batchelor ceo nonalcoholic beverage brand kin euphorics told cnn sober curiosity real thing long time says thought two options night go drinking stay home nothing',
       'case pushed people pura luka vega supposedly offended filipinos large decided weaponize existing laws lgbt people conde told cnn emailed statement']



tokens = tokenizer(inp, padding="max_length", max_length=40, truncation=True, return_tensors="tf")
print(tokens['input_ids'].shape)



# print(tokenizer.decode(tokens["input_ids"][0]))
