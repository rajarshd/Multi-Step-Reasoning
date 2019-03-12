
# Paragraph encoder

* If you just want to use the paragraph representation used in the paper, please download the pretrained vectors. Refer to the data section of this [README](https://github.com/rajarshd/multi-step-for-multi-hop#data) for more details.


Please run the following commands from the top-level directory.

## Training
```
python paragraph_encoder/train_para_encoder.py --data_dir data/ --src quasart|searchqa|triviaqa --embed_dir data/embeddings --model_dir model_save_dir 
```

### To save the vectors using the pretrained models
```
python paragraph_encoder/train_para_encoder.py --eval_only 1 --pretrained /path/to/model_save_dir/model.mdl --src quasart|searchqa|triviaqa --save_dir data/
```


