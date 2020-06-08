# CS397SeqAdapative

This is for developing codes of Sequence Adaptive Transformer.

Data Link: https://ai.stanford.edu/~amaas/data/sentiment/

Original Code link: https://github.com/SamLynnEvans/Transformer

# How to train?
python3 "./train21_v.py" -src_data "./train.txt" -trg_data "./label_train.txt" -src_datav "./test.txt" -trg_datav "./testlabel.txt" -aaa 100 -src_lang en -trg_lang en -n_layers 4 -restart 0 -bestval 0 -batchsize 200 -epochs 200
