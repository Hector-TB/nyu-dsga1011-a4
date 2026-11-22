from load_data import load_lines
from transformers import T5TokenizerFast

data_folder = 'data'
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

train_nl = load_lines(f'{data_folder}/train.nl')
train_sql = load_lines(f'{data_folder}/train.sql')
dev_nl = load_lines(f'{data_folder}/dev.nl')
dev_sql = load_lines(f'{data_folder}/dev.sql')

def token_stats(lines):
    ids = tokenizer(lines)['input_ids']
    token_lens = [len(token) for token in ids]
    mean_len = sum(token_lens) / len(token_lens)
    vocab = set()
    for sentence in ids:
        for id in sentence:
            vocab.add(id)
    vocab_size = len(vocab)
    return mean_len, vocab_size

train_nl_mean, train_nl_size = token_stats(train_nl)
train_sql_mean, train_sql_size = token_stats(train_sql)
dev_nl_mean, dev_nl_size = token_stats(dev_nl)
dev_sql_mean, dev_sql_size = token_stats(dev_sql)

prep_train_sql = [sql.replace("<", "less_than") for sql in train_sql]
prep_dev_sql = [sql.replace("<", "less_than") for sql in dev_sql]

prep_train_sql_mean, prep_train_sql_size = token_stats(prep_train_sql)
prep_dev_sql_mean, prep_dev_sql_size = token_stats(prep_dev_sql)

print('examples:', len(train_nl), len(dev_nl))
print('nl mean:', train_nl_mean, dev_nl_mean)
print('nl vocab:', train_nl_size, dev_nl_size)
print('sql mean:', train_sql_mean, dev_sql_mean)
print('sql vocab:', train_sql_size, dev_sql_size)
print('prep sql mean:', prep_train_sql_mean, prep_dev_sql_mean)
print('prep sql vocab:', prep_train_sql_size, prep_dev_sql_size)
