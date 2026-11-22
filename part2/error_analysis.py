import pickle
from collections import Counter
from load_data import load_lines

pred_sql = load_lines('results/t5_ft_experiment_dev.sql')
gold_sql = load_lines('data/dev.sql')
nl       = load_lines('data/dev.nl')
pred_recs, error_msgs = pickle.load(open('records/t5_ft_experiment_dev.pkl', 'rb'))
gold_recs, _          = pickle.load(open('records/ground_truth_dev.pkl', 'rb'))

for i, error_msg in enumerate(error_msgs):
    if error_msg == '':
        continue
    print(i, error_msg)

c = Counter(error_msgs)
print(c.most_common())


for i in [39, 52, 66, 2, 116, 129, 23, 40, 160, 228]:
    print('\n#', i, error_msgs[i])
    print('NL:  ', nl[i])
    print('PRED:', len(pred_sql[i]), pred_sql[i][-250:])
    print('GOLD:', len(gold_sql[i]), gold_sql[i][-250:])
        