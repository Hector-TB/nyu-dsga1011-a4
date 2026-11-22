import pickle
from load_data import load_lines

pred_sql = load_lines('results/t5_ft_experiment_dev.sql')
gold_sql = load_lines('data/dev.sql')
nl       = load_lines('data/dev.nl')
pred_recs, error_msgs = pickle.load(open('records/t5_ft_experiment_dev.pkl', 'rb'))
gold_recs, _          = pickle.load(open('records/ground_truth_dev.pkl', 'rb'))

