!pip install pytorch_transformers
!pip install fast_bert
import torch
import logger
from pytorch_transformers.tokenization import BertTokenizer
from fast_bert.data import BertDataBunch
from fast_bert.learner import BertLearner
from fast_bert.metrics import accuracy
device = torch.device('cuda')
logger = logging.getLogger()

metrics = [{'name': 'accuracy', 'function': accuracy}]
tokenizer = BertTokenizer.from_pretrained
 ('bert-base-uncased',
 do_lower_case=True)
databunch = BertDataBunch([PATH_TO_DATA],
 [PATH_TO_LABELS],
 tokenizer,
train_file=[TRAIN_CSV],
 val_file=[VAL_CSV],
 test_data=[TEST_CSV],
 text_col=[TEST_FEATURE_COL], label_col=[0],
 bs=64,
maxlen=140,
multi_gpu=False,
multi_label=False)
learner = BertLearner.from_pretrained_model(databunch,
 'bert-base-uncased',
metrics,
device,
logger,
is_fp16=False,
multi_gpu=False,
multi_label=False)
learner.fit(3, lr='1e-2')