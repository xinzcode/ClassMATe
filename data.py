import json
import torch


class Data:
    def __init__(self):

        self.seed = 2024

        # dataset & task
        self.dataset_path = "./dataset"
        self.datasets_dict = {'matscholar.json':0,'sofc_token.json':1,'synthesis_procedures.json':2,'sc_comics.json':3,'glass_non_glass.json':4,'structured_re.json':5,'synthesis_actions.json':6,'sofc_sent.json':7}
        self.datasets = [0]
        self.lower = False
        self.task_type_dict = {'ner':0,'pc':1,'rc':2,'ee':3,'sar':4,'sc':5,'sf':6}
        self.tasks = [0]
        # 0：0
        # 1：0、6
        # 2：0、2、3
        # 3：0、2、3
        # 4：1
        # 5：2
        # 6：4
        # 7：5

        # data split
        self.train_size = 0.9 # Test with partial data first
        self.even_split = True
        self.default_train_num = 1
        self.max_text_len = 2048

        # data save
        mark = str(self.train_size)
        self.gen_train_data_path = "./dataset/convert_data/{}_train.json".format(mark)
        self.gen_test_data_path = "./dataset/convert_data/{}_test.json".format(mark)

        # model
        self.model = "llama3"  # llama3 gpt4 gpt3.5

        # llama
        self.ckpt_dir = "./Meta-Llama-3-8B-Instruct-hf"
        # self.ckpt_dir = "./DeepSeek-R1-Distill-Llama-8B-hf"
        self.max_batch_size = 1
        self.max_seq_len = 10240
        self.max_gen_len = 2048
        self.temperature = 0.0
        self.top_p = 0.9
        self.print_dialog = True
        self.print_gold_pred = False

        # guideline
        self.retrieve = True
        self.gold_description_path = "dataset/guidelines/gold_description"

        # code
        self.code_style = True
        self.code_path = "./dataset/class_defs"
        self.use_gold_description = True
        self.gen_class = True
        self.property_num = 5
        self.status = 2  # 0,1,2

        # text
        self.use_description = True
        self.use_properties = False  # must after class_gen

        # learning
        self.learning = False
        self.use_learned = False
        self.learning_iteration = 3
        self.learning_size = 100
        self.learning_description_path = "dataset/guidelines/gold_description_learning"
