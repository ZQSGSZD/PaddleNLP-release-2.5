from pprint import pprint
from paddlenlp import Taskflow
schema = {
    '项目名称': [
        '结果',
        '单位',
        '参考范围'

    ]
}
my_ie = Taskflow("information_extraction", schema=schema, task_path='../checkpoint/model_best', precison='fp16')
doc_path = "./img_2.png"
pprint(my_ie({"doc": doc_path}))