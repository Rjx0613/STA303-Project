import argparse
import functools
import csv
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/res2net.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                  '是否使用GPU预测')
add_arg('audio_path',       str,    '/openbayes/input/input0/UrbanSound8K/audio/fold5/156634-5-2-5.wav', '音频路径')
add_arg('model_path',       str,    'models/Res2Net_Fbank/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

file_path = '/openbayes/home/AudioClassification-Pytorch/test_list.txt'
title = []

with open(file_path, 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 使用制表符 '\t' 分割每一行的元素
        elements = line.strip().split('\t')
        
        # 获取每一行的第一个元素
        first_element = elements[0]
        # 打印或处理第一个元素
        title.append(first_element)
final = []         
predict = []
            
for i in range(len(title)): 
    try:
        label, score = predictor.predict(audio_data=title[i])
            # 打印或处理预测结果
        # print(f'音频：{title[i]} 的预测结果标签为：{label}，得分：{score}')
        predict.append(label)
        final.append(title[i])
    except AssertionError as e:
        continue

    
# 数据
def cut_name(name):
    c = 0
    for i in range(len(name)):
        if name[i] == "/":
            c+=1
            if c == 7:
                return name[i+1:]
    return None


def read_csv_to_dict(file_path):
    data_dict = {}
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # 读取并保存第一行（表头）
        
    # 读取剩余行
        rows = [row for row in reader]
        for row in rows:
            if row:  # 确保行不为空
                key = row[0]  # 每行的第一个元素作为键
                value = row[-2]  # 每行的倒数第二个元素作为值
                data_dict[key] = value
    
    return data_dict


csv_file_path = '/openbayes/home/AudioClassification-Pytorch/UrbanSound8K.csv'  
result_dict = read_csv_to_dict(csv_file_path)
dict_label = {"air_conditioner":0,"car_horn":1,"children_playing":2,"dog_bark":3,"drilling":4,"engine_idling":5,"gun_shot":6,"jackhammer":7,"siren":8,"street_music":9}


count = 0 
for i in range(len(predict)):

    ID = str(dict_label[predict[i]])
    name = cut_name(final[i])
    if ID == result_dict[name]:
        count += 1
   
print(count/len(predict))

# 定义文件路径
file_path = "/openbayes/home/AudioClassification-Pytorch/output.txt"

for i in range(len(final)):
    final[i] = cut_name(final[i])
    

# 打开文件并写入元素
with open(file_path, "w") as file:
    for element in final:
        file.write(element + "\n")
    