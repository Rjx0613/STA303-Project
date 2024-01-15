predict = []
final = []
def cut(str):
    count = 0
    for i in str:
        if i == "/":
            count+=1
            if count == 7:
                return str[i+1:]


def read_csv_to_dict(file_path):
    data_dict = {}
    
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        for row in csv_reader:
            if row:  # 确保行不为空
                key = row[0]  # 每行的第一个元素作为键
                value = row[-2]  # 每行的倒数第二个元素作为值
                data_dict[key] = value
    
    return data_dict

import csv



    
with open("/openbayes/home/output.csv", 'r') as csvfile:
    csv_reader = csv.reader(csvfile)    
    for row in csv_reader:
        if row:  # 确保行不为空
            final.append(row[0])  # 每行的第一个元素作为键
            predict.append(row[1])  # 每行的倒数第二个元素作为值


# 用法示例
csv_file_path = '/openbayes/home/AudioClassification-Pytorch/UrbanSound8K.csv'  # 替换为你的CSV文件路径
result_dict = read_csv_to_dict(csv_file_path)

dict_label = {"air_conditioner":0,"car_horn":1,"children_playing":2,"dog_bark":3,"drilling":4,"engine_idling":5,"gun_shot":6,"jackhammer":7,"siren":8,"street_music":9}

count = 0 
for i in range(len(predict)):
    ID = str(dict_label[predict[i]])
    print(cut(final[i])
    if ID == result_dict[cut(final[i])]:
        count += 1

print(count/len(predict))
