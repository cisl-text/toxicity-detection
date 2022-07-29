import json
from random import shuffle
from os.path import join
def load_data(file_name, mode):
    res_data = []
    with open(file_name, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            # todo: 需要处理文本
            text = line.strip()
            if mode == 'none':
                # (text, toxic, implicit)
                res_data.append({'text':text, 'label':'0'})
            else:
                res_data.append({'text':text, 'label':'1'})# (text, 1)
    return res_data

if __name__ == '__main__':
    split_ratio = 0.8
    data_dir = r"ImplicitHate"

    toxic_data = load_data(join(data_dir, "explicit.txt"), mode="toxic") + load_data(join(data_dir, "implicit.txt"), mode="toxic")
    non_toxic = load_data(join(data_dir, "non_toxic.txt"), mode="none")[:len(toxic_data)]

    shuffle(non_toxic)
    shuffle(toxic_data)
    # toxic_data = toxic_data[:2000]
    # non_toxic = non_toxic[:2000]
    print(f"toxic:{len(toxic_data)}", f"non-toxic:{len(non_toxic)}")

    split_non, split_toxic = int(len(non_toxic) * split_ratio), int(len(toxic_data) * split_ratio)
    train_data = non_toxic[:split_non] + toxic_data[:split_toxic]
    test_data = non_toxic[split_non:] + toxic_data[split_toxic:]

    shuffle(train_data)
    shuffle(test_data)
    train_dic, test_dic = {str(i):d for i, d  in enumerate(train_data)}, {str(i):d for i, d  in enumerate(test_data)}
    all_data = {
        'train': train_dic,
        'test': test_dic
    }

    json.dump(all_data, open("./implicit_split.json", 'w', encoding='utf-8'))
