import torch

def custom_collate_fn_BM(batch):
    # 将批次中的数据分离为数据和标签
    # print(batch)
    data = [item[0] for item in batch]
    labels = []
    for item in batch:
        label = [0.] * 6
        action = int(item[1][0])
        label[action] = 1.
        labels.append(label)
    data = torch.tensor(data)

    labels = torch.tensor(labels)

    return data, labels

def custom_collate_fn_BA(batch):
    # 将批次中的数据分离为数据和标签
    # print(batch)
    data = [item[0] for item in batch]
    labels = []
    for item in batch:
        label = [0.] * 6
        action = int(item[1][0])
        label[action] = 1.
        labels.append(label)
    data = torch.tensor(data)

    labels = torch.tensor(labels)

    return data, labels

def switch_prob2user_ret(switch_prob):
    user_ret = [1.0]
    for i in range(len(switch_prob)):
        user_ret.append(user_ret[i] - switch_prob[i])
    user_ret[-1] = 0.
    return user_ret

def user_ret2switch_prob(user_ret):
    switch_prob = [0.] * (len(user_ret) - 2)
    for i in range(1, len(switch_prob) + 1):
        switch_prob[i - 1] = user_ret[i - 1] - user_ret[i]
    switch_prob[-1] += user_ret[-2]
    return switch_prob