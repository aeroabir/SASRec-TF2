from collections import defaultdict
import sys

def data_process_with_time(fname, pname, sep=" "):
    User = defaultdict(list)
    Items = set()
    user_dict, item_dict = {}, {}

    with open('data/%s.txt' % fname, 'r') as fr:
        for line in fr:
            u, i, t = line.rstrip().split(sep)
            User[u].append((i, t))
            Items.add(i)
    
    print(len(User), len(Items))
    item_count = 1
    for item in Items:
        item_dict[item] = item_count
        item_count += 1

    count_del = 0
    user_count = 1
    with open('data/%s.txt' % pname, 'w') as fw:
        for user in User.keys():
            if len(User[user]) <= 2:
                del User[user]
                count_del += 1
            else:
                # user_dict[user] = user_count
                items = sorted(User[user], key=lambda x: x[1])
                items = [item_dict[x[0]] for x in items]
                for item in items:
                    fw.write(str(user_count) + ' ' + str(item) + '\n')
                user_count += 1

    print(user_count-1, count_del)


if __name__ == "__main__":

    data_process_with_time("ae_original", "ae", "\t")