import random

random.seed(0)

data_path = []
for geom in ['Cre', 'Tri', 'Spl']:
    for i in range(1, 161):
        for k in [1, 2]:
            data_path.append((f"{geom}/{i}/{k}"))

random.shuffle(data_path)

with open("train.txt", "w") as f:
    for i in range(0, len(data_path)):
        f.write(data_path[i] + "\n")


data_path = []
for geom in ['Cre', 'Tri', 'Spl']:
    for i in range(161, 181):
        for k in [1, 2]:
            data_path.append((f"{geom}/{i}/{k}"))

random.shuffle(data_path)

with open("valid.txt", "w") as f:
    for i in range(0, len(data_path)):
        f.write(data_path[i] + "\n")

data_path = []
for geom in ['Cre', 'Tri', 'Spl']:
    for i in range(181, 201):
        for k in [1, 2]:
            data_path.append((f"{geom}/{i}/{k}"))

random.shuffle(data_path)

with open("test.txt", "w") as f:
    for i in range(0, len(data_path)):
        f.write(data_path[i] + "\n")