import matplotlib.pyplot as plt

data = []

with open('./tinyml_contest_data_training/S45-VPD-1.txt') as file:
    for line in file:
        data.append(float(line.strip()))

time = [x for x in range(len(data))]

plt.plot(time, data)
plt.show()
plt.close()