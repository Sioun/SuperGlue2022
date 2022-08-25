import matplotlib.pyplot as plt

train_loss = []
train_axis = []

with open('logs\log.txt') as f:
    lines = f.readlines()
    for line in lines:
        if 'Loss:' in line:
            # print(float(line[-7:]))
            train_loss.append(float(line[-7:]))

for i in range(0,len(train_loss)):
    train_axis.append(100*i)

plt.plot(train_axis,train_loss,label="train_loss")
plt.grid()
leg0 = plt.legend(loc='upper right')
plt.show()