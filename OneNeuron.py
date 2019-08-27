import pandas as pd
import matplotlib.pyplot as plt
import numpy
import scipy as sc
import torch
from torch.nn import Linear, Sigmoid

data = pd.read_csv("apples_pears.csv")


X = data.iloc[:,:2].values  # матрица объекты-признаки
y = data['target'].values.reshape((-1, 1))  # классы (столбец из нулей и единиц)


num_features = X.shape[1]

neuron = torch.nn.Sequential(
    Linear(num_features, out_features=1),
    Sigmoid()
)

#Проба на необученном
#proba_pred = neuron(torch.tensor(X).float())
#y_pred = proba_pred > 0.5
#y_pred = y_pred.numpy().reshape(-1)

#plt.figure(figsize=(10, 8))
#plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred, cmap='spring')
#plt.title('Яблоки и груши', fontsize=15)
#plt.xlabel('симметричность', fontsize=14)
#plt.ylabel('желтизна', fontsize=14)
#plt.show()

#!!!!!!!!!!!!!!!!!!
X = torch.tensor(X).float()
y = torch.tensor(y).float()

#Код обучения
# квадратичная функция потерь (можно сделать другую)
loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.NLLLoss()

#loss_fn = torch.nn.CrossEntropyLoss()
# шаг градиентного спуска (точнее -- метода оптимизации)
learning_rate = 0.1  # == 1e-3
# сам метод оптимизации нейросети (обычно лучше всего по-умолчанию рабоатет Adam)

optimizer = torch.optim.SGD(neuron.parameters(), lr=learning_rate)
# итерируемся num_epochs раз, здесь 500
for t in range(10000):
    # foward_pass() -- применение нейросети (этот шаг ещё называют inference)
    y_pred = neuron(X)

    # выведем loss
    loss = loss_fn(y_pred, y)
    print('{} {}'.format(t, loss))

    # ВСЕГДА обнуляйте градиенты перед backard_pass'ом
    # подробнее: читайте документацию PyTorch
    optimizer.zero_grad()

    # backward_pass() -- вычисляем градиенты loss'а по параметрам (весам) нейросети
    # ВНИМАНИЕ! На это шаге мы только вычисляем градиенты, но ещё не обновляем веса
    loss.backward()

    # А вот тут уже обновляем
    optimizer.step()


proba_pred = neuron(X)
y_pred = proba_pred > 0.5
y_pred = y_pred.numpy().reshape(-1)

plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=y_pred, cmap='spring')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()