import matplotlib.pyplot as plt
from random_data import *
from quest_net import *
from topology import *



if __name__ == '__main__':
    topo=Topology()
    num_nodes = 100
    num_features = 1
    quest_range = np.array_split(np.arange(num_nodes),3)
    min_quest,max_quest= 5,10
    num_quests = np.random.randint(min_quest, max_quest, 3)
    model = QuestNet(num_features, 4, num_quests)
    epochs = range(400)
    graph, A = random_network(num_nodes)

    res=[]
    for epoch in epochs:
        x = random_features(num_nodes, num_features)

        slices, label = random_slices(num_nodes,10,quest_range,num_quests)
        inputs = [x, A, slices]
        pred=model(x, A, slices)
        # if epoch %20 ==0 :
        #     print(pred)
        #     print(label)
        current_loss = loss(pred, label).numpy()
        res.append(current_loss)
        train(model, inputs, label, learning_rate=0.1)  # need to add batch training
    plt.plot(epochs, res, 'r')
    plt.show()