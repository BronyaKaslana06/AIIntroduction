# 使用DQN算法玩cartPole

DQN算法玩的游戏：cartPole

[Cart Pole - Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

目前大概收敛在60左右



## 运行

```shell
pip install torch
pip install gym
pip install pygame
```

若报错按照报错中的提示安装相应的库

安装好后，在项目根目录中运行mainTrain.py文件

```shell
python mainTrain.py
```

运行成功后可以看到cartPole游戏界面

![](img\pygame.jpg)

控制台中显示的当前训练的轮数(Episode)和当前的奖励(Avg.Reward)

![](img\console.png)

当到达一定训练次数后会自动退出
