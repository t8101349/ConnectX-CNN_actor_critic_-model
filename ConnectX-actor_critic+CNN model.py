import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import PIL.Image

import tensorflow as tf
import logging

from sklearn import preprocessing
from tensorflow.keras.optimizers import Adam
import random
import matplotlib.pyplot as plt
import seaborn as sns

from kaggle_environments import evaluate, make
#from kaggle_environments.envs.connectx.helpers import *
"""
# 設置隨機種子以確保結果的可重現性
seed = 123
tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
"""
# 使用環境變量設置線程數量，在 TensorFlow 初始化之前
os.environ['TF_INTRA_OP_PARALLELISM_THREADS'] = '1'
os.environ['TF_INTER_OP_PARALLELISM_THREADS'] = '1'


# training mode or just submitting
training = True


##Analyzing the environment
env = make('connectx', debug=True)
#env.run(['random'])
env.render(mode="ipython", width=800, height=600)

env.agents

print(env.configuration)

print(env.specification)

env.specification.reward

env.specification.action

env.specification.observation


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

# 初始化 Config 
config_dict = {
    'episodeSteps': 1000,        
    'actTimeout': 2,             
    'runTimeout': 1200,          
    'columns': 7,                
    'rows': 6,                   
    'inarow': 4,                 
    'agentTimeout': 60,          
    'timeout': 2                 
}

config = Config(config_dict)

obs = env.specification.observation
#grid = np.asarray(obs.board).reshape(config.rows, config.columns)

# Configuration paramaters
eps = np.finfo(np.float32).eps.item() #minimize none zero value
gamma = 0.95  # Discount factor for past rewards
env = make('connectx', debug=True)
trainer = env.train([None,"random"])
board_size = 42 * 2    # my board + oppenent's board
n_actions = n_cols = 7
n_rows = 6
n_players = 2
input_dim = (n_rows, n_cols, n_players)


# 定義共享輸入層
def input_layer(in_):
    conv1 = tf.keras.layers.Conv2D(filters=42, kernel_size=(3,3), activation='relu', name='input_conv1')(in_)
    conv2 = tf.keras.layers.Conv2D(filters=42, kernel_size=(3,3), activation='relu', padding='same', name='input_conv2')(conv1)
    batch_norm = tf.keras.layers.BatchNormalization(name='input_batch_norm')(conv2)
    flat = tf.keras.layers.Flatten(name='input_flat')(batch_norm)
    dense = tf.keras.layers.Dense(256, activation='relu', name='input_dense')(flat)
    return dense

##The Actor-Critic model
def ActorModel(num_actions,in_):
    actions = tf.keras.layers.Dense(128, activation='tanh', name='actor_dense1')(in_)
    actions = tf.keras.layers.Dropout(0.2, name='actor_dropout1')(actions)
    actions = tf.keras.layers.Dense(64, activation ='tanh', name='actor_dense2')(actions)
    actions = tf.keras.layers.LayerNormalization(name='actor_layer_norm')(actions)
    actions = tf.keras.layers.Dense(num_actions, activation='softmax', name='actor_dense3')(actions)

    return actions 

def CriticModel(in_):
    hidden1 = tf.keras.layers.Dense(128, activation='relu', name='critic_dense1')(in_)
    hidden1 = tf.keras.layers.Dropout(0.2, name='critic_dropout1')(hidden1)
    hidden2 = tf.keras.layers.Dense( 64, activation='relu', name='critic_dense2')(hidden1)
    hidden2 = tf.keras.layers.LayerNormalization(name='critic_layer_norm')(hidden2)
    value = tf.keras.layers.Dense(1, name='critic_dense3')(hidden2)
    
    return value


input_ = tf.keras.layers.Input(shape=input_dim, name='input_layer')

# 定義 Actor 和 Critic 模型的輸出
shared_features = input_layer(input_)
actor_output = ActorModel(n_actions, shared_features)
critic_output = CriticModel(shared_features)


# 建立 Actor-Critic 模型
model = tf.keras.Model(inputs=input_, outputs=[actor_output, critic_output])


# 使用學習率調度器和梯度裁剪
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-3,
    decay_steps=10000,
    decay_rate=0.95
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)


#action_loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

critic_loss_function = tf.keras.losses.Huber()
print(model.summary())




## prepare board for use in nn

def get_board(state, player):

    # get player id 1, 2
    # board is 0, open move, 1 player 1, 2 for player 2    
    board1 = np.asarray([1 if z==1 else 0 for z in state]).reshape(n_rows, n_cols)
    board2 = np.asarray([1 if z==2 else 0 for z in state]).reshape(n_rows, n_cols)

    # one per player, put current player 1st
    if player == 1:
        b = np.concatenate([board1, board2], axis=0).reshape(2, n_rows, n_cols)
    else:
        b = np.concatenate([board2, board1], axis=0).reshape(2, n_rows, n_cols)

    # rotate to feed into convolutional network 
    return np.transpose(b, [1, 2, 0])

'''
def pretty_print_board(b):
# Flatten the array to ensure it's 1D
    b = b.flatten()
    for row in range(6):  # printout board
        print('%d %d %d %d %d %d %d' % tuple(b[row * 7:(row + 1) * 7]))


#test
def test_get_board():
    b = np.random.randint(0, 3, size=42)
    
    pretty_print_board(b)
    
    print('player 1')
    input = get_board(b, 1)
    pretty_print_board(input)

    print('player 2')
    input = get_board(b, 2)
    pretty_print_board(input)

    input = np.expand_dims(input, axis=0)

    print('input shape', input.shape)
    #print(input)

    print('top row')
    # batch, row, col, channels
    top_row = input[0,0,:,0] + input[0,0,:,1]
    print(top_row)
    
    action = model(input)
    print('action', action)

test_get_board()
'''



def plot_timesteps(steps, rewards, a_loss, c_loss):

    plt.figure(figsize=(20, 10))
    plt.title('Steps, rewards')
    plt.plot(steps)
    plt.plot(rewards)
    plt.axhline(y=1.)  # rewards >= 1 indicate game won
    plt.axhline(y=5.)  #min moves to lose a game
    plt.show()

    plt.figure(figsize=(20,10))
    plt.title('Actor, Critic losses')
    plt.plot(a_loss)
    plt.plot(c_loss)
    plt.show()
    

# log discounted rewards
def discount_rewards(rewards):
    T = len(rewards)
    discounts = np.logspace(0, T, num=T, base=gamma, endpoint=False)
    dr = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
   
    return dr


# for crossentropy
def expand_rewards(rewards, actions):
    # mask out action taken
    masks = tf.one_hot(actions, n_actions)
    # convert ones to rewards
    expanded = masks * rewards
    return expanded
    


## Prepare for Training
class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.clear_size = max_size // 4
        self.states = []
        self.actions = []
        self.rewards = []
        self.target_values = []
    
    def add(self, state, action, reward, target_value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.target_values.append(target_value)
        
        # 當超過最大容量時，清除最舊的記錄
        if len(self.states) > self.max_size:
            self.states = self.states[self.clear_size:]
            self.actions = self.actions[self.clear_size:]
            self.rewards = self.rewards[self.clear_size:]
            self.target_values = self.target_values[self.clear_size:]
    
    def get_all(self):
        return (self.states, self.actions, self.rewards, self.target_values)

"""
#reward function
def calculate_reward(raw_reward, done):
    if raw_reward is None:  # 非法移動
        return 0.0
    elif raw_reward == 0 and not done:  # 遊戲繼續
        return 1  # 存活獎勵
    elif raw_reward == 1:  # 獲勝
        return 1000000
    else:  # 失敗或平局
        return -10000
"""
    
# Helper function for calculate_reward: checks if window satisfies heuristic conditions
def check_window(window, num_discs, piece, config):
    return (np.sum(window == piece) == (num_discs) and np.sum(window == 0) == (config.inarow - num_discs))


# Helper function for calculate_reward: counts number of windows satisfying specified heuristic conditions
def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = np.asarray(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = np.asarray(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = np.asarray(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = np.asarray(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows



# Helper function for calculates value of heuristic for grid
def calculate_reward(grid, mark, config):
    num_threes = count_windows(grid, config.inarow-1, mark, config)
    num_fours = count_windows(grid, config.inarow, mark, config)
    num_threes_opp = count_windows(grid, config.inarow-1, mark%2+1, config)
    num_fours_opp = count_windows(grid, config.inarow, mark%2+1, config)
    score = 50*num_threes - 1e2*num_threes_opp - 1e4*num_fours_opp + 1e6*num_fours
    return score


def plot_progress(episode_data, title, window_size=10):
    plt.figure(figsize=(12, 6))
    data = np.array(episode_data)
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    plt.plot(data, alpha=0.3, label='Raw')
    plt.plot(moving_avg, label=f'Moving Average (window={window_size})')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def test_agent(model, n_games=12):
    #env = make('connectx', debug=True)

    # 重置環境
    obs = trainer.reset() 
    
    # 創建代理函數
    agent = my_agent(obs, config, model)
    
    # 使用 evaluate 進行評估
    print('********************  test agent **********************************')
    print(f'Testing for {n_games} games')
    
    # 對戰 negamax
    results_vs_negamax = evaluate("connectx", [agent, "negamax"], num_episodes=n_games)
    print(f"\nVs Negamax - Results: {results_vs_negamax}")  
    
    # 對戰 random
    results_vs_random = evaluate("connectx", ["random", agent], num_episodes=n_games)
    print(f"Vs Random - Results: {results_vs_random}")
    
    # 計算總體勝率和先後手勝利
    total_games = n_games * 2  # negamax + random
    first_player_wins = 0
    second_player_wins = 0
    total_wins = 0
    total_losses = 0
    total_ties = 0
    
    # 計算對戰 negamax 的結果
    for i, r in enumerate(results_vs_negamax):
        if r[0] and r[0] > r[1]:  # 獲勝
            total_wins += 1
            first_player_wins += 1
        elif r[0] and r[0] < r[1]:  # 失敗
            total_losses += 1
        else:  # 平局
            total_ties += 1
    
    # 計算對戰 random 的結果
    for i, r in enumerate(results_vs_random):
        if r[1] and r[1] > r[0]:  # 獲勝
            total_wins += 1
            second_player_wins += 1
        elif r[1] and r[1] < r[0]:  # 失敗
            total_losses += 1
        else:  # 平局
            total_ties += 1
    
    results = {
        'wins': total_wins,
        'losses': total_losses,
        'draws': total_ties,
        'first_player_wins': first_player_wins,
        'second_player_wins': second_player_wins,
        'vs_negamax': results_vs_negamax,
        'vs_random': results_vs_random
    }



    print("\nOverall Results:")
    print(f"Total Games: {total_games}")
    print(f"Total Wins: {total_wins} ({total_wins/total_games:.2%})")
    print(f"Total Losses: {total_losses} ({total_losses/total_games:.2%})")
    print(f"Total Draws: {total_ties} ({total_ties/total_games:.2%})")
    print(f"First Player Wins: {first_player_wins}")
    print(f"Second Player Wins: {second_player_wins}")
    
    # 檢查代理是否正常運作
    env = make('connectx', debug=True)
    test_game = env.run([agent, agent])
    if env.state[0].status == env.state[1].status == 'DONE':
        print('Agent functioning successfully')
    else:
        print('Agent failed to complete game')
    
    return results

def train_agent(model, save_path = '/kaggle/working/model_weights'):
    # 訓練參數設置
    max_episodes = 900  # 總共進行900場遊戲
    
    # 探索參數
    explore = 1.0
    decay_explore = 0.95
    min_explore = 0.01
    
    # 記憶體設置
    memory = ReplayMemory(max_size=1024)
    
    # 記錄訓練過程的數據
    episode_steps = []
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    
    # 保存最佳模型的變數
    best_reward = float('-inf')
    best_weights = None
    
    for episode in range(max_episodes):
        # 設置環境和對手
        env = make('connectx', debug=True)
        if episode % 2 == 0:   
            trainer = env.train(['negamax', None])
            player_mark = 2
        else:                  
            trainer = env.train([None, 'negamax'])
            player_mark = 1
            
        # 重置環境
        obs = trainer.reset()
        state = get_board(obs['board'], player_mark)
        state = np.expand_dims(state, axis=0)
        
        episode_reward = 0
        episode_memory = []
        done = False
        step = 0
        
        print(obs['mark'])
        print(config.inarow)
        
        
        # 進行一場完整的遊戲
        while not done:
            step += 1
            state_tensor = tf.convert_to_tensor(state)
            #print(state_tensor)
            action_probs, state_value = model(state_tensor, training=True)
            print(action_probs)
            #print(state_value)
            
            # 確保動作概率的維度正確
            if len(action_probs.shape) > 2:
                action_probs = tf.reduce_mean(action_probs, axis=[1,2])
            
            # 獲取有效移動
            
            valid_moves = [i for i in range(n_cols) if np.asarray(obs['board']).reshape(n_rows, n_cols)[0][i] == 0]
            #print(valid_moves)
            
            if not valid_moves:
                break
            
            # 決定是探索還是利用
            if np.random.random() < explore:
                action = np.random.choice(valid_moves)
                print("Chosen action:", action)
            else:
                # 根據action_probs提取合法動作的概率
                valid_probs = np.array([action_probs[0][i] for i in valid_moves])
                valid_probs /= np.sum(valid_probs)  # 正規化
                
                # 隨機選擇合法動作
                action = np.random.choice(valid_moves, p=valid_probs)
                print("Chosen action:", action)
                      
                
            # 執行動作
            obs, raw_reward, done, info = trainer.step(int(action))


            
            # 計算獎勵
            print(np.asarray(obs['board']).reshape(n_rows, n_cols))
            reward = calculate_reward(np.asarray(obs['board']).reshape(n_rows, n_cols), player_mark, config)
            print(reward)
            
            state = get_board(obs['board'], player_mark)
            state = np.expand_dims(state, axis=0)
            
            # 計算下一個狀態的價值
            if done:
                target_value = reward
                
            else:
                next_state = get_board(obs['board'], player_mark)
                next_state = np.expand_dims(next_state, axis=0)
                next_state_tensor = tf.convert_to_tensor(next_state)
                _, next_state_value = model(next_state_tensor, training=False)
                target_value = reward + gamma * next_state_value[0,0]
            
            # 存儲經驗
            episode_memory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'target_value': target_value
            })
            
            
            if not done:
                state = get_board(obs['board'], player_mark)
                state = np.expand_dims(state, axis=0)
        
        # 更新記憶體
        for exp in episode_memory:
            memory.add(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['target_value']
            )

        # 從記憶體中獲取訓練數據並進行訓練
        if len(memory.states) > 0:
            states = tf.concat(memory.states, axis=0)
            actions = tf.convert_to_tensor(memory.actions)
            target_values = tf.convert_to_tensor(memory.target_values, dtype=tf.float32)
            
            with tf.GradientTape(persistent=True) as tape:
                action_probs, predicted_values = model(states, training=True)
                
                if len(action_probs.shape) > 2:
                    action_probs = tf.reduce_mean(action_probs, axis=[1,2])
                
                # 創建 action masks
                action_masks = tf.one_hot(actions, n_actions)
                
                # 計算 log probabilities
                selected_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
                selected_log_probs = tf.math.log(selected_probs + eps)
                
                # 計算優勢
                advantages = target_values - tf.squeeze(predicted_values)
                actor_loss = -tf.math.reduce_sum(selected_log_probs * advantages)

                
                # Critic 損失
                critic_loss = critic_loss_function(target_values, tf.squeeze(predicted_values))

                
                # 獲取模型變量
                all_variables = model.trainable_variables
                actor_variables = [var for var in all_variables if 'actor' in var.path]
                critic_variables = [var for var in all_variables if 'critic' in var.path]
                shared_variables = [var for var in all_variables if 'input' in var.path]
        
    
    
                #  計算梯度
                actor_grads = tape.gradient(actor_loss, actor_variables + shared_variables)
                critic_grads = tape.gradient(critic_loss, critic_variables + shared_variables)
    
                # 合併共享層的梯度
                shared_grads_actor = tape.gradient(actor_loss, shared_variables)
                shared_grads_critic = tape.gradient(critic_loss, shared_variables)
    
                # 平均共享層梯度
                shared_grads = [
                    (ga + gc) / 2.0
                    for ga, gc in zip(shared_grads_actor, shared_grads_critic)
                ]
    
                # 更新參數
                actor_optimizer.apply_gradients(zip(actor_grads, actor_variables + shared_variables))
                critic_optimizer.apply_gradients(zip(critic_grads, critic_variables + shared_variables))
    
    
                # 更新共享層參數
                optimizer.apply_gradients(zip(shared_grads, shared_variables))
    
                actor_losses.append(float(actor_loss))
                critic_losses.append(float(critic_loss))
    
                del tape
        
        # 更新探索率
        explore = max(min_explore, explore * decay_explore)
        
        # 記錄訓練數據
        episode_steps.append(step)
        episode_rewards.append(episode_reward)
        
        # 每50回合顯示訓練進度
        if episode % 50 == 0:
            print(f"\nEpisode {episode}")
            print(f"Steps: {step}, Reward: {episode_reward:.4f}")
            print(f"Player Position: {'Second' if episode % 2 == 0 else 'First'}")
            print(f"Explore Rate: {explore:.4f}")
            if len(actor_losses) > 0:
                print(f"Actor Loss: {actor_losses[-1]:.4f}")
                print(f"Critic Loss: {critic_losses[-1]:.4f}")
                avg_target = np.mean([exp['target_value'] for exp in episode_memory])
                print(f"Avg Target Value: {avg_target:.4f}")
                
                # 計算移動平均獎勵
                recent_rewards = episode_rewards[-50:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Average Reward (last 50 episodes): {avg_reward:.4f}")
            
            # 繪製訓練進度圖
            plot_timesteps(
                episode_steps[-100:] if len(episode_steps) > 100 else episode_steps,
                episode_rewards[-100:] if len(episode_rewards) > 100 else episode_rewards,
                actor_losses[-100:] if len(actor_losses) > 100 else actor_losses,
                critic_losses[-100:] if len(critic_losses) > 100 else critic_losses
            )
        
        # 每100回合評估和保存模型
        if episode > 0 and episode % 100 == 0:
            # 測試當前模型
            test_results = test_agent(model, n_games=20)
            win_rate = test_results['wins'] / 40
            
            print("\nTest Results:")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"First Player Wins: {test_results['first_player_wins']}")
            print(f"Second Player Wins: {test_results['second_player_wins']}")
            print(f"Draws: {test_results['draws']}")
            
            # 如果性能提升，保存模型
            current_reward = win_rate
            

            if current_reward > best_reward:
                best_reward = current_reward
                best_weights = model.get_weights()
                
                
                model.save_weights(f'{save_path}/best_model.weights.h5')
                print(f"New best model saved with win rate: {win_rate:.2%}")
            
            # 繪製詳細的進度圖
            plot_progress(episode_rewards, 'Episode Rewards')
            plot_progress(actor_losses, 'Actor Losses')
            plot_progress(critic_losses, 'Critic Losses')
    
    # 訓練結束後的最終評估
    print("\nFinal Evaluation:")
    final_results = test_agent(model, n_games=20)
    
    # 計算對戰 negamax 和 random 的總場次
    total_games = 40  # 20場對戰negamax + 20場對戰random
    final_win_rate = final_results['wins']/total_games
        
    print(f"\nDetailed Results:")
    print(f"Total Games: {total_games}")
    print(f"Wins: {final_results['wins']}")
    print(f"Losses: {final_results['losses']}")
    print(f"Draws: {final_results['draws']}")
    
    print(f"Final Model Win Rate: {final_win_rate:.2%}")
    print(f"Best Model Win Rate: {best_reward:.2%}")
    
    # 比較最終模型和最佳模型的性能
    if final_win_rate > best_reward:
        print("Final model performs better than best model")
        model.save_weights(f'{save_path}/final_model.weights.h5')
        print(f"Final model saved with win rate: {final_win_rate:.2%}")
    else:
        print("Best model performs better than final model")
        if best_weights is not None:
            model.set_weights(best_weights)
            model.save_weights(f'{save_path}/final_model.weights.h5')
            print(f"Restored best model weights with win rate: {best_reward:.2%}")
            
    '''
    # 分別顯示對戰 negamax 和 random 的結果
    print("\nVs Negamax:")
    print(f"Results: {final_results['vs_negamax']}")
    
    print("\nVs Random:")
    print(f"Results: {final_results['vs_random']}")
    '''

    return model, {
        'episode_steps': episode_steps,
        'episode_rewards': episode_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'best_reward': best_reward,
        'final_reward': final_win_rate
    }



def my_agent(obs, config, model):
    # Number of Columns on the Board.
    columns = config.columns
    # Number of Rows on the Board.
    rows = config.rows
    # Number of Checkers "in a row" needed to win.
    inarow = config.inarow
    # The current serialized Board (rows x columns).
    board = obs.board
    # Which player the agent is playing as (1 or 2).
    mark = obs.mark
    
    # 獲取遊戲狀態
    state = get_board(board, mark)
    state = np.expand_dims(state, axis=0)
    state_tensor = tf.convert_to_tensor(state)
    
    # 使用模型預測動作概率和狀態價值
    action_probs, state_value = model(state_tensor, training=False)
    if len(action_probs.shape) > 2:
        action_probs = tf.reduce_mean(action_probs, axis=[1,2])
    
    # 獲取有效移動
    valid_moves = [i for i in range(n_cols) if np.asarray(board).reshape(rows, columns)[0][i] == 0]
    if not valid_moves:
        best_action = np.random.choice([i for i in range(n_cols)])

    else:            
        valid_probs = np.array([action_probs[0][i] for i in valid_moves])
        valid_probs /= np.sum(valid_probs)  # 正規化
        
        # 隨機選擇合法動作
        best_action = np.random.choice(valid_moves, p=valid_probs)
    
    
    return int(best_action)


#提交生成
import inspect
import os

def write_agent_to_file(function, file):
    with open(file, "a" if os.path.exists(file) else "w") as f:
        f.write(inspect.getsource(function))
        print(function, "written to", file)




if __name__ == "__main__":
    # 確保保存目錄存在
    os.makedirs('/kaggle/working/model_weights', exist_ok=True)

    # 創建環境
    env = make('connectx', debug=True)

    # 訓練模型
    trained_model, history = train_agent(model, save_path = '/kaggle/working/model_weights')
    
    
    # 載入最佳模型行最終測試
    best_weights = '/kaggle/working/model_weights/final_model.weights.h5'
    best_model = tf.keras.models.clone_model(model)
    best_model.load_weights(best_weights)
    
    # 進行最終測試
    final_test_results = test_agent(best_model, n_games=100)
    print("\nFinal Test Results (Best Model):")
    print(f"Win Rate: {final_test_results['wins']/100:.2%}")
    print(f"First Player Win Rate: {final_test_results['first_player_wins']/50:.2%}")
    print(f"Second Player Win Rate: {final_test_results['second_player_wins']/50:.2%}")

    
    # 保存模型和代理函數
    best_model.save('/kaggle/working/my_model.h5', include_optimizer=False)
    # 創建提交用的代理函數
    def Agent(obs, config):
        return my_agent(obs, config, best_model)
    
    # 創建提交用的代理函數
    write_agent_to_file(Agent, "submission.py")

