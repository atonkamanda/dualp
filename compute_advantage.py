def true_return(rewards, gamma):
    T = len(rewards)
    G = 0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
    return G

def advantage(rewards, values, gamma):
    T = len(rewards)
    advantages = []
    G = 0
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        G = delta + gamma * G
        advantages.append(G)
    advantages.reverse()
    return advantages


