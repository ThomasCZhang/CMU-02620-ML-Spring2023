import numpy as np
import copy

transition_matrix = {
    "P": {"P": 0.3, "E": 0.1, "R": 0.3, "B": 0.3},
    "E": {"P": 0.1, "E": 0.4, "R": 0.1, "B": 0.4},
    "R": {"P": 0.1, "E": 0.1, "R": 0.4, "B": 0.4},
    "B": {"P": 0.1, "E": 0.1, "R": 0.1, "B": 0.7},
}

emission_matrix = {"P": [0.9, 0.1], "E": [0.8, 0.2], "R": [0.9, 0.1], "B": [0.2, 0.8]}

p_state = {"P": 0.2, "E": 0.3, "R": 0.3, "B": 0.2}

# keys = set(['P', 'E', 'R', 'B'])
# for key1 in keys:
#     for key2 in keys:
#         transition_matrix[key1][key2] = np.log(transition_matrix[key1][key2])
#     emission_matrix[key1] = np.log(emission_matrix[key1])


def main():
    seq = [0, 1, 0, 0, 1, 1]
    state_path = 'PBRRRB'
    state_path2 = 'RRBBBB'
    state_path3 = 'RBBBBB'
    state_path4 = 'RBRRBB'
    alpha_matrix, beta_matrix = EStep(seq, transition_matrix, emission_matrix, p_state)
    PrintAlphaBeta(alpha_matrix, beta_matrix)
    prob = ComputeStatePathProbability(seq, state_path, transition_matrix, emission_matrix, p_state)
    print('Probability of PBRRRB | 010011: ', prob/np.sum(alpha_matrix[:, -1]))
    states = sorted(list(transition_matrix.keys()))
    prob_matrix = CalculateProbabiiltyMatrix(alpha_matrix, beta_matrix)
    PrintProbMatrix(prob_matrix, states)
    best_path = Viterbi(seq, [0, 1], states, transition_matrix, emission_matrix, p_state)
    print(f'Optimal state path: {best_path}')
    
    # print(prob)

def CalculateProbabiiltyMatrix(alpha_matrix, beta_matrix):
    prob_matrix = np.multiply(alpha_matrix, beta_matrix)
    prob_matrix = prob_matrix/np.sum(prob_matrix, axis = 0)
    return prob_matrix

def PrintProbMatrix(prob_matrix, states):
    print(f'\n\n{"": >8s}Probability Matrix for P(q | O)')
    print(f'{"": <8s}', end="")
    for i in range(prob_matrix.shape[1]):
        print(f'T{i+1}{"": <6s}', end="")
    print()
    for i in range(prob_matrix.shape[0]):
        print(f"{states[i]:>7s} ", end="")
        for j in range(prob_matrix.shape[1]):
            print(f"{prob_matrix[i,j]: <8.5f}", end="")
        print()
    print('\n\n')


def BaumWelch():
    pass

def PrintAlphaBeta(alpha_matrix, beta_matrix):
    
    print(f'\n\n{"": >8s}Alpha Matrix')
    states = sorted(list(transition_matrix.keys()))
    print(f'{"": <8s}', end="")
    for i in range(alpha_matrix.shape[1]):
        print(f'T{i+1}{"": <6s}', end="")
    print()

    for i in range(alpha_matrix.shape[0]):
        print(f"{states[i]:>7s} ", end="")
        for j in range(alpha_matrix.shape[1]):
            print(f"{alpha_matrix[i,j]: <8.5f}", end="")
        print()

    print(f'\n\n{"": >8s}Beta Matrix')
    states = sorted(list(transition_matrix.keys()))
    print(f'{"": <8s}', end="")
    for i in range(alpha_matrix.shape[1]):
        print(f'T{i+1}{"": <6s}', end="")
    print()

    for i in range(beta_matrix.shape[0]):
        print(f"{states[i]:>7s} ", end="")
        for j in range(beta_matrix.shape[1]):
            print(f"{beta_matrix[i,j]: <8.5f}", end="")
        print()
    print("\n\n")

    print(f'Probability of 010011: {np.sum(alpha_matrix[:, -1])}')
    pass

def EStep(
    seq: list[int],
    transition_matrix: dict[str, dict[str, int]],
    emission_matrix: dict[str, list[int]],
    p_state: dict[str, int],
):
    alpha_matrix = Forward(seq, transition_matrix, emission_matrix, p_state)
    beta_matrix = Backward(seq, transition_matrix, emission_matrix)
    return alpha_matrix, beta_matrix


def Forward(
    seq: list[int],
    transition_matrix: dict[str, dict[str, int]],
    emission_matrix: dict[str, list[int]],
    p_state: dict[str, int],
):
    """
    Calculates the forwards probability in Baum-Welch algorithm.
    Input:
        seq: The sequence of observations.
        transition_matrix: The current transition probabilities.
        emissione_matrix: The current emission probabilities.
        p_state: The probability of a state.
    Output:
        The forward matrix in Baum-Welch algorithm.
    """
    states = sorted(list(transition_matrix.keys()))
    alpha_matrix = np.zeros((len(states), len(seq)))

    for i, state in enumerate(states):
        alpha_matrix[i, 0] = emission_matrix[state][seq[0]] * p_state[state]

    for t in range(1, len(seq)):
        obs = seq[t]
        for i, state in enumerate(states):
            for j, prev_state in enumerate(states):
                alpha_matrix[i, t] += transition_matrix[prev_state][state] * alpha_matrix[j, t - 1]
            alpha_matrix[i, t] *= emission_matrix[state][obs]

    return alpha_matrix


def Backward(seq: list[int], transition_matrix: dict[str, dict[str, int]], emission_matrix: dict[str, list[int]]):
    """
    Calculates the backwards probability in Baum-Welch algorithm.
    Input:
        seq: The sequence of observations.
        transition_matrix: The current transition probabilities.
        emissione_matrix: The current emission probabilities.
    Output:
        The backward matrix in Baum-Welch algorithm.
    """
    states = sorted(list(transition_matrix.keys()))
    beta_matrix = np.zeros((len(states), len(seq)))

    for i, state in enumerate(states):
        beta_matrix[i, len(seq) - 1] = 1

    for t in range(len(seq) - 2, -1, -1):
        obs = seq[t+1]
        for i, state in enumerate(states):
            for j, next_state in enumerate(states):
                beta_matrix[i, t] += (
                    transition_matrix[state][next_state] * emission_matrix[next_state][obs] * beta_matrix[j, t + 1]
                )

    return beta_matrix


def ComputeStatePathProbability(
    seq: str,
    state_path: str,
    transition_matrix: dict[str, dict[str, int]],
    emission_matrix: dict[str, list[int]],
    p_state: dict[str, int],
):
    prob = p_state[state_path[0]]*emission_matrix[state_path[0]][seq[0]]
    
    for i in range(1, len(seq)):
        obs = seq[i]
        state = state_path[i]
        prob *= transition_matrix[state_path[i-1]][state]*emission_matrix[state][obs]
    
    return prob
    
def Viterbi(output:str, emission_vals: list[str], state_vals: list[str], transition_mm: dict[str, dict[str, float]],
            emission_mm: dict[str, list[float]], init_prob: dict[str, int]) -> str:
    """
    Viterbi: Finds the most likely sequence of states that create a specific outcome based on a state transition
    matrix and an emission matrix.

    Input:
        output: The given output.
        emission_vals: The possible emission values. 
        state_vals: The possible states.
        transition_mm: The markov model for state transitions. (log probabilities)
        emission_mm: The model for emissions based on state. (log probabilities)
        init_prob: the probability of the state of the first position.
    Output:
        The most likely sequence of states.
    """

    transition_mm = copy.deepcopy(transition_mm)
    emission_mm = copy.deepcopy(emission_mm)
    for key1 in transition_mm:
        for key2 in transition_mm[key1]:
            transition_mm[key1][key2] = np.log(transition_mm[key1][key2])
        emission_mm[key1] = np.log(emission_mm[key1])
    

    score_matrix = [[0 for y in output] for x in state_vals]
    for x in range(len(state_vals)):
        score_matrix[x][0] = emission_mm[state_vals[x]][output[0]]+np.log(init_prob[state_vals[x]]) # Assume equal starting probabilities

    prior_dict = {}
    for x in state_vals:
        prior_dict[x] = False
    path_matrix = [[prior_dict.copy() for y in output] for x in state_vals]

    for x in range(1, len(output)):
        for y in range(len(state_vals)):
            current_state = state_vals[y]
            scores = {}
            for z in range(len(state_vals)):
                prior_state = state_vals[z]
                scores[prior_state] = score_matrix[z][x-1]+\
                    transition_mm[prior_state][current_state]+\
                    emission_mm[current_state][output[x]]
            score_matrix[y][x] = max(scores.values())
            
            for key in scores:
                if score_matrix[y][x] == scores[key]:
                    path_matrix[y][x][key] = True
    
    final_state = 0
    final_state_score = score_matrix[final_state][-1]
    for x in range(1, len(state_vals)):
        if score_matrix[x][-1] > final_state_score:
            final_state_score = score_matrix[x][-1]
            final_state = x

    state_path = Backtrack(final_state, state_vals, path_matrix)
    return state_path

def Backtrack(final_state: int, state_vals: list[str], path_matrix: list[list[dict[str, bool]]]):
    """
    Backtrack: Backtrack through a viterbi graph.
    Input:
        final_state: The state to start backtracking from.
        state_vals: a list of the possible states.
        path_matrix: A matrix that holds the path used to solve a viterbi graph.
    Output:
        The path taken as a string.
    """
    state_path = state_vals[final_state]
    current_state = final_state
    i = len(path_matrix[0])-1
    while i > 0:
        for idx, key in enumerate(state_vals):
            if path_matrix[current_state][i][key] == True:
                state_path = "".join([key, state_path])
                current_state = idx
                break
        i -= 1
    return state_path


if __name__ == "__main__":
    main()
