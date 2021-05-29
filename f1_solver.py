"""
Example call: python -m f1_solver './Data/ED-00010-00000048.soi' -v

In general, there are N profiles V_1,...,V_N, each with n voters v_1,...,v_n.
Each voter v is a linear order over the set of candidates C.
Here, one season is a profile and voters are rankings/races.
Each season has only one profile, thus the first dimension N can be omitted.
(N,n,m) -> (n,m)
"""

import csv
import numpy as np
import sys
from argparse import ArgumentParser
from pulp import getSolver, listSolvers, LpInteger, LpMinimize, LpProblem, LpStatus, LpStatusOptimal, lpSum, LpVariable, value


# Linear Problem

def generate_lp(C, V, nums, p, alpha, eps=1):
    """ Generates the LP from the proof of Theorem 1 which minimizes the
    Manhattan-distance (L1) to the scoring system given by 'alpha'.
    Returns an LpProblem-object and alpha' as LpVariables for extensions. """
    m, n = nums[0], nums[1]
    assert eps>0, "eps needs to be positive."
    assert C.shape==(m,)
    assert len(V)==n
    assert isinstance(alpha, np.ndarray) and alpha.shape==(m,)

    C_wo_p = np.setdiff1d(C, [p])

    # LP problem and referenced variables
    prob = LpProblem("Scoring_System_Existence", LpMinimize)
    alpha_p_vars = LpVariable.dicts("alpha_p", C, lowBound=0, cat=LpInteger)
    beta_vars = LpVariable.dicts("beta", C, lowBound=0, cat=LpInteger)

    # objective function
    prob += lpSum([beta_vars[k] for k in C]), "Manhattan_distance_to_alpha"

    for k in C: # model absolute values (non-linear function)
        prob += alpha_p_vars[k] - beta_vars[k] <= alpha[k-1], f"Left_abs{k}"
        prob += alpha_p_vars[k] + beta_vars[k] >= alpha[k-1], f"Right_abs{k}"

    # constraint (1)
    for q in C_wo_p:
        prob += lpSum([((T(V,p,k) - T(V,q,k)) * alpha_p_vars[k]) for k in C]) >= eps, f"Constraint1_{q}"
    # constraint (2)
    for k in np.arange(start=1, stop=m): # stop is exclusive
        prob += alpha_p_vars[k] - alpha_p_vars[k+1] >= 0, f"Constraint2_{k}"
    # constraint (3)
    prob += alpha_p_vars[m] == 0, f"Constraint3"

    return prob, alpha_p_vars

def add_restriction_1(prob, C, nums, alpha_p_vars, ties):
    """ Adds restriction (I) to LpProblem 'prob'.
    All positions which tie with the last at least once need to be zero. """
    m = nums[0]
    assert C.shape==(m,)

    num_tied = get_max_num_of_ties(ties)
    for t in C[m-num_tied:m-1]:
        prob += alpha_p_vars[t] == 0, f"Restriction1_{t}"

    return prob

def add_restriction_2(prob, C, nums, alpha_p_vars, alpha):
    """ Adds restriction (II) to LpProblem 'prob'.
    All positions which are zero in the old system 'alpha' need to be zero. """
    m = nums[0]
    assert C.shape==(m,)

    num_zero = (alpha == 0).sum()
    for z in C[m-num_zero:m-1]:
        prob += alpha_p_vars[z] == 0, f"Restriction2_{z}"

    return prob

def add_restriction_3(prob, C, nums, alpha_p_vars):
    """ Adds restriction (III) to LpProblem 'prob'.
    Point differences between top positions must be non-decreasing. """
    m = nums[0]
    assert C.shape==(m,)

    for j in C[:m-2]:
        prob += alpha_p_vars[j] - 2*alpha_p_vars[j+1] + alpha_p_vars[j+2] >= 0, f"Restriction3_{j}"

    return prob

def generate_alpha(scores, length):
    """ Generates a scoring vector with given length from a list of scores. """
    assert isinstance(scores, list), "scores has invalid type."
    if len(scores) > length:
        return np.array(scores)[:length]
    return np.pad(scores, (0, length-len(scores)), 'constant', constant_values=0)

def get_max_num_of_ties(tied_positions_per_race):
    """ Calculates the maximum number of tied positions in a list of lists. """
    assert isinstance(tied_positions_per_race, list)
    assert len(tied_positions_per_race) > 0
    num_ties = [len(race_ties) for race_ties in tied_positions_per_race]
    return np.max(num_ties)

def pos(v, c):
    """ Returns the one-indexed position of candidate 'c' in linear order 'v'.
    If 'c' is not in 'v', 'c' is assumed to be tied with the last position. """
    assert isinstance(v, np.ndarray)
    indices = np.where(v==c)[0] # list of indices where v contains c
    assert indices.size<=1, "Candidate c must occur at most once."
    if indices.size == 0:
        return v.size + 1
    return indices[0] + 1

def T(V, c, k):
    """ Returns the number of voters for which candidate 'c' is on position 'k'
    in profile 'V'. """
    assert isinstance(V, list) and isinstance(V[0], np.ndarray)
    return sum([pos(v, c)==k for v in V])


# Helpers

def evaluate_lp(prob, solver):
    """ Solves LpProblem 'prob' and returns if an optimal solution exists. """
    prob.solve(solver)
    return value(prob.objective), prob.status

def print_log(message, verbose):
    """ Prints 'message' if 'verbose' is True. """
    if verbose:
        print(message)

def print_result(dists, stati, verbose=False):
    """ Prints distances and stati, as well as all possible winners. """
    if verbose: # more detailed statistics
        print(f"(d,s): {[(d, LpStatus[s]) for d,s in zip(dists, stati)]}")
        print(f"min d={np.min(dists)} with c={np.argmin(dists)+1}")
        print(f"second to min d={np.partition(dists, 1)[1]}")

    # print results for optimal values
    for c, (d,s) in enumerate(zip(dists, stati), start=1):
        if s==LpStatusOptimal:
            print(f"Candidate {c:2d} wins with distance {d:2d}.")

def check_shapes(cs, vs, nums, verbose=False):
    """ Checks shapes for candidates 'cs', voters 'vs' and metadata 'nums'. """
    assert len(nums)==2, "Metadata is corrupt."
    assert cs.shape==(nums[0], 2), f"cs has invalid shape {cs.shape}."
    assert len(vs)==nums[1], f"vs has invalid length {len(vs)}."
    print_log(f"Data has correct shapes.", verbose)
    return True

def read_dataset(rel_path, verbose=False):
    """ Reads data from soi-file referred to by rel_path.
    Returns candidates and nums=[num_candidates, num_voters] as np.array, voters as list of np.array and ties as list of lists. """
    candidates, voters, ties = [], [], []
    nums = np.zeros(2, dtype=int)

    csv.register_dialect('skip_space', skipinitialspace=True)
    with open(rel_path, 'r') as f:
        reader = csv.reader(f , delimiter=',', dialect='skip_space')

        for line in reader:
            if len(line)==1:
                assert nums[0]==0, "Number of candidates already set"
                nums[0]=line[0]
            elif len(line)==3 and nums[1]==0:
                nums[1]=line[0]
            else:
                (candidates if len(line)==2 else voters).append(line)

    candidates = np.array(candidates)
    for v in voters: # append ties to voters
        tie = np.setdiff1d(candidates[:,0], v[1:]).tolist()
        ties.append([int(e) for e in tie]) # save all ties as int
    voters = [np.array(a[1:], dtype=int) for a in voters] # remove "1-index"

    print_log(f"Dataset read successfully. {nums[0]} candidates, {nums[1]} voters.", verbose)
    return candidates, voters, nums, ties


# Main

def parse_arguments(args):
    """ Creates an ArgumentParser with help messages. """
    info = "Framework for scoring system evaluations using PULP."
    parser = ArgumentParser(description=info)
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="activate output")
    parser.add_argument('-r1', '--restr1', action='store_true',
                        default=False, help="activate restriction (I)")
    parser.add_argument('-r2', '--restr2', action='store_true',
                        default=False, help="activate restriction (II)")
    parser.add_argument('-r3', '--restr3', action='store_true',
                        default=False, help="activate restriction (III)")
    parser.add_argument('-s', '--solver', default='COIN_CMD',
                        choices=listSolvers(onlyAvailable=True),
                        help="choose solver")
    parser.add_argument('dataset', help="specify relative path to toc-file")

    if len(args) < 1:  # show help, if no arguments are given
        parser.print_help(sys.stderr)
        sys.exit()
    return parser.parse_args(args)

def main(args):
    parsed_args = parse_arguments(args)
    print_log(f"Called with {parsed_args}", parsed_args.verbose)

    cs, vs, nums, ties = read_dataset(parsed_args.dataset, parsed_args.verbose)
    assert check_shapes(cs, vs, nums, parsed_args.verbose), "Invalid shapes."
    C = cs[:,0].astype(int)
    m = nums[0]

    alpha = generate_alpha([10,8,6,5,4,3,2,1], m) # original alpha 2003–2008
    #alpha = generate_alpha([10,6,4,3,2,1], m) # original alpha 1991–2002

    solver = getSolver(parsed_args.solver, msg=False) # get silent solver
    distances, stati = np.zeros_like(C), np.zeros_like(C)
    for p in C:
        prob, alpha_p_vars = generate_lp(C, vs, nums, p, alpha, 1)
        if parsed_args.restr1:
            prob = add_restriction_1(prob, C, nums, alpha_p_vars, ties)
        if parsed_args.restr2:
            prob = add_restriction_2(prob, C, nums, alpha_p_vars, alpha)
        if parsed_args.restr3:
            prob = add_restriction_3(prob, C, nums, alpha_p_vars)

        distances[p-1], stati[p-1] = evaluate_lp(prob, solver)
        #print([f"{v}={v.varValue}" for v in prob.variables()])
    print_result(distances, stati, parsed_args.verbose)

if __name__ == '__main__':
    main(sys.argv[1:])
