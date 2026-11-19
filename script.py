#### HOMINES MORALES

## The four moral types and their perceived utilities (Section II) are:
##   e  homo economicus   u_e(x, y) = π(x, y)
##   m  homo milliensis   u_m(x, y) = (π(x, y) + π(y, x)) / 2
##   k  homo kantiensis   u_k(x, y) = π(x, x)
##   s  homo smithiensis  u_s(x, y) = min(π(x, y), π(y, x))
##
## The code then proceeds as the paper does:
##   1. It lists the Nash equilibria of every θ × θ game (Section III) and
##      checks each table against the perceived utilities above.
##   2. It averages each type's material payoff over the equilibria of each
##      game, treating multiple equilibria as equally likely.
##   3. It derives each matchup's long-run outcome from the payoffs the types
##      earn against one another, in exact arithmetic, and confirms it by
##      following the distribution of types for 100,000 generations under the
##      replicator dynamics of eq. [8] (Section IV).
##   4. It checks every Prisoner's Dilemma payoff combination in steps of 1/8
##      with T = 10 (82,160 combinations) and every Hawk-Dove ratio V/C from
##      0.01 to 0.99 in steps of 0.01 (footnotes 6 and 8).

# Initialize libraries.
import sys
import time
from fractions import Fraction
from itertools import product

import numpy as np
from scipy.special import logsumexp

# Initialize the four moral types and their one-letter abbreviations.
type_names = ("homo_economicus", "homo_milliensis", "homo_kantiensis", "homo_smithiensis")
type_letter = dict(zip(type_names, "emks"))
type_name = {letter: name for name, letter in type_letter.items()}


#### PERCEIVED UTILITIES

## `compute_perceived_utility` returns type θ's perceived utility u_θ(x, y),
## given a table of material payoffs π(x, y) (Section II).

def compute_perceived_utility(theta, material_payoff):
    # Select the utility function that belongs to the given type.
    return {
        "e": lambda x, y: material_payoff[(x, y)],
        "m": lambda x, y: (material_payoff[(x, y)] + material_payoff[(y, x)]) / 2,
        "k": lambda x, y: material_payoff[(x, x)],
        "s": lambda x, y: min(material_payoff[(x, y)], material_payoff[(y, x)]),
    }[theta]


#### PRISONER'S DILEMMA

## `classify_case` returns which of the paper's five cases a set of payoffs
## (T, R, P, S) falls in.

def classify_case(T, R, P, S):
    # Reject payoffs that are not a Prisoner's Dilemma.
    if not (T > R > P > S >= 0):
        raise ValueError(f"Need T > R > P > S >= 0, got {(T, R, P, S)}")
    # Compare S + T against 2R and 2P.
    if 2 * R > S + T > 2 * P:
        return 1
    if 2 * R > 2 * P > S + T:
        return 2
    if S + T > 2 * R > 2 * P:
        return 3
    if 2 * R == S + T > 2 * P:
        return 4
    # The only remaining case is 2R > S + T = 2P.
    return 5


## `pd_equilibria` holds the Nash equilibrium tables of Section III. Each row
## is the player's own type, and each entry (one per opponent type, in the
## order e, m, k, s) lists the equilibria as (own move)(opponent's move), or
## as a mixed Nash equilibrium MNE1, MNE2 or MNE3.

pd_equilibria = {
    1: {"e": ["DD", "DC", "DC", "DD"],
        "m": ["CD", "CC", "CC", "CC"],
        "k": ["CD", "CC", "CC", "CC"],
        "s": ["DD", "CC", "CC", "CC DD MNE1"]},
    2: {"e": ["DD", "DD", "DC", "DD"],
        "m": ["DD", "CC DD MNE2", "CC", "CC DD MNE3"],
        "k": ["CD", "CC", "CC", "CC"],
        "s": ["DD", "CC DD MNE3", "CC", "CC DD MNE1"]},
    3: {"e": ["DD", "DC", "DC", "DD"],
        "m": ["CD", "CD DC MNE2", "DC", "MNE3"],
        "k": ["CD", "CD", "CC", "CC"],
        "s": ["DD", "MNE3", "CC", "CC DD MNE1"]},
    4: {"e": ["DD", "DC", "DC", "DD"],
        "m": ["CD", "CC CD DC", "CC DC", "CC MNE3"],
        "k": ["CD", "CC CD", "CC", "CC"],
        "s": ["DD", "CC MNE3", "CC", "CC DD MNE1"]},
    5: {"e": ["DD", "DC DD", "DC", "DD"],
        "m": ["CD DD", "CC DD", "CC", "CC DD MNE3"],
        "k": ["CD", "CC", "CC", "CC"],
        "s": ["DD", "CC DD MNE3", "CC", "CC DD MNE1"]},
}


## `compute_pd_mixed_equilibria` returns the mixed Nash equilibria of the
## Prisoner's Dilemma as (probability the row player plays C, probability the
## column player plays C), using the notation of footnotes 1-3: γ leaves a
## homo smithiensis indifferent, and δ leaves a homo milliensis indifferent.

def compute_pd_mixed_equilibria(T, R, P, S, case):
    # Compute γ (footnote 1).
    gamma = (P - S) / ((P - S) + (R - S))
    # Initialize MNE1, the s × s mixed equilibrium.
    mixed_equilibria = {("s", "s"): (gamma, gamma)}
    # Add MNE2 and MNE3, which exist only in Cases 2 to 5.
    if case >= 2:
        # Compute ε and δ (footnote 2).
        epsilon = (S + T) / 2
        delta = (P - epsilon) / ((P - epsilon) + (R - epsilon))
        # Initialize MNE2, the m × m mixed equilibrium.
        mixed_equilibria[("m", "m")] = (delta, delta)
        # Initialize MNE3, in which m plays γ and s plays δ (footnote 3).
        mixed_equilibria[("m", "s")] = (gamma, delta)
        mixed_equilibria[("s", "m")] = (delta, gamma)
    return mixed_equilibria


## `build_prisoners_dilemma` returns each type's average material payoff
## against each other type in the Prisoner's Dilemma with payoffs (T, R, P, S).

def build_prisoners_dilemma(T, R, P, S=0):
    # Initialize the material payoffs π(x, y).
    material_payoff = {("C", "C"): R, ("C", "D"): S, ("D", "C"): T, ("D", "D"): P}
    # Determine the case and build the payoffs from its equilibrium table.
    case = classify_case(T, R, P, S)
    return build_payoffs(material_payoff, "CD", pd_equilibria[case],
                         compute_pd_mixed_equilibria(T, R, P, S, case))


#### HAWK-DOVE GAME

## `hd_equilibria` holds the Nash equilibrium tables of the Hawk-Dove Game,
## laid out as `pd_equilibria` is.

hd_equilibria = {
    "e": ["HD DH MNE4", "HD DH MNE5", "HD", "HD"],
    "m": ["HD DH MNE5", "HD DH DD", "HD DD", "HD DD"],
    "k": ["DH", "DH DD", "DD", "DD"],
    "s": ["DH", "DH DD", "DD", "DD"],
}


## `compute_hd_mixed_equilibria` returns the mixed Nash equilibria of the
## Hawk-Dove Game as (probability the row player plays H, probability the
## column player plays H).

def compute_hd_mixed_equilibria(V, C):
    # Compute the probability of playing H in the e × e mixed equilibrium.
    hawk_probability = V / C
    # Initialize MNE4 (e × e) and MNE5 (e plays D, m mixes).
    return {("e", "e"): (hawk_probability, hawk_probability),
            ("e", "m"): (0, hawk_probability),
            ("m", "e"): (hawk_probability, 0)}


## `build_hawk_dove` returns each type's average material payoff against each
## other type in the Hawk-Dove Game with resource value V and fighting cost C.

def build_hawk_dove(V, C):
    # Reject payoffs that are not a Hawk-Dove Game.
    if not C > V > 0:
        raise ValueError(f"Need C > V > 0, got V={V}, C={C}")
    # Initialize the material payoffs π(x, y).
    material_payoff = {("H", "H"): (V - C) / 2, ("H", "D"): V,
                       ("D", "H"): 0, ("D", "D"): V / 2}
    # Build the payoffs from the equilibrium table.
    return build_payoffs(material_payoff, "HD", hd_equilibria,
                         compute_hd_mixed_equilibria(V, C))


#### PAYOFFS FROM EQUILIBRIA

## `compute_expected_value` returns the expected value of a function f(x, y)
## when the row player plays the first move with one probability and the
## column player plays it with another.

def compute_expected_value(function, row_probability, column_probability, moves):
    # Initialize each player's probability of each move.
    first_move, second_move = moves
    row_probabilities = {first_move: row_probability, second_move: 1 - row_probability}
    column_probabilities = {first_move: column_probability, second_move: 1 - column_probability}
    # Sum the function over every pair of moves, weighted by their probabilities.
    return sum(row_probabilities[x] * column_probabilities[y] * function(x, y)
               for x in moves for y in moves)


## `parse_equilibria` turns a table entry such as "CC DD MNE1" into a list of
## (row probability, column probability) pairs, one per equilibrium.

def parse_equilibria(entry, row, column, mixed_equilibria, moves):
    # Initialize the list of equilibria.
    equilibria = []
    # Read each equilibrium in the entry.
    for token in entry.split():
        # Look up mixed equilibria by the pair of types.
        if token.startswith("MNE"):
            equilibria.append(mixed_equilibria[(row, column)])
        # Convert pure equilibria into probabilities of 0 or 1.
        else:
            equilibria.append((int(token[0] == moves[0]), int(token[1] == moves[0])))
    return equilibria


## `find_pure_equilibria` returns every pure Nash equilibrium of the game
## between a row type and a column type, as strings such as "CD" (row plays C,
## column plays D). A pair of moves is an equilibrium if neither player can
## raise its perceived utility by switching moves alone.

def find_pure_equilibria(row_utility, column_utility, moves, tolerance=1e-12):
    # Initialize the list of equilibria.
    equilibria = []
    # Try every pair of moves.
    for row_move, column_move in product(moves, moves):
        # Check that the row player has no better move.
        row_is_best = all(row_utility(row_move, column_move)
                          >= row_utility(other_move, column_move) - tolerance
                          for other_move in moves)
        # Check that the column player has no better move.
        column_is_best = all(column_utility(column_move, row_move)
                             >= column_utility(other_move, row_move) - tolerance
                             for other_move in moves)
        # Keep the pair if both are best responses.
        if row_is_best and column_is_best:
            equilibria.append(row_move + column_move)
    return sorted(equilibria)


## `is_mixed_equilibrium` tests whether a pair of mixed strategies is a Nash
## equilibrium: each player's mix must earn at least as much as either pure
## move against the other player's mix.

def is_mixed_equilibrium(row_utility, column_utility, row_probability, column_probability,
                         moves, tolerance=1e-12):
    # Check that both probabilities are valid.
    if not (0 <= row_probability <= 1 and 0 <= column_probability <= 1):
        return False
    # Compute what each player's mix earns.
    row_value = compute_expected_value(row_utility, row_probability, column_probability, moves)
    column_value = compute_expected_value(column_utility, column_probability, row_probability,
                                          moves)
    # Compute what each player's best pure move would earn instead.
    row_best = max(compute_expected_value(row_utility, int(move == moves[0]),
                                          column_probability, moves) for move in moves)
    column_best = max(compute_expected_value(column_utility, int(move == moves[0]),
                                             row_probability, moves) for move in moves)
    # Accept the mixes if neither player can do better.
    return row_value >= row_best - tolerance and column_value >= column_best - tolerance


## `check_equilibria` stops the program if any equilibrium table disagrees with
## the perceived utilities. For every pair of types it checks: 
## (1) the table is symmetric (the row × column entry mirrors the column × row entry),
## (2) the listed pure equilibria are exactly the real ones, and every listed mixed
##     equilibrium is a real one.

def check_equilibria(material_payoff, moves, equilibrium_table, mixed_equilibria):
    # Initialize the list of problems found.
    problems = []
    # Check every pair of types.
    for row_index, row in enumerate("emks"):
        for column_index, column in enumerate("emks"):
            # Get both players' perceived utilities and the table entry.
            row_utility = compute_perceived_utility(row, material_payoff)
            column_utility = compute_perceived_utility(column, material_payoff)
            entry = equilibrium_table[row][column_index]

            # Check that the row × column entry mirrors the column × row entry.
            this_side = sorted(parse_equilibria(entry, row, column, mixed_equilibria, moves))
            other_side = sorted((q, p) for p, q in parse_equilibria(
                equilibrium_table[column][row_index], column, row, mixed_equilibria, moves))
            if this_side != other_side:
                problems.append(f"{row} x {column} is not the mirror of {column} x {row}")

            # Check that the listed pure equilibria are exactly the real ones.
            actual_pure = find_pure_equilibria(row_utility, column_utility, moves)
            listed_pure = sorted(token for token in entry.split() if not token.startswith("MNE"))
            if actual_pure != listed_pure:
                problems.append(f"{row} x {column}: listed pure {listed_pure}, "
                                f"actual {actual_pure}")

            # Check that the listed mixed equilibrium is a real one.
            if "MNE" in entry:
                p, q = mixed_equilibria[(row, column)]
                if not is_mixed_equilibrium(row_utility, column_utility, p, q, moves):
                    problems.append(f"{row} x {column}: mixed profile {(p, q)} "
                                    f"is not an equilibrium")
    # Report any problems found.
    if problems:
        raise AssertionError("Equilibrium table errors:\n  " + "\n  ".join(problems))


## `build_payoffs` returns each type's material payoff against each other type,
## averaged over the equilibria of their game (each treated as equally likely).

def build_payoffs(material_payoff, moves, equilibrium_table, mixed_equilibria):
    # Check the equilibrium table before using it.
    check_equilibria(material_payoff, moves, equilibrium_table, mixed_equilibria)
    # Initialize the payoff table.
    payoffs = {}
    # Average the row player's material payoff over each game's equilibria.
    for row in "emks":
        for column_index, column in enumerate("emks"):
            equilibria = parse_equilibria(equilibrium_table[row][column_index], row, column,
                                          mixed_equilibria, moves)
            total = sum(compute_expected_value(lambda x, y: material_payoff[(x, y)], p, q, moves)
                        for p, q in equilibria)
            payoffs[(type_name[row], type_name[column])] = total / len(equilibria)
    return payoffs


#### REPLICATOR DYNAMICS

## `build_payoff_matrix` returns the payoffs as a 4 × 4 array, with entry
## [i, j] the payoff of type i against type j (in the order e, m, k, s).

def build_payoff_matrix(payoffs):
    # Arrange the payoffs by row and column type.
    return np.array([[float(payoffs[(row, column)]) for column in type_names]
                     for row in type_names])


## `step_shares` advances one game by one generation of eq. [8], using plain
## population shares. It is the clearest statement of the dynamics: each type's
## fitness is its average payoff against the current population, and each
## type's share grows in proportion to its fitness relative to the average.
## (`evolve` below does the same thing in a form that never rounds a tiny share
## to zero, and is what the program actually uses.)

def step_shares(payoff_matrix, shares):
    # Compute each type's fitness, W(θ) = Σ_τ p_τ A(θ, τ).
    fitness = payoff_matrix @ shares
    # Compute the population's average fitness, Σ_θ p_θ W(θ).
    average_fitness = shares @ fitness
    # Scale each share by its fitness relative to the average, p'_θ = p_θ W(θ) / W̄.
    return shares * fitness / average_fitness


## `evolve` follows eq. [8] for a number of generations, exactly as
## `step_shares` does, but in two ways that differ for practical reasons:
##   1. It stores the logarithm of each share rather than the share itself.
##      In a rock-paper-scissors cycle some shares fall to 10^-200 and below,
##      which ordinary numbers would round to zero; their logarithms (about
##      −460) are stored without trouble. Multiplying shares becomes adding
##      logarithms, and adding shares becomes `logsumexp`, the logarithm of a
##      sum of exponentials.
##   2. It can run many games side by side, which is much faster when checking
##      many parameter sets. Arrays therefore have a leading "game" dimension:
##      payoff matrices are (games, 4, 4) and shares are (games, 4).
## It returns the final log-shares, or their history every `record_every`
## generations, with shape (records, games, 4).

def evolve(payoff_matrix, counts, generations, record_every=None):
    # Note whether a single game was given, so the result can be returned in that shape.
    single_game = np.ndim(counts) == 1
    # Arrange the inputs as a batch of games: counts (games, 4), payoffs (games, 4, 4).
    counts = np.asarray(counts, float).reshape(-1, 4)
    payoff_matrix = np.asarray(payoff_matrix, float).reshape(-1, 4, 4)
    payoff_matrix = np.broadcast_to(payoff_matrix, (len(counts), 4, 4))
    # Take logarithms of the payoffs and of the starting shares (an absent type becomes −∞).
    with np.errstate(divide="ignore"):
        log_payoffs = np.log(payoff_matrix)
        log_shares = np.log(counts / counts.sum(axis=1, keepdims=True))
    present = np.isfinite(log_shares)
    # Initialize the history.
    history = []
    # Advance one generation at a time.
    for generation in range(generations):
        # Compute each type's log-fitness, log W(θ) = log Σ_τ p_τ A(θ, τ).
        log_fitness = logsumexp(log_payoffs + log_shares[:, None, :], axis=2)
        # Multiply each share by its fitness (add the logarithms), keeping absent types absent.
        with np.errstate(invalid="ignore", over="ignore"):
            new_log_shares = np.where(present, log_shares + log_fitness, -np.inf)
        # Divide by the total so the shares sum to one (subtract its logarithm).
        log_shares = new_log_shares - logsumexp(new_log_shares, axis=1, keepdims=True)
        # Record the shares if requested.
        if record_every and generation % record_every == 0:
            history.append(log_shares.copy())
    # Return the history or the final shares, in the shape that was given.
    result = np.array(history) if record_every else log_shares
    if single_game:
        return result[:, 0, :] if record_every else result[0]
    return result


#### LONG-RUN OUTCOMES

## With two types i and j, type i's fitness advantage when its share is x is
## x α + (1 − x) β, where α = A(i, i) − A(j, i) and β = A(i, j) − A(j, j).
## The signs of α and β therefore decide the matchup:
##   α, β ≥ 0 (α ≠ β ≠ 0):   i wins (dominance).
##   α, β ≤ 0 (α ≠ β ≠ 0):   j wins (dominance).
##   α = β = 0:              both keep their starting shares (a tie).
##   α > 0 > β:              whoever starts above x* = −β / (α − β) wins (bistable).
##   α < 0 < β:              both coexist at x* (coexistence).
## With more types, a type dies out if another type beats it and also beats
## every type it beats. What remains is a single type, tied types, or three
## types that beat each other in a loop (a cycle).

## `get_sign` returns 1, 0 or −1 for a positive, zero or negative number.

def get_sign(value):
    # Compare the value with zero.
    return (value > 0) - (value < 0)


## `predict_pair` returns the long-run survivors of a two-type matchup and the
## kind of matchup, given the first type's starting share.

def predict_pair(payoffs, first, second, first_share=Fraction(1, 2)):
    # Look up the full type names.
    first_name, second_name = type_name[first], type_name[second]
    # Compute α and β.
    alpha = payoffs[(first_name, first_name)] - payoffs[(second_name, first_name)]
    beta = payoffs[(first_name, second_name)] - payoffs[(second_name, second_name)]
    alpha_sign, beta_sign = get_sign(alpha), get_sign(beta)
    # Classify the matchup by the signs of α and β.
    if alpha_sign == 0 and beta_sign == 0:
        return (first, second), "tie"
    if alpha_sign >= 0 and beta_sign >= 0:
        return (first,), "dominance"
    if alpha_sign <= 0 and beta_sign <= 0:
        return (second,), "dominance"
    # Locate the interior rest point x*.
    rest_point = -beta / (alpha - beta)
    # In a bistable matchup, compare the starting share with x*.
    if alpha_sign > 0:
        side = get_sign(first_share - rest_point)
        if side > 0:
            return (first,), "bistable"
        if side < 0:
            return (second,), "bistable"
        return (first, second), "bistable"
    # Otherwise the two types coexist.
    return (first, second), "coexistence"


## A worked example. In the Prisoner's Dilemma with (T, R, P, S) = (10, 8, 6, 0),
## sub-case 2(a), homo economicus and homo milliensis both defect against each
## other, so A(e, e) = A(e, m) = A(m, e) = P = 6. Against itself, homo
## milliensis plays (C, C), (D, D) or the mixed equilibrium MNE2 (each player
## cooperates with probability δ = 1/4, earning 5.75 on average), each with
## probability 1/3, so A(m, m) = (8 + 6 + 5.75) / 3 ≈ 6.58. With i = e and
## j = m, α = 6 − 6 = 0 and β = 6 − 6.58 = −0.58: both are ≤ 0, so homo
## milliensis wins e × m.

## `find_pair_winners` predicts every pair of types within a matchup and
## returns the survivors of each pair.

def find_pair_winners(payoffs, types):
    # Predict each pair once, in the order e, m, k, s.
    return {(x, y): predict_pair(payoffs, x, y)[0]
            for x in types for y in types if "emks".index(x) < "emks".index(y)}


## `beats` tests whether type x wins its pair against type y.

def beats(pair_winners, x, y):
    # Look up the pair in either order.
    survivors = pair_winners[(x, y)] if (x, y) in pair_winners else pair_winners[(y, x)]
    return survivors == (x,)


## `ties` tests whether types x and y both survive their pair.

def ties(pair_winners, x, y):
    # Look up the pair in either order.
    survivors = pair_winners[(x, y)] if (x, y) in pair_winners else pair_winners[(y, x)]
    return len(survivors) == 2


## `remove_beaten_types` repeatedly removes any type x for which some other
## type y beats x and also beats (or ties) every type that x beats. Such a type
## y does at least as well as x against everyone, so x dies out.

def remove_beaten_types(pair_winners, types):
    # Initialize the types still in the running.
    remaining = list(types)
    # Keep removing types until none can be removed.
    while True:
        beaten_type = None
        for x in remaining:
            for y in remaining:
                # Find the types that x beats.
                beaten_by_x = [z for z in remaining if z not in (x, y)
                               and beats(pair_winners, x, z)]
                # Check whether y beats x and everything x beats.
                if (x != y and beats(pair_winners, y, x)
                        and all(beats(pair_winners, y, z) or ties(pair_winners, y, z)
                                for z in beaten_by_x)):
                    beaten_type = x
                    break
            if beaten_type:
                break
        # Stop when no type can be removed.
        if beaten_type is None:
            return remaining
        remaining.remove(beaten_type)


## `group_tied_types` groups types that tie with each other, since tied types
## behave as a single type.

def group_tied_types(pair_winners, types):
    # Initialize the list of groups.
    groups = []
    # Place each type in the first group it ties with, or in a new group.
    for x in types:
        for group in groups:
            if all(ties(pair_winners, x, y) for y in group):
                group.append(x)
                break
        else:
            groups.append([x])
    return groups


## `find_cycle` returns three groups arranged as a loop x → y → z (y beats x,
## z beats y and x beats z), or None if they do not form one.

def find_cycle(pair_winners, groups):
    # Represent each group by its first type.
    first, second, third = (group[0] for group in groups)
    # Try both directions around the loop.
    if (beats(pair_winners, second, first) and beats(pair_winners, third, second)
            and beats(pair_winners, first, third)):
        return [groups[0], groups[1], groups[2]]
    if (beats(pair_winners, third, first) and beats(pair_winners, second, third)
            and beats(pair_winners, first, second)):
        return [groups[0], groups[2], groups[1]]
    return None


## `predict_matchup` returns the long-run outcome of a matchup in which every
## type starts with the same population, as (label, survivors, loop), where
## loop is the order of a cycle or None. It predicts every pair, removes beaten
## types, groups tied types, and then reports a single survivor (or tied group)
## or a cycle.

def predict_matchup(payoffs, types):
    # Put the types in the order e, m, k, s.
    types = [letter for letter in "emks" if letter in types]
    # Handle a single type.
    if len(types) == 1:
        return types[0], tuple(types), None

    # Predict every pair within the matchup.
    pair_winners = find_pair_winners(payoffs, types)

    # Return the pair result directly for two types.
    if len(types) == 2:
        survivors = pair_winners[(types[0], types[1])]
        return ", ".join(survivors), survivors, None

    # Stop if two types coexist inside a larger matchup (this never occurs here).
    if any(predict_pair(payoffs, x, y)[1] == "coexistence" for x, y in pair_winners):
        raise NotImplementedError("coexisting pair inside a matchup of three or more types")

    # Remove beaten types, then group the tied types that remain.
    remaining = remove_beaten_types(pair_winners, types)
    groups = group_tied_types(pair_winners, remaining)

    # Return a single surviving group.
    if len(groups) == 1:
        return ", ".join(groups[0]), tuple(groups[0]), None

    # Return a cycle of three groups.
    loop = find_cycle(pair_winners, groups) if len(groups) == 3 else None
    if loop:
        name = " → ".join("(" + ", ".join(group) + ")" if len(group) > 1 else group[0]
                          for group in loop)
        members = tuple(x for group in loop for x in group)
        return f"cycle {name}", members, loop

    # Stop on any other structure (this never occurs here).
    raise NotImplementedError(f"unexpected structure {groups}")


## `find_ess` returns the evolutionarily stable strategies of a matchup in the
## sense of Maynard Smith (1982): θ resists τ if A(θ, θ) > A(τ, θ), or if the
## two are equal and A(θ, τ) > A(τ, τ), and θ is an ESS if it resists every
## other type present.

def find_ess(payoffs, types):
    # Initialize a helper that reads payoffs by letter.
    def get_payoff(x, y):
        return payoffs[(type_name[x], type_name[y])]

    # Put the types in the order e, m, k, s.
    types = [letter for letter in "emks" if letter in types]
    # Keep each type that resists every other type.
    return tuple(theta for theta in types if all(
        get_payoff(theta, theta) > get_payoff(tau, theta)
        or (get_payoff(theta, theta) == get_payoff(tau, theta)
            and get_payoff(theta, tau) > get_payoff(tau, tau))
        for tau in types if tau != theta))


#### SUB-CASES OF THE PRISONER'S DILEMMA

## Subtracting S from every payoff changes no equilibrium and no comparison, so
## the thresholds of footnote 7 are written with T' = T − S, R' = R − S and
## P' = P − S.

## `does_milliensis_beat_economicus` tests the Case 2 threshold for e × m:
## 2R' − T' > √2 (2P' − T'), where both sides are positive.

def does_milliensis_beat_economicus(T, R, P, S):
    # Shift the payoffs by S.
    T, R, P = T - S, R - S, P - S
    # Compare the squares, which avoids computing √2.
    return (2 * R - T) ** 2 > 2 * (2 * P - T) ** 2


## `does_smithiensis_beat_economicus` tests the Case 2 threshold for e × s:
## P'R'T' > (P' + R')(2P'² − R'²).

def does_smithiensis_beat_economicus(T, R, P, S):
    # Shift the payoffs by S.
    T, R, P = T - S, R - S, P - S
    # Compare the two sides.
    return P * R * T > (P + R) * (2 * P * P - R * R)


## `does_smithiensis_beat_kantiensis` tests the Case 3 threshold for k × s:
## P'R'T' > (P' + R')(2R'² − P'²).

def does_smithiensis_beat_kantiensis(T, R, P, S):
    # Shift the payoffs by S.
    T, R, P = T - S, R - S, P - S
    # Compare the two sides.
    return P * R * T > (P + R) * (2 * R * R - P * P)


## `is_on_boundary` tests whether a set of payoffs lies exactly on one of the
## thresholds, where the two types tie.

def is_on_boundary(T, R, P, S):
    # Shift the payoffs by S.
    shifted_T, shifted_R, shifted_P = T - S, R - S, P - S
    case = classify_case(T, R, P, S)
    # Check the two Case 2 thresholds.
    if case == 2:
        return ((2 * shifted_R - shifted_T) ** 2 == 2 * (2 * shifted_P - shifted_T) ** 2
                or shifted_P * shifted_R * shifted_T
                == (shifted_P + shifted_R) * (2 * shifted_P ** 2 - shifted_R ** 2))
    # Check the Case 3 threshold.
    if case == 3:
        return (shifted_P * shifted_R * shifted_T
                == (shifted_P + shifted_R) * (2 * shifted_R ** 2 - shifted_P ** 2))
    return False


## `classify_subcase` returns the sub-case ("1", "2a", ..., "5") a set of
## payoffs falls in.

def classify_subcase(T, R, P, S=0):
    # Determine the case.
    case = classify_case(T, R, P, S)
    # Mark payoffs that lie exactly on a threshold.
    if is_on_boundary(T, R, P, S):
        return f"{case} (boundary)"
    # Split Case 2 by its two thresholds.
    if case == 2:
        if does_milliensis_beat_economicus(T, R, P, S):
            return "2a"
        if does_smithiensis_beat_economicus(T, R, P, S):
            return "2b"
        return "2c"
    # Split Case 3 by its threshold.
    if case == 3:
        return "3b" if does_smithiensis_beat_kantiensis(T, R, P, S) else "3a"
    return str(case)


# Initialize the conditions that define each sub-case.
subcase_conditions = {
    "1":  "2R > S + T > 2P",
    "2a": "2R > 2P > S + T,  2R' - T' > √2(2P' - T')",
    "2b": "2R > 2P > S + T,  2R' - T' < √2(2P' - T'),  P'R'T' > (P' + R')(2P'² - R'²)",
    "2c": "2R > 2P > S + T,  P'R'T' < (P' + R')(2P'² - R'²)",
    "3a": "S + T > 2R > 2P,  P'R'T' < (P' + R')(2R'² - P'²)",
    "3b": "S + T > 2R > 2P,  P'R'T' > (P' + R')(2R'² - P'²)",
    "4":  "2R = S + T > 2P",
    "5":  "2R > S + T = 2P",
}

# Initialize a plain description of each split sub-case.
subcase_meanings = {
    "2a": "m and s both beat e",
    "2b": "s beats e, m does not",
    "2c": "neither m nor s beats e",
    "3a": "k beats s",
    "3b": "s beats k",
}

# Initialize the note that defines the shifted payoffs.
shifted_payoff_note = "T' = T − S,  R' = R − S,  P' = P − S"

# Initialize one set of payoffs inside each sub-case.
subcase_examples = {
    "1": (10, 6, 4, 0), "2a": (10, 8, 6, 0), "2b": (10, 8.1, 7.2, 0), "2c": (10, 9, 8, 0),
    "3a": (10, 4, 2, 0), "3b": (10, 1, 0.5, 0), "4": (10, 5, 3, 0), "5": (10, 7, 5, 0),
}


#### RESULTS TABLES

# Initialize the eleven matchups of Section IV.
matchups = ["e x m", "e x k", "e x s", "m x k", "m x s", "k x s",
            "e x m x k", "e x m x s", "e x k x s", "m x k x s", "e x m x k x s"]

# Initialize the Hawk-Dove payoffs (C, V) used in the paper.
hawk_dove_samples = [(2, 1), (5, 1), (100, 2)]

# Initialize the note that explains cycles.
cycle_note = ("x → y → z: the population cycles, z overtaking y, y overtaking x, and x "
              "overtaking z (rock-paper-scissors). In a finite population one type would "
              "eventually go extinct; so the result does not depend on population size, "
              "the cycle is reported instead.")


## `convert_to_fractions` converts a set of payoffs to exact fractions.

def convert_to_fractions(parameters):
    # Convert each number to the nearest simple fraction.
    return tuple(Fraction(x).limit_denominator(10 ** 6) for x in parameters)


## `build_pd_payoffs` builds the Prisoner's Dilemma payoffs from (T, R, P, S),
## in exact arithmetic.

def build_pd_payoffs(parameters):
    # Convert the payoffs to fractions and build the game.
    return build_prisoners_dilemma(*convert_to_fractions(parameters))


## `build_hd_payoffs` builds the Hawk-Dove payoffs from (C, V), in exact
## arithmetic.

def build_hd_payoffs(cost_and_value):
    # Convert the payoffs to fractions and build the game.
    C, V = convert_to_fractions(cost_and_value)
    return build_hawk_dove(V=V, C=C)


## `tabulate_matchups` returns the long-run outcome of every matchup in a game.

def tabulate_matchups(payoffs):
    # Predict each matchup from equal starting populations.
    return {matchup: predict_matchup(payoffs, matchup.split(" x "))[0] for matchup in matchups}


## `compare_skewed_pairs` prints the outcome of every ordered pair of types when
## the first type starts with one population and the second with another.

def compare_skewed_pairs(payoffs, first_count, second_count):
    # Loop through every ordered pair of distinct types.
    for first, second in product("emks", repeat=2):
        if first != second:
            # Predict the pair from the given starting share.
            survivors, kind = predict_pair(payoffs, first, second,
                                           Fraction(first_count, first_count + second_count))
            print(f"{first} x {second} ({first_count} vs {second_count}):  "
                  f"{', '.join(survivors)}   [{kind}]")


## `get_starting_counts` returns the starting population of each type in a
## matchup (1000 for each type present, 0 otherwise).

def get_starting_counts(matchup, start=1000):
    # Find the types present.
    present = matchup.split(" x ")
    # Assign each present type the starting population.
    return [start if letter in present else 0 for letter in "emks"]


## `format_matchup` writes a matchup with multiplication signs, as in the paper.

def format_matchup(matchup):
    # Replace each "x" with "×".
    return matchup.replace(" x ", " × ")


## `format_subcase` writes a sub-case as in the paper, e.g. "2(a)".

def format_subcase(subcase):
    # Put the letter in parentheses.
    return f"{subcase[0]}({subcase[1]})" if len(subcase) == 2 else subcase


## `print_table` prints a table with a title, a header row, body rows and notes,
## either as plain text or as Markdown.

def print_table(title, header, rows, markdown=False, notes=()):
    # Print the table as Markdown.
    if markdown:
        print(f"**{title}**\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join([":---"] + [":---:"] * (len(header) - 1)) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    # Print the table as plain text.
    else:
        # Size each column to its widest cell.
        widths = [max(len(row[index]) for row in [header] + rows)
                  for index in range(len(header))]
        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

        # Initialize a helper that left-aligns the first column and centers the rest.
        def format_row(cells):
            first_cell = f" {cells[0]:<{widths[0]}} "
            other_cells = [f" {cell:^{width}} " for cell, width in zip(cells[1:], widths[1:])]
            return "|" + "|".join([first_cell] + other_cells) + "|"

        # Print the title, header and rows.
        print(title)
        print(border)
        print(format_row(header))
        print(border.replace("-", "="))
        for row in rows:
            print(format_row(row))
        print(border)
    # Print the notes beneath the table.
    for note in notes:
        print(note)
    print()


## `print_subcase_tables` prints Table 1: the long-run outcome of every matchup
## in every sub-case of the Prisoner's Dilemma, one table per case.

def print_subcase_tables(markdown=False):
    # Loop through the five cases.
    for case in "12345":
        # Find the sub-cases of this case.
        subcases = [subcase for subcase in subcase_conditions if subcase[0] == case]
        # Tabulate each sub-case from its example payoffs.
        header = ["θ × θ"] + [format_subcase(subcase) for subcase in subcases]
        tables = {subcase: tabulate_matchups(build_pd_payoffs(subcase_examples[subcase]))
                  for subcase in subcases}
        rows = [[format_matchup(matchup)] + [tables[subcase][matchup] for subcase in subcases]
                for matchup in matchups]
        # Write the notes: the conditions, then the ESS rule.
        notes = [f"  {format_subcase(subcase)}: {subcase_conditions[subcase]}"
                 + (f"   ({subcase_meanings[subcase]})" if subcase in subcase_meanings else "")
                 for subcase in subcases]
        if any("'" in subcase_conditions[subcase] for subcase in subcases):
            notes.append(f"  where {shifted_payoff_note}")
        notes.append("  ESS: wherever a single type survives, it is also the ESS of that "
                     "matchup; ties and cycles have no ESS.")
        # Add the Case 5 bistability note.
        if case == "5":
            notes.append("  In part of Case 5, m × s is bistable: when R − S > √3(P − S), m wins "
                         "if it starts with a large enough head start; from equal numbers s "
                         "always wins. There, both m and s are ESSs of m × s.")
        print_table(f"Prisoner's Dilemma, Case {case}", header, rows, markdown, notes)
    # Explain the cycles.
    print(cycle_note + "\n")


## `print_hawk_dove_table` prints Table 2: the long-run outcome of every matchup
## in the Hawk-Dove Game, which is the same for every C > V > 0.

def print_hawk_dove_table(markdown=False):
    # Tabulate the matchups for each of the paper's Hawk-Dove payoffs.
    results = {cost_and_value: tabulate_matchups(build_hd_payoffs(cost_and_value))
               for cost_and_value in hawk_dove_samples}
    # Confirm the payoffs do not matter.
    first_result = results[hawk_dove_samples[0]]
    assert all(result == first_result for result in results.values())
    # Print the single column.
    rows = [[format_matchup(matchup), first_result[matchup]] for matchup in matchups]
    print_table("Hawk-Dove Game", ["θ × θ", "All C > V > 0"], rows, markdown,
                ["  All single outcomes are ESS; k × s (a tie) has none."])
    return results


#### CHECKS

## `build_pd_grid` returns every Prisoner's Dilemma (T, R, P, S) with
## T > R > P > S ≥ 0 on a grid of the given step, with T fixed (multiplying all
## payoffs by a constant does not change the results).

def build_pd_grid(step=Fraction(1, 8), T=10):
    # Initialize the values each payoff can take.
    step = Fraction(step)
    values = [index * step for index in range(int(T / step))]
    # Keep every combination in the right order.
    return [(Fraction(T), R, P, S) for S in values for P in values for R in values
            if T > R > P > S]


## `build_hd_grid` returns every Hawk-Dove (C, V) with C = 1 and V/C from the
## step up to 1 (only the ratio matters).

def build_hd_grid(step=Fraction(1, 100)):
    # Step V/C from the step size up to just below 1.
    step = Fraction(step)
    return [(Fraction(1), index * step) for index in range(1, int(1 / step))]


## `check_subcases` checks, in exact arithmetic, that every set of payoffs on
## the grid gives the same table as the rest of its sub-case, that the
## thresholds agree with the pairwise results, and that the ESS rule stated in
## the paper holds. It returns the number of grid points in each sub-case.

def check_subcases(step=Fraction(1, 8)):
    # Initialize the table found for each sub-case.
    tables = {}
    # Loop through every Prisoner's Dilemma on the grid.
    for parameters in build_pd_grid(step):
        subcase = classify_subcase(*parameters)
        payoffs = build_prisoners_dilemma(*parameters)

        # Check the thresholds against the pairwise results.
        if subcase[0] == "2":
            assert ((predict_pair(payoffs, "e", "m")[0] == ("m",))
                    == does_milliensis_beat_economicus(*parameters)), parameters
            assert ((predict_pair(payoffs, "e", "s")[0] == ("s",))
                    == does_smithiensis_beat_economicus(*parameters)), parameters
        if subcase[0] == "3":
            assert ((predict_pair(payoffs, "k", "s")[0] == ("s",))
                    == does_smithiensis_beat_kantiensis(*parameters)), parameters

        # Tabulate every matchup.
        table = tabulate_matchups(payoffs)

        # Check the ESS rule: single survivors are the ESS, except that both m and s
        # are ESSs of m × s in the bistable part of Case 5.
        is_bistable = subcase == "5" and (parameters[1] - parameters[3]) ** 2 \
            > 3 * (parameters[2] - parameters[3]) ** 2
        for matchup in matchups:
            types = matchup.split(" x ")
            _, survivors, loop = predict_matchup(payoffs, types)
            if is_bistable and matchup == "m x s":
                expected = ("m", "s")
            elif len(survivors) == 1 and not loop:
                expected = survivors
            else:
                expected = ()
            assert find_ess(payoffs, types) == expected, (parameters, matchup)

        # Check that the sub-case gives one table.
        if subcase in tables and tables[subcase] != table:
            raise AssertionError(f"sub-case {subcase} not uniform at {parameters}")
        tables.setdefault(subcase, table)
        if subcase in subcase_examples:
            assert table == tabulate_matchups(build_pd_payoffs(subcase_examples[subcase])), \
                (subcase, parameters)

    # Check that every Hawk-Dove ratio gives the same table.
    hawk_dove_tables = {repr(tabulate_matchups(build_hawk_dove(V=V, C=C)))
                        for C, V in build_hd_grid()}
    assert len(hawk_dove_tables) == 1, "Hawk-Dove not uniform"

    # Count the grid points in each sub-case.
    return {subcase: sum(classify_subcase(*parameters) == subcase
                         for parameters in build_pd_grid(step))
            for subcase in tables}


## `verify_by_simulation` follows eq. [8] for 100,000 generations and confirms
## every predicted outcome. For a winner or tied group, each survivor's share
## must not be falling at the end, and every other type must be below e^−50 or
## still falling. For a cycle, every type outside the loop must fall below
## e^−50, and the loop must not settle into coexistence (some member's share
## falls below e^−50 as the cycle swings). It returns the list of failures.

def verify_by_simulation(parameter_list, game="pd", generations=100_000):
    # Build every game's payoff matrix.
    build_game = build_pd_payoffs if game == "pd" else build_hd_payoffs
    payoff_matrices = np.array([build_payoff_matrix(build_game(parameters))
                                for parameters in parameter_list])
    # Initialize the list of failures.
    failures = []
    # Loop through every matchup, simulating all games at once.
    for matchup in matchups:
        starting_counts = np.tile(np.array(get_starting_counts(matchup), float),
                                  (len(parameter_list), 1))
        histories = evolve(payoff_matrices, starting_counts, generations, record_every=50)
        # Compare each game's simulation with its prediction.
        for game_index, parameters in enumerate(parameter_list):
            label, survivors, loop = predict_matchup(build_game(parameters), matchup.split(" x "))
            history = histories[:, game_index, :]
            survivor_indices = ["emks".index(letter) for letter in survivors]
            other_indices = [index for index in range(4)
                             if np.isfinite(history[0, index]) and index not in survivor_indices]
            midpoint = len(history) // 2
            # Check a cycle.
            if loop:
                confirmed = (all(history[-1, index] < -50 for index in other_indices)
                             and history[:, survivor_indices].min() < -50)
            # Check a winner or tied group.
            else:
                confirmed = (all(history[-1, index] > -50
                                 and history[-1, index] >= history[midpoint, index] - 1e-9
                                 for index in survivor_indices)
                             and all(history[-1, index] < -50
                                     or history[-1, index] < history[midpoint, index]
                                     for index in other_indices))
            # Record any failure.
            if not confirmed:
                failures.append((parameters, matchup, label))
    return failures


#### MAIN

if __name__ == "__main__":
    # Print Tables 1 and 2.
    print_subcase_tables()
    print_hawk_dove_table()

    # Run both checks unless only the tables were requested.
    if "--tables" not in sys.argv:
        # Run the exact check across every payoff combination.
        start_time = time.time()
        subcase_counts = check_subcases()
        print(f"Exact check: {sum(subcase_counts.values()):,} Prisoner's Dilemma payoff "
              f"combinations (steps of 1/8, T = 10) and {len(build_hd_grid())} Hawk-Dove "
              f"ratios; every sub-case gives one table. ({time.time() - start_time:.0f}s)")

        # Run the simulation check on one example per sub-case and on the paper's
        # Hawk-Dove payoffs.
        start_time = time.time()
        pd_failures = verify_by_simulation(list(subcase_examples.values()))
        hd_failures = verify_by_simulation(hawk_dove_samples, game="hd")
        outcome = ("all outcomes confirmed" if not (pd_failures or hd_failures)
                   else pd_failures + hd_failures)
        print(f"Simulation check (100,000 generations): {outcome} "
              f"({time.time() - start_time:.0f}s)")

    # For tables to paste into a document, call print_subcase_tables(markdown=True).
