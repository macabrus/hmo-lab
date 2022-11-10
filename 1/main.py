import os
import csv
import random
from pprint import pprint as pp

from attr import define
from cattr import structure
from collections import defaultdict, Counter
from operator import attrgetter
from itertools import combinations, product


@define
class Player:
    id: int
    position: str
    last_name: str
    club: str
    points: int
    price: float
    quality: float = None

# pre computed structure for efficient retrieval of players
@define
class Dataset:
    raw: list[Player]
    clubs: dict[str, list[Player]]
    positions: dict[str, list[Player]]
    club_positions: dict[tuple[str, str], list[Player]]

def load_csv(file, model):
    with open(file, 'r', encoding='cp1252') as f:
        reader = csv.reader(f)
        players = []
        for row in reader:
            player = Player(
                id=int(row[0]),
                position=row[1],
                last_name=row[2],
                club=row[3],
                points=int(row[4]),
                price=float(row[5])
            )
            # my custom quality measure for players
            player.quality = player.points / player.price
            players.append(player)
        return players

def score(team: list[Player]):
    return sum(p.points for p in team)

def by_points(player):
    return player.points

def by_position(position: str):
    def fn(player):
        return player.position == position
    return fn

def first_team_lineup(team: list[Player]):
    first_team = []
    # must be exactly one goalkeeper
    first_team += sorted(filter(by_position('GK'), team), key=by_points, reverse=True)[:1]
    # at least 3 defenders
    first_team += sorted(filter(by_position('DEF'), team), key=by_points, reverse=True)[:3]
    # at least 1 forward
    first_team += sorted(filter(by_position('FW'), team), key=by_points, reverse=True)[:1]
    # select best from rest of them (excluding goalkeeper which was not initially selected)
    is_remaining = lambda p: p not in first_team and p.position != 'GK'
    non_selected = sorted(filter(is_remaining, team), key=by_points, reverse=True)
    first_team += non_selected[:11 - len(first_team)]
    return first_team

def has_at_most(players: list[Player], position, limit):
    return limit >= len([p for p in players if p.position == position])

def valid_final_solution(players: list[Player]):
    if not valid_solution(players):
        return False
    return len(players) == 15

def valid_solution(players: list[Player]):
    if len(set(map(lambda p: p.id, players))) != len(players):
        return False
    if sum(p.price for p in players) > 100:
        return False
    if not has_at_most(players, 'GK', 2):
        return False
    if not has_at_most(players, 'FW', 3):
        return False
    if not has_at_most(players, 'DEF', 5):
        return False
    if not has_at_most(players, 'MID', 5):
        return False
    club_occurences = Counter()
    for player in players:
        club_occurences[player.club] += 1
    if max(club_occurences.values()) > 3:
        return False
    return True

def group_by(fn, iterable, order=None):
    groups = defaultdict(list)
    for e in iterable:
        groups[fn(e)].append(e)
    if order:
        for g in groups:
            groups[g] = sorted(groups[g], key=order, reverse=True)
    return groups

def prepare_dataset(players):
    return Dataset(
        raw=players,
        clubs=group_by(attrgetter('club'), players, order=attrgetter('quality')),
        positions=group_by(attrgetter('position'), players, order=attrgetter('quality')),
        club_positions=group_by(attrgetter('club', 'position'), players, order=attrgetter('quality'))
    )

def exp_weights(n): # exponentially descending
    return [2 ** -i for i in range(n)]

def linear_weights(n): # linearly descending
    return [i for i in range(n, 0, -1)]

def generate_initial_solution(dataset: Dataset, max_iter=100):
    team = []
    # selecting n from every position
    current_iter = 0
    for position, n in [('GK', 2), ('DEF', 5), ('MID', 5), ('FW', 3)]:
        len_before = len(team)
        while n > len(team) - len_before:
            #print(len(team))
            position_players = dataset.positions[position]
            weights = linear_weights(len(position_players))
            #if len(team) >= 11: # flip weights if filled team
            #    weights = weights[::-1]
            selection = random.choices(position_players, weights=weights, k=1)
            #print(selection)
            if valid_solution(team + selection):
                team += selection
            current_iter += 1
            if current_iter >= max_iter:
                print('exceeded')
                return None
    return team

def unique(iterable):
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item

def grasp(dataset, original_solution, n_neighbourhood=1):
    original_score = score(original_solution)
    current_best = original_solution[:]
    current_best_score = original_score
    # current = 0
    # total = len(list(combinations(range(len(original_solution)), n_neighbourhood)))
    for solution_indices in combinations(range(len(original_solution)), n_neighbourhood):
        # print(f'{current}/{total}')
        # current += 1
        positions = tuple(original_solution[i].position for i in solution_indices)
        len_ranges = tuple(range(len(dataset.positions[p])) for p in positions)
        for sample_indices in product(*len_ranges):
            tmp_solution = original_solution[:]
            for sol_index, position, sample_index in zip(solution_indices, positions, sample_indices):
                tmp_solution[sol_index] = dataset.positions[position][sample_index]
            if not valid_final_solution(tmp_solution):
                continue
            tmp_score = score(tmp_solution)
            if tmp_score > current_best_score:
                print('Grasp managed to optimize!')
                current_best = tmp_solution
                current_best_score = tmp_score
    return current_best

def ids(players: list[Player]):
    return [p.id for p in players]

def main():
    random.seed(1)
    f = '1/2022_instance3.csv'
    instance, _ = os.path.splitext(os.path.basename(f))
    dataset = prepare_dataset(load_csv(f, Player))
    while 1:
        greedy_team = generate_initial_solution(dataset)
        if greedy_team is None:  # failed to find solution
            continue  # retry
        greedy_lineup = first_team_lineup(greedy_team)
        greedy_score = score(greedy_lineup)
        grasp_team = grasp(dataset, greedy_team, n_neighbourhood=2)
        grasp_lineup = first_team_lineup(grasp_team)
        grasp_score = score(grasp_lineup)
        print(f'Greedy: {greedy_score} {ids(greedy_team)}')
        print(f'GRASP : {grasp_score} {ids(grasp_team)}')
        # if  best is None or team_score > best:
        #     best = team_score
        #     print(f'Found new best team: {list(map(lambda p: p.id, first_team))}')
        #     print(f'Score: {team_score}')
        # with open(f'best-{instance}.txt', 'w') as f:
        #     first_team_ids = [p.id for p in first_team]
        #     f.write(','.join(map(str, first_team_ids)) + '\n')
        #     f.write(','.join(str(p.id) for p in team if p.id not in first_team_ids))
        # print(first_team)
        # print('--- GRASP optimized solution ---')
        # grasp_team = grasp(dataset, team, n_neighbourhood=2)
        # grasp_first_team = first_team_lineup(team)
        # print(score(grasp_first_team))
    # print(valid_final_solution(team))

if __name__ == '__main__':
    main()
