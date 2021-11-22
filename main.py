import copy
from typing import List
import random
import pdb

# Set up
RANDOM_GEN = None

# Configuration
POPULATION_LIMIT = 50
DISTANCE = [
    [0, 1, 1, 4],
    [1, 0, 2, 5],
    [1, 2, 0, 3],
    [4, 5, 3, 0]
]
ITEMS = [
    [2, 3],
    [3, 2],
    [1, 1],
    [4, 1]
]

ROUTE_FITNESS_WEIGHT = 1
BUY_FITNESS_WEIGHT = 1

GENERATION_THRESHOLD = 100

# Definition
class Person:
    def __init__(self, route: List[int], items: List[int], buy_order: List[int]):
        self.route = list(route) #genome 2
        self.buy = list(items)  #genome 1
        self.buy_order = list(buy_order) # genome 3

        # data
        self.bought = 0
        self.done = False
        self.qty_to_buy = len(self.buy)
        self.fitness = self.calculate_fitness()


    def update_fitness(self):
        self.bought = 0
        self.done = False
        self.fitness = self.calculate_fitness()

    def _update_bought(self):
        self.bought += 1
        if self.bought == self.qty_to_buy:
            self.done = True

    def calculate_fitness(self):
        """Calculate fitness per person"""
        sum_route = 0
        sum_items = 0

        for i in range(len(self.route)-1):
            #calculate route fitness.
            sum_route += DISTANCE[self.route[i]][self.route[i+1]]
            #calculate items
            if self.route[i+1] in self.buy:
                index = self.buy.index(self.route[i+1])
                sum_items += ITEMS[self.route[i+1]][self.buy_order[index]]
                self._update_bought()
            if self.done:
                break
            
        # for i in range(len(self.route)):
        #     if self.done:
        #         break
        #     if i != len(self.route) - 1:
        #         sum_route += DISTANCE[temp_route[i]][self.route[i]]
        #         print(f"f[{i}]: {DISTANCE[self.route[i]][temp_route[i]]}")
        #     if self.route[i] in self.buy:
        #         order = self.buy.index(self.route[i])
        #         sum_items += ITEMS[self.route[i]][self.buy_order[order]]
        #         print(f"c[{i}]: {ITEMS[self.route[i]][self.buy_order[order]]}")
        #         self._update_bought()
        #         if self.done:
        #             print("im fucking done")
        #         else:
        #             print("im not done")
        
        total = sum_items * BUY_FITNESS_WEIGHT + sum_route * ROUTE_FITNESS_WEIGHT
        return total

    def __str__(self):
        return ((self.route.__str__() + self.buy.__str__() + self.buy_order.__str__()) + " - " + str(self.fitness))

# Mutation for new generations
class Mutator:
    def _shuffle_buy_order(buy_order: List[str]) -> List[str]:
        return RANDOM_GEN.sample(buy_order, k=len(buy_order))

    def _modify_buy(p: Person) -> Person:
        """Modify buy."""
        new_p = copy.deepcopy(p)
        rand_buy = RANDOM_GEN.choice(p.buy)
        rand_node = RANDOM_GEN.choice(p.route)

        buy_index = new_p.buy.index(rand_buy)
        new_p.buy.pop(buy_index)
        if rand_node not in new_p.buy:
            new_p.buy.insert(buy_index, rand_node)
        else:
            new_p.buy.insert(buy_index, rand_buy)
            new_p.buy_order = Mutator._shuffle_buy_order(buy_order=new_p.buy_order)
        new_p.update_fitness()
        # if new_p.fitness < p.fitness:
        #     return new_p
        return new_p
         

    def _modify_route(p: Person) -> Person:
        """Function only swaps once"""
        route = p.route[1:]
        index = RANDOM_GEN.choice(range(len(route)))
        replaced = route.pop(index)
        index = RANDOM_GEN.choice(range(len(route)))
        route.insert(index ,replaced)
        route.insert(0, 0) # append start
        p.route = list(route)
        return copy.deepcopy(p)

    def modify(p: Person) -> Person:
        """Modifies attributes depending on chances.
        
        Here are the set of rules based on random generation. 
        - Just argue about modus tollens and modus ponens seen in class. /shrug
        """
        random = RANDOM_GEN.randint(1, 100)
        p = Mutator._modify_route(p)
        if random <= 65:
            p = Mutator._modify_route(p)
        elif random <= 15:
            p = Mutator._modify_route(p)

        random = RANDOM_GEN.randint(1, 100)
        p = Mutator._modify_buy(p)

        return p

# Generator
class PopulationGenerator:
    def _util_fitness_sort(p: Person):
        """Fitness sort utility for .sort()"""
        return p.fitness

    def natural_selection(pop: List[Person]) -> List[Person]:
        """Destroy bad contenders"""
        new_list = list(pop)
        new_list.sort(key=PopulationGenerator._util_fitness_sort)
        # Thanos'd
        new_list = new_list[:(POPULATION_LIMIT//2)]
        return new_list

    def _mutation(self, p: Person) -> Person:
        """Create a child of Person based on a previous Person."""
        new_p = copy.deepcopy(p)
        new_p = Mutator.modify(p=new_p)
        return new_p

    def generate_first(self) -> List[Person]:
        """z"""
        route_choices = [i for i in range(len(DISTANCE))]
        route_choices.pop(0)
        buy_order_choices = [i for i in range(len(ITEMS[0]))]
        population = []
        for i in range(POPULATION_LIMIT):
            route = []
            buy = []
            buy_order = []
            route = RANDOM_GEN.sample(route_choices, k=len(route_choices))
            route.insert(0,0)
            buy = RANDOM_GEN.sample(route, k=2)
            buy_order = RANDOM_GEN.sample(buy_order_choices, k=len(buy_order_choices))
            population.append(
                Person(
                    route=list(route),
                    items=list(buy),
                    buy_order=list(buy_order)
                )
            )

        # p.route = random.sample(p.route[1:], k=len(p.route)) # remove initial
        # p.route.insert(0, 0) # add initial at 0
        return population

    def generate_new(self, pop: List[Person]) -> List[Person]:
        """Generate new generation"""
        new_pop = PopulationGenerator.natural_selection(pop=pop)
        amount_to_generate = POPULATION_LIMIT - len(new_pop)
        p = None
        for i in range(amount_to_generate):
            p = self._mutation(p=new_pop[i])
            new_pop.append(copy.deepcopy(p))
        return new_pop


def print_generation(pop: List[Person], gen: int):
    print(f"Gen {gen+1} min: {min((p.fitness for p in pop))}")
    for person in pop:
        print(person)
    

def run() -> Person:
    pop_gen = PopulationGenerator()
    for i in range(GENERATION_THRESHOLD):
        if i == 0:
            population = pop_gen.generate_first()
        else:
            population = list(pop_gen.generate_new(pop=population))
        print_generation(pop=population, gen=i)
    population.sort(key=PopulationGenerator._util_fitness_sort)
    return population[0] # Return first one. AKA winner.

def _util_init_random_gen(seed):
    global RANDOM_GEN
    RANDOM_GEN = random.Random(seed)

if __name__ == "__main__":
    seed = "quiero morir"
    
    RANDOM_GEN = random.Random(seed)
    winner = run()
    print(winner)