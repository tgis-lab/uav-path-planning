import numpy as np
import random
import matplotlib.pyplot as plt

def read_distances(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    n_cities = len(lines) - 1
    distances = np.zeros((n_cities, n_cities))
    intra_city_distances = np.zeros(n_cities)
    city_names = lines[0].strip().split()[1:]  # 从第一行读取城市名称，跳过第一个元素
    for i in range(1, len(lines)):
        parts = lines[i].strip().split()
        intra_city_distances[i - 1] = float(parts[i])  # 读取城市内距离（对角线上的数据）
        for j in range(1, len(parts)):
            distances[j - 1][i - 1] = float(parts[j])  # 转置矩阵以匹配输入和输出
    return city_names, distances, intra_city_distances

class ACO_SA_MTSP:
    def __init__(self, n_cities, n_agents, alpha, beta, evaporation_rate, sa_initial_temperature, sa_cooling_rate, iterations, start_city_index, fairness_coefficient):
        self.n_cities = n_cities
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.sa_initial_temperature = sa_initial_temperature
        self.sa_cooling_rate = sa_cooling_rate
        self.iterations = iterations
        self.pheromone = np.ones((n_cities, n_cities))
        self.best_paths = None
        self.best_max_distance = float('inf')
        self.start_city_index = start_city_index

    def run(self, city_names, distances, intra_city_distances):
        convergence = []
        prev_best_solution = None  # 保存上一次迭代的最优解
        for iteration in range(self.iterations):
            best_paths = None
            best_max_distance = float('inf')
            for _ in range(30):  # 尝试不同的随机分组
                paths = []
                cities_per_agent = [[] for _ in range(self.n_agents)]
                available_cities = set(range(self.n_cities))
                available_cities.remove(self.start_city_index)  # 不包括起始城市
                if iteration==0:
                    while available_cities:
                        agent = random.randint(0, self.n_agents - 1)
                        city = random.choice(list(available_cities))
                        cities_per_agent[agent].append(city)
                        available_cities.remove(city)

                else:
                    prev_best_solution_indices = random.sample(range(len(self.prev_best_solution)), k=self.n_agents // 2)
                    selected_sublists = [self.prev_best_solution[i] for i in prev_best_solution_indices]
                    selected_sublists = [sublist[1:-1] if len(sublist) >= 2 and sublist[0] == 0 and sublist[-1] == 0 else sublist for sublist in selected_sublists]
                    cities_per_agent[:self.n_agents // 2] = selected_sublists


                    # 重新生成剩余的可用城市集合
                    available_cities_set = set(range(self.n_cities))
                    for path in selected_sublists:
                        available_cities_set -= set(path)  # 去掉已选择子列表中的城市


                    # 随机分配剩余城市给代理
                    remaining_cities = list(available_cities_set - {self.start_city_index})  # 移除起始城市
                    random.shuffle(remaining_cities)

                    for i in range(self.n_agents // 2, self.n_agents-1):
                        cities_per_agent[i] = []

                    for i, city in enumerate(remaining_cities):
                        agent = random.randint(self.n_agents // 2, self.n_agents-1)
                        cities_per_agent[agent].append(city)


                for agent_cities in cities_per_agent:
                    current_city = self.start_city_index
                    path = [city_names[current_city]]
                    tour_distance = 0

                    while agent_cities:
                        next_city = self.select_next_city(current_city, agent_cities, distances, intra_city_distances)
                        path.append(city_names[next_city])
                        tour_distance += distances[current_city][next_city] + intra_city_distances[next_city]
                        current_city = next_city
                        agent_cities.remove(next_city)

                    path.append(city_names[self.start_city_index])
                    tour_distance += distances[current_city][self.start_city_index] + intra_city_distances[self.start_city_index]
                    paths.append((path, tour_distance))
                    self.update_pheromone(paths)

                max_distance = max(tour_distance for _, tour_distance in paths)
                if max_distance < best_max_distance:
                    best_max_distance = max_distance
                    best_paths = paths

            if best_max_distance < self.best_max_distance:
                self.best_max_distance = best_max_distance
                self.best_paths = best_paths

            # 运行模拟退火算法来优化城市分组
            initial_solution = [path for path, _ in best_paths]
            sa_solution = self.run_simulated_annealing(initial_solution, distances, intra_city_distances)
            sa_distance = self.calculate_total_distance(sa_solution, distances, intra_city_distances)
            if sa_distance < self.best_max_distance:
                self.best_max_distance = sa_distance
                self.best_paths = [(path, self.calculate_total_distance(path, distances, intra_city_distances)) for path in sa_solution]

            if best_paths:
                self.prev_best_solution = []
                for path, _ in best_paths:
                    index_path = [city_names.index(city) for city in path]
                    self.prev_best_solution.append(index_path)


            # 更新信息素
            self.update_pheromone(self.best_paths)

            convergence.append(self.best_max_distance)

        plt.plot(convergence)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Reduction Over Time in Ant Colony Optimization with Simulated Annealing')
        plt.show()

    def select_next_city(self, current_city, available_cities, distances, intra_city_distances):
        probabilities = []
        total_prob = 0
        for city in available_cities:
            # 通过限制概率范围来避免数值溢出
            numerator = min((self.pheromone[current_city][city] ** self.alpha), 1e100) * ((1 / (distances[current_city][city] + intra_city_distances[city] + 1)) ** self.beta)
            prob = numerator

            probabilities.append(prob)
            total_prob += prob

        # 处理NaN值
        if np.isnan(total_prob) or total_prob == 0:
            # 如果total_prob为NaN或零，则将所有概率设置为均匀分布
            probabilities = [1 / len(available_cities)] * len(available_cities)
        else:
            # 归一化概率
            probabilities = [prob / total_prob for prob in probabilities]

        # 根据概率选择下一个城市
        selected_index = np.random.choice(len(available_cities), p=probabilities)
        return available_cities[selected_index]

    def run_simulated_annealing(self, initial_solution, distances, intra_city_distances):
        sa = SimulatedAnnealing(self.n_agents, self.sa_initial_temperature, self.sa_cooling_rate, self.fairness_coefficient)
        return sa.run(initial_solution, distances, intra_city_distances)

    def update_pheromone(self, paths):
        max_distance = max(tour_distance for _, tour_distance in paths)
        min_distance = min(tour_distance for _, tour_distance in paths)
        range_distance = max_distance - min_distance
        for path, tour_distance in paths:
            sum_term = 1 /max_distance
            for i in range(len(path) - 1):
                from_city = path[i]
                to_city = path[i + 1]
                from_index = city_names.index(from_city)
                to_index = city_names.index(to_city)
                # 更新信息素
                updated_pheromone= (1 - self.evaporation_rate) * self.pheromone[from_index][to_index] + sum_term

                self.pheromone[from_index][to_index] = updated_pheromone


    def calculate_total_distance(self, paths, distances, intra_city_distances):
        total_distance = 0
        for path in paths:
            for i in range(len(path) - 1):
                from_city = path[i]
                to_city = path[i + 1]
                from_index = city_names.index(from_city)
                to_index = city_names.index(to_city)
                total_distance += distances[from_index][to_index] + intra_city_distances[to_index]
        return total_distance

    def get_best_solution(self):
        total_distance = sum(tour_distance for _, tour_distance in self.best_paths)
        return self.best_paths, self.best_max_distance, total_distance

class SimulatedAnnealing:
    def __init__(self, n_agents, initial_temperature, cooling_rate, fairness_coefficient):
        self.n_agents = n_agents
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.fairness_coefficient = fairness_coefficient

    def acceptance_probability(self, energy, new_energy, temperature):
        if new_energy < energy:
            return 1.0
        return np.exp((energy - new_energy) / temperature)

    def run(self, initial_solution, distances, intra_city_distances):
        current_solution = initial_solution
        best_solution = initial_solution
        current_energy = self.calculate_max_cost(current_solution, distances, intra_city_distances)

        best_energy = current_energy
        temperature = self.initial_temperature

        while temperature > 1:
            new_solution = self.get_neighbor(current_solution)

            new_energy = self.calculate_max_cost(new_solution, distances, intra_city_distances)  # 使用最大的旅行商成本作为能量值

            if self.acceptance_probability(current_energy, new_energy, temperature) > random.random():
                current_solution = new_solution
                current_energy = new_energy

            if new_energy < best_energy:
                best_solution = new_solution
                best_energy = new_energy

            temperature *= self.cooling_rate

        return best_solution

    def get_neighbor(self, solution):
        neighbor_solution = solution.copy()

        # 选择两个不同的组
        index1, index2 = random.sample(range(len(solution)), 2)

        # 交换两个组的城市
        city1 = solution[index1].copy()
        city2 = solution[index2].copy()

        # 从每个组中选择一个城市并交换
        if len(city1)  > 3 and len(city2) > 3:
            city_index1 = random.randint(1, len(city1) - 2)
            city_index2 = random.randint(1, len(city2) - 2)

            temp = city1[city_index1]
            city1[city_index1] = city2[city_index2]
            city2[city_index2] = temp

            neighbor_solution[index1] = city1
            neighbor_solution[index2] = city2

        return neighbor_solution



    def calculate_max_cost(self, paths, distances, intra_city_distances):
        max_cost = 0
        for path in paths:
            path_cost = 0
            for i in range(len(path) - 1):
                from_city = path[i]
                to_city = path[i + 1]
                from_index = city_names.index(from_city)
                to_index = city_names.index(to_city)
                path_cost += distances[from_index][to_index] + intra_city_distances[to_index]
            if path_cost > max_cost:
                max_cost = path_cost
        return max_cost


# 示例使用
city_names, distances, intra_city_distances = read_distances('path_lengths.txt')
start_city_index = city_names.index("start")
aco_sa = ACO_SA_MTSP(n_cities=len(city_names), n_agents=5, alpha=1, beta=2, evaporation_rate=0.1, sa_initial_temperature=1000, sa_cooling_rate=0.98, iterations=1000, start_city_index=start_city_index)
aco_sa.run(city_names, distances, intra_city_distances)
best_paths, best_max_distance, total_distance = aco_sa.get_best_solution()
print(f"Best Paths: {best_paths}")
print(f"Best Maximum Distance: {best_max_distance}")
print(f"Total distance of all paths: {total_distance}")
