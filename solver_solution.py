# Import library
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from pulp import *
import time


# Define Function
def define_tsp(n_point):
	
	'''
	This function creates model of TSP.

	n_customer : Number of cities or customers to visit
	x, y : Coordinates of each city
	'''

	# Random number specification
	np.random.seed(0)

	# Set coordinates of each city
	df = pd.DataFrame({
		'x' : np.random.randint(0, 100, n_point),
		'y' : np.random.randint(0, 100, n_point)
	})

	# Set coordinates of initial city
	df.iloc[0]['x'] = 0
	df.iloc[0]['y'] = 0

	return df


def show_model(df, ax):

	'''
	This function plots the model
	'''

	# ax.figure(figsize=(5, 5))
 
	# draw problem state
	for i, row in df.iterrows():
		if i == 0:
			ax.scatter(row['x'], row['y'], c='r')
			ax.text(row['x'] + 1, row['y'] + 1, 'depot')
		else:
			ax.scatter(row['x'], row['y'], c='black')
			ax.text(row['x'] + 1, row['y'] + 1, f'{i}')

	ax.set_xlim([-10, 110])
	ax.set_ylim([-10, 110])
	ax.set_title('points: id')


def calc_distance(a, b):

	a = np.array([a['x'], a['y']])
	b = np.array([b['x'], b['y']])
	u = b - a

	return np.linalg.norm(u)


# Calculate Proguram
if __name__ == '__main__':

	# Record of calculate time
	record_time = []

	# Record of result
	result_list = []

	# Record of route
	route_list = []

	for point in range(1, 11):

		# Start watch
		start = time.time()
		
		# Set TSP
		n_customer = point
		n_point = n_customer + 1
		model = define_tsp(n_point)

		# Set problem
		problem = pulp.LpProblem(sense=LpMinimize)

		# Set valiables
		# x is Determining variable
		# If points i to j are the optimal route, x_i_j = 1, else x_i_j = 0
		x = pulp.LpVariable.dicts('x', ((i, j) for i in range(n_point) for j in range(n_point)), lowBound=0, upBound=1, cat='Binary')

		# u_i represents the number of each point
		u = pulp.LpVariable.dicts('u', (i for i in range(n_point)), lowBound=1, upBound=n_point, cat='Integer')

		# Set Objective function
		distance = [[calc_distance(model.iloc[i, :], model.iloc[j, :]) for j in range(n_point)] for i in range(n_point)]
		problem += pulp.lpSum(distance[i][j] * x[i, j] for i in range(n_point) for j in range(n_point))

		# Set Constrains
		# No move to the same city
		for i in range(n_point):
			problem += x[i, i] == 0

		# The section between cities always passes only once
		for i in range(n_point):
			problem += pulp.lpSum(x[i, j] for j in range(n_point)) == 1
			problem += pulp.lpSum(x[j, i] for j in range(n_point)) == 1

		# Eliminate subtour
		for i in range(n_point):
			for j in range(n_point):
				if i != j and (i !=0 and j != 0):
					problem += u[i] - u[j] <= n_point * (1 - x[i, j]) - 1

		# Solve problems
		status = problem.solve()
		result_list.append(LpStatus[status])

		# End watch
		end = time.time()

		# Record calculate time
		record_time.append(end - start)

		# check TSP problem and optimized route
		# draw problem state
		# plt.figure(figsize=(5, 5))
		# show_model(model)

		# draw optimal route
		routes = [(i, j) for i in range(n_point) for j in range(n_point) if value(x[i, j]) == 1]
		route_list.append(routes)
		# arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
		# for i, j in routes:
		# 	plt.annotate('', xy=[model.iloc[j]['x'], model.iloc[j]['y']], xytext=[model.iloc[i]['x'], model.iloc[i]['y']], arrowprops=arrowprops)
		# plt.show()


	# Save images
	fig, axes = plt.subplots(len(route_list), 1, figsize=(10, 50))
	arrowprops = dict(arrowstyle='->', connectionstyle='arc3', edgecolor='blue')
	for n in range(len(route_list)):
		for i, j in route_list[n]:
			show_model(model.iloc[:n+2, :], axes[n])
			axes[n].annotate('', xy=[model.iloc[j]['x'], model.iloc[j]['y']], xytext=[model.iloc[i]['x'], model.iloc[i]['y']], arrowprops=arrowprops)
	# fig.tight_layout()
	fig.savefig('images/opt.png')


	# Save calculatetime and value as csv
	df = pd.DataFrame({'calculate_time' : record_time, 'result': result_list})
	df.to_csv('result/data.csv', index=False)


