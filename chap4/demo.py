a = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])

def initial_trajectory(pos,v,a,theta):
    for(int)
# define the objective function for the optimization
def objective(x, t, my_team_traj, opp_team_traj):
    """
    x: a 2x5 matrix, representing the states of the two vehicles at time t
    t: the current time step
    my_team_traj: a 2x5x5 matrix, representing the planned trajectories of my team's vehicles
    opp_team_traj: a 2x5x5 matrix, representing the planned trajectories of the opponent team's vehicles
    """0
0                                                                                                                                           ggggggggggggggggggggggggggggggggggb                                                                                                                                                                                                                                                                                                                                                                      0   print(my_team_traj)
    my_team_progress = 0
    opp_team_progress = 0

0    for i in range(my_team_traj.shape[0]):
        my_team_progress += my_team_traj[i, t, x[i]]
    for i in range(opp_team_traj.shape[0]):
        opp_team_progress += opp_team_traj[i, t, x[i]]

    return -(my_team_progress - opp_team_progress)


# define the constraints for the optimization
def constraints(x, t, my_team_traj, opp_team_traj):
    """
    x: a 2x5 matrix, representing the states of the two vehicles at time t
    t: the current time step
    my_team_traj: a 2x5x5 matrix, representing the planned trajectories of my team's vehicles
    opp_team_traj: a 2x5x5 matrix, representing the planned trajectories of the opponent team's vehicles
    """

    constraints = []

    for i in range(x.shape[0]):
        if t < my_team_traj.shape[2] - 1:
            next_state = np.argmax(my_team_traj[i, t + 1])
            constraints.append(x[i] - next_state)
        if t < opp_team_traj.shape[2] - 1:
            next_state = np.argmax(opp_team_traj[i, t + 1])
            constraints.append(x[i] - next_state)

    return constraints


# initialize the planned trajectories of the vehicles
my_team_traj = np.random.rand(2, 5, 5)
opp_team_traj = np.random.rand(2, 5, 5)

# initialize the initial states of the vehicles
x0 = np.random.randint(0, 5, (2,))
print(x0)
# iterate over time steps to find the Nash equilibrium states
for t in range(5):
    res = minimize(objective, x0, args=(t, my_team_traj, opp_team_traj),
                   constraints={'type': 'eq', 'fun': constraints, 'args': (t, my_team_traj, opp_team_traj)})
    x0 = res.x
    print(f"Time step {t + 1}: My team's states: {x0[0]}, {x0[1]}")
