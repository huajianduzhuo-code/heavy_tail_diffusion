### Simulation of an arrival process with arrival rate lambda(t), 0<t<T
import numpy as np


# Define the time-varying intensity function (lambda(t))
def intensity_function(t, T=100):
    return (0.75 + 0.25 * np.sin(2 * np.pi * t / T))*1/10  # Example of a time-varying intensity function


# Simulate nonhomogeneous Poisson process without discretization using thinning method
def nonhomogeneous_poisson_process(t0, T, intensity_func, LAM_MAX=1.1/10, a=2, m=1):
    t = t0  # Initialize time
    events = []  # List to store event times

    while t < T:
        u = np.random.uniform(0, 1)  # Generate a uniform random number
        t = t-np.log(u) / LAM_MAX  # Time to the next candidate event

        if t < T:
            rho = intensity_func(t, T) / LAM_MAX  # Probability of accepting the candidate event
            if np.random.uniform(0, 1) < rho:
                # event time will be recorded, now decide the number of arrivals
                s = (np.random.pareto(a) + 1) * m
                n_arrived = np.random.poisson(s)
                events.extend([t]*n_arrived)  # Record the event time

    return np.array(events)


class ServiceStation:
    def __init__(self, n_servers, dist_type, arrival_intensity_function, t0=0, T=100, N0=0, original_arrivals = np.array([])):
    # t0 is starting time, N0 is number of customers already in system at t0, original_arrivals is the arrival time of existing customers (only the first n_servers customers' arrival time matters)
        self.T = T
        self.t0 = t0
        self.n_servers = n_servers
        self.arrivals = np.array([])
        self.departs = np.array([])
        self.dist_type = dist_type
        self.server_free = np.zeros(n_servers) # record the last free time of the servers
        self.arrival_intensity_function = arrival_intensity_function

        # generate exogenous arrival process (including the arrivals before t0)

        # for original arrivals, only the arrival time for the first "n_servers" customers matter. All the arrival time of other customers can be regarded as t0
        # if len(original_arrivals)<N0, fill the empty ones with t0
        assert len(original_arrivals)<=N0, "too many arrival times"
        num_to_fill = max(N0-len(original_arrivals), 0)
        original_arrivals = np.pad(original_arrivals, (0,num_to_fill), constant_values = t0) # arrivals before t0
        remaining_arrivals = nonhomogeneous_poisson_process(t0, T, self.arrival_intensity_function) # arrivals after t0
        self.exogenous_arrivals = np.sort(np.concatenate([original_arrivals,remaining_arrivals]))

    def set_distribution_args(self, *args):
        self.args = args  # Store the arguments as the class attribute

    def serving_time(self):
        dist_func = getattr(np.random, self.dist_type, None)

        if dist_func is not None:
        # Generate a random variable using the chosen distribution function
            random_variable = dist_func(*self.args)
            assert len(random_variable)==1, "length mismatch"
            return random_variable[0]
        else:
            raise ValueError("Invalid distribution type")

    def line_length(self, t):
        return np.sum((self.arrivals[:, np.newaxis] <= t) & (self.departs[:, np.newaxis] > t), axis=0)
    
# list operation
def insert_event(events,new_event):
    # extract time for sorting
    time = events[:,0]

    # Find the index where the new event should be inserted to maintain sorting
    index_to_insert = np.searchsorted(time, new_event[0])

    # Insert the new row at the appropriate index to maintain sorted order
    new_events = np.insert(events, index_to_insert, new_event, axis=0)

    return new_events