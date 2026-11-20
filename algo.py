import numpy as np


def bat_algorithm(
    objective,
    dim=10,
    lb=-5.12,
    ub=5.12,
    n=10,
    t_max=1000,
    f_min=0,
    f_max=2,
    A0=1.0,
    r0=1.0,
    alpha=0.97,
    gamma=0.1,
    sigma=0.1,
    tol=None,
    seed=None,
):
    """Run the Bat Algorithm to optimize the given objective function."""

    # --------------------------------------------------
    # Step 1: Initialize the random number generator
    # --------------------------------------------------

    # Create a reproducible random generator if a seed is provided.
    rng = np.random.default_rng(seed)

    # --------------------------------------------------
    # Step 2: Initialize the bat population
    # --------------------------------------------------

    # Generate random positions for all bats within the search boundaries.
    positions = lb + (ub - lb) * rng.random((n, dim))

    # Initialize all bat velocities to zero.
    velocities = np.zeros((n, dim))

    # Initialize the loudness of every bat.
    A = np.full(n, A0)

    # Initialize the pulse emission rate of every bat.
    r = np.full(n, r0)

    # --------------------------------------------------
    # Step 3: Evaluate the initial population
    # --------------------------------------------------

    # Calculate the objective function value for each bat.
    fitness = np.array([objective(pos) for pos in positions])

    # Identify the bat with the lowest objective value.
    best_idx = np.argmin(fitness)

    # Store the best position and its corresponding fitness.
    best = positions[best_idx].copy()
    f_best = fitness[best_idx]

    # --------------------------------------------------
    # Step 4: Initialize the optimization history
    # --------------------------------------------------

    # Store the best fitness value found at each iteration.
    history = []

    # Store the iteration at which the stopping condition is satisfied.
    stop_iter = None

    # --------------------------------------------------
    # Step 5: Main optimization loop
    # --------------------------------------------------

    for t in range(t_max):

        # Reduce the loudness of all bats over time.
        A *= alpha

        # Update the pulse emission rate according to the book's equation.
        r *= 1 - np.exp(-gamma * t)

        # --------------------------------------------------
        # Step 8: Update each bat
        # --------------------------------------------------

        for i in range(n):

            # Generate a random frequency within the specified range.
            frequency = f_min + (f_max - f_min) * rng.random()

            # Update the bat's velocity using its previous velocity
            # and the difference between its position and the global best.
            velocities[i] += (positions[i] - best) * frequency

            # Calculate the new candidate position.
            candidate = positions[i] + velocities[i]

            # --------------------------------------------------
            # Step 9: Perform local search
            # --------------------------------------------------

            # Generate a random number to decide whether local search is used.
            if rng.random() > r[i]:

                # Generate a new candidate around the current global best.
                # The random perturbation is controlled by sigma and loudness.
                candidate = best + sigma * rng.standard_normal(dim) * np.mean(A)

            # --------------------------------------------------
            # Step 10: Evaluate the new candidate
            # --------------------------------------------------

            # Calculate the objective function value of the candidate.
            f_new = objective(candidate)
    
            # --------------------------------------------------
            # Step 11: Update the global best solution
            # --------------------------------------------------

            # If the candidate improves the global best, accept it as the new best.
            if f_new <= f_best:
                best = candidate.copy()
                f_best = f_new

            # --------------------------------------------------
            # Step 12: Update the bat's position
            # --------------------------------------------------

            # Accept the candidate if it improves the bat's fitness
            # and a random number exceeds the bat's loudness.
            if f_new <= fitness[i] and rng.random() > A[i]:

                # Replace the bat's old position with the new candidate.
                positions[i] = candidate

                # Update the bat's fitness value.
                fitness[i] = f_new

        # --------------------------------------------------
        # Step 13: Check the stopping condition
        # --------------------------------------------------

        # Stop optimization if the best fitness reaches the tolerance.
        if tol is not None and f_best < tol:
            stop_iter = t
            break

    # --------------------------------------------------
    # Step 14: Return the optimization results
    # --------------------------------------------------

    # Return the best position, its fitness, fitness history,
    # and the iteration at which early stopping occurred.
    return best, f_best, history, stop_iter


def bat_algorithm_steps(
    objective,
    dim=10,
    lb=-5.12,
    ub=5.12,
    n=10,
    t_max=1000,
    f_min=0,
    f_max=2,
    A0=1.0,
    r0=1.0,
    alpha=0.97,
    gamma=0.1,
    sigma=0.1,
    variant="book",
    tol=None,
    seed=None,
):
    """Run the Bat Algorithm incrementally, yielding one frame per iteration."""

    if variant not in ("book", "D", "eq28"):
        raise ValueError(f"Unknown variant: {variant}")

    rng = np.random.default_rng(seed)

    # Initialize bats
    positions = lb + (ub - lb) * rng.random((n, dim))
    velocities = np.zeros((n, dim))

    A = np.array([A0] * n)
    r0_vec = np.array([r0] * n)
    r = np.copy(r0_vec)

    fitness = np.array([objective(pos) for pos in positions])
    f_best = np.min(fitness)
    best = positions[np.argmin(fitness)].copy()

    # Yield the initial state (iteration 0)
    yield dict(
        iteration=0,
        pos=positions.copy(),
        best=best.copy(),
        f=float(f_best),
        A=float(np.mean(A)),
        r=float(np.mean(r)),
        cands=np.empty((0, dim)),
    )

    # Perform one iteration at a time
    for t in range(t_max):
        A = alpha * A

        if variant in ("eq28", "D"):
            r = r0_vec * (1 - np.exp(-gamma * t))
        elif variant == "book":
            r = r * (1 - np.exp(-gamma * t))

        cands = []

        for i in range(n):
            beta = rng.random()
            freq_i = f_min + (f_max - f_min) * beta

            velocities[i] = velocities[i] + (positions[i] - best) * freq_i

            candidate = positions[i] + velocities[i]

            rand = rng.random()

            if (rand < r[i] and variant == "D") or (
                rand > r[i] and variant in ("book", "eq28")
            ):
                candidate = best + sigma * rng.standard_normal(dim) * np.mean(A)

            cands.append(candidate.copy())

            f_new = objective(candidate)

            if f_new <= f_best:
                best = candidate.copy()
                f_best = f_new

            if f_new <= fitness[i] and rng.random() > A[i]:
                positions[i] = candidate
                fitness[i] = f_new

        # Yield the state after completing this iteration
        yield dict(
            iteration=t + 1,
            pos=positions.copy(),
            best=best.copy(),
            f=float(f_best),
            A=float(np.mean(A)),
            r=float(np.mean(r)),
            cands=np.array(cands),
        )

        if tol is not None and f_best < tol:
            break
