import numpy as np


def evaluate(x, objective):
    value = np.asarray(objective(x))
    if (
        value.ndim != 0
        or not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value)
    ):
        raise ValueError("Objective must return one real number.")

    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Objective returned a non-finite value.")
    return value


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
    r0=0.5,
    alpha=0.97,
    gamma=0.1,
    sigma=0.1,
    tol=None,
    seed=None,
):
    """Run the Bat Algorithm to optimize the given objective function."""

    # --------------------------------------------------
    # Step 0: Validate input parameters
    # --------------------------------------------------
    # Validate that the objective function is callable.
    if callable(objective):
        pass
    else:
        raise TypeError("The objective function must be callable.")

    # Validate the types of the input parameters.
    if not isinstance(dim, int):
        raise TypeError("The dimension must be an integer.")
    if not isinstance(n, int):
        raise TypeError("The number of bats must be an integer.")
    if not isinstance(t_max, int):
        raise TypeError("The maximum number of iterations must be an integer.")
    if not isinstance(f_min, (int, float)):
        raise TypeError("The minimum frequency must be a number.")
    if not isinstance(f_max, (int, float)):
        raise TypeError("The maximum frequency must be a number.")
    if not isinstance(A0, (int, float)):
        raise TypeError("The initial loudness must be a number.")
    if not isinstance(r0, (int, float)):
        raise TypeError("The initial pulse emission rate must be a number.")
    if not isinstance(alpha, (int, float)):
        raise TypeError("The absorption coefficient must be a number.")
    if not isinstance(gamma, (int, float)):
        raise TypeError("The scaling factor must be a number.")
    if not isinstance(sigma, (int, float)):
        raise TypeError("The local search parameter must be a number.")

    # Validate the values of the input parameters.
    if dim <= 0:
        raise ValueError("The dimension must be a positive integer.")
    if n <= 0:
        raise ValueError("The number of bats must be a positive integer.")
    if t_max <= 0:
        raise ValueError("The maximum number of iterations must be a positive integer.")
    if lb >= ub:
        raise ValueError("The lower bound must be less than the upper bound.")
    if f_min >= f_max:
        raise ValueError(
            "The minimum frequency must be less than the maximum frequency."
        )
    if f_min < 0 or f_max < 0:
        raise ValueError("Frequencies must be non-negative.")
    if A0 <= 0:
        raise ValueError("The initial loudness must be positive.")
    if r0 <= 0 or r0 >= 1:
        raise ValueError("The initial pulse emission rate must be in (0, 1).")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("The absorption coefficient must be in (0, 1).")
    if gamma <= 0:
        raise ValueError("The scaling factor must be positive.")
    if sigma <= 0:
        raise ValueError("The local search parameter must be positive.")

    # Check for finite numerical parameters
    parameters = [lb, ub, f_min, f_max, A0, r0, alpha, gamma, sigma]
    if not np.all(np.isfinite(parameters)):
        raise ValueError("All numerical parameters must be finite.")

    if tol is not None and not np.isfinite(tol):
        raise ValueError("tol must be finite or None.")

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
    A = np.full(n, A0, dtype=float)

    # Initialize the pulse emission rate of every bat.
    r = np.full(n, r0, dtype=float)
    r0 = r.copy()

    # --------------------------------------------------
    # Step 3: Evaluate the initial population
    # --------------------------------------------------

    # Calculate the objective function value for each bat.
    fitness = np.array([evaluate(pos, objective) for pos in positions], dtype=float)

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

    # Store the number of completed iterations at which the stopping condition is satisfied.
    stop_iter = None

    # --------------------------------------------------
    # Step 5: Main optimization loop
    # --------------------------------------------------

    for t in range(t_max):
        # --------------------------------------------------
        # Step 6: Check the stopping condition
        # --------------------------------------------------

        # Stop optimization if the best fitness reaches the tolerance.
        if tol is not None and f_best <= tol:
            stop_iter = t
            break

        # --------------------------------------------------
        # Step 7: Update each bat
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
            # Step 8: Perform local search
            # --------------------------------------------------

            # Generate a random number to decide whether local search is used.
            if rng.random() > r[i]:

                # Generate a new candidate around the current global best.
                # The random perturbation is controlled by sigma and loudness.
                candidate = best + sigma * rng.standard_normal(dim) * np.mean(A)

            # --------------------------------------------------
            # Step 9: Evaluate the new candidate
            # --------------------------------------------------

            if np.any(candidate < lb) or np.any(candidate > ub):
                # If the candidate is out of bounds, clip it to the search boundaries.
                np.clip(candidate, lb, ub, out=candidate)

            # Calculate the objective function value of the candidate.
            f_new = evaluate(candidate, objective)

            # --------------------------------------------------
            # Step 10: Update the global best solution
            # --------------------------------------------------

            # If the candidate improves the global best, accept it as the new best.
            if f_new <= f_best:
                best = candidate.copy()
                f_best = f_new

            # --------------------------------------------------
            # Step 11: Update the bat's position
            # --------------------------------------------------

            # Accept the candidate if it improves the bat's fitness
            # and a random number is less than the bat's loudness.
            if f_new <= fitness[i] and rng.random() < A[i]:

                # Replace the bat's old position with the new candidate.
                positions[i] = candidate

                # Update the bat's fitness value.
                fitness[i] = f_new

                # Reduce the loudness of all bats over time.
                A[i] = alpha * A[i]

                # Update the pulse emission rate according to the book's equation.
                r[i] = r0[i] * (1 - np.exp(-gamma * (t + 1)))

        # --------------------------------------------------
        # Step 12: Record the best fitness value for this iteration
        # --------------------------------------------------
        history.append(float(f_best))

    # --------------------------------------------------
    # Step 13: Check the stopping condition after the loop
    # --------------------------------------------------
    if tol is not None and f_best <= tol:
        stop_iter = len(history)

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
    r0=0.5,
    alpha=0.97,
    gamma=0.1,
    sigma=0.1,
    tol=None,
    seed=None,
):
    """Run the Bat Algorithm incrementally, yielding one frame per iteration."""

    # --------------------------------------------------
    # Step 0: Validate input parameters
    # --------------------------------------------------
    # Validate that the objective function is callable.
    if callable(objective):
        pass
    else:
        raise TypeError("The objective function must be callable.")

    # Validate the types of the input parameters.
    if not isinstance(dim, int):
        raise TypeError("The dimension must be an integer.")
    if not isinstance(n, int):
        raise TypeError("The number of bats must be an integer.")
    if not isinstance(t_max, int):
        raise TypeError("The maximum number of iterations must be an integer.")
    if not isinstance(f_min, (int, float)):
        raise TypeError("The minimum frequency must be a number.")
    if not isinstance(f_max, (int, float)):
        raise TypeError("The maximum frequency must be a number.")
    if not isinstance(A0, (int, float)):
        raise TypeError("The initial loudness must be a number.")
    if not isinstance(r0, (int, float)):
        raise TypeError("The initial pulse emission rate must be a number.")
    if not isinstance(alpha, (int, float)):
        raise TypeError("The absorption coefficient must be a number.")
    if not isinstance(gamma, (int, float)):
        raise TypeError("The scaling factor must be a number.")
    if not isinstance(sigma, (int, float)):
        raise TypeError("The local search parameter must be a number.")

    # Validate the values of the input parameters.
    if dim <= 0:
        raise ValueError("The dimension must be a positive integer.")
    if n <= 0:
        raise ValueError("The number of bats must be a positive integer.")
    if t_max <= 0:
        raise ValueError("The maximum number of iterations must be a positive integer.")
    if lb >= ub:
        raise ValueError("The lower bound must be less than the upper bound.")
    if f_min >= f_max:
        raise ValueError(
            "The minimum frequency must be less than the maximum frequency."
        )
    if f_min < 0 or f_max < 0:
        raise ValueError("Frequencies must be non-negative.")
    if A0 <= 0:
        raise ValueError("The initial loudness must be positive.")
    if r0 <= 0 or r0 >= 1:
        raise ValueError("The initial pulse emission rate must be in (0, 1).")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("The absorption coefficient must be in (0, 1).")
    if gamma <= 0:
        raise ValueError("The scaling factor must be positive.")
    if sigma <= 0:
        raise ValueError("The local search parameter must be positive.")

    # Check for finite numerical parameters
    parameters = [lb, ub, f_min, f_max, A0, r0, alpha, gamma, sigma]
    if not np.all(np.isfinite(parameters)):
        raise ValueError("All numerical parameters must be finite.")

    if tol is not None and not np.isfinite(tol):
        raise ValueError("tol must be finite or None.")

    # Create a reproducible random generator if a seed is provided.
    rng = np.random.default_rng(seed)

    # Generate random positions for all bats within the search boundaries.
    positions = lb + (ub - lb) * rng.random((n, dim))

    # Initialize all bat velocities to zero.
    velocities = np.zeros((n, dim))

    # Initialize the loudness of every bat.
    A = np.full(n, A0, dtype=float)

    # Initialize the pulse emission rate of every bat.
    r = np.full(n, r0, dtype=float)
    r0 = r.copy()

    # Calculate the objective function value for each bat.
    fitness = np.array([evaluate(pos, objective) for pos in positions], dtype=float)

    # Identify the bat with the lowest objective value.
    best_idx = np.argmin(fitness)

    # Store the best position and its corresponding fitness.
    best = positions[best_idx].copy()
    f_best = fitness[best_idx]

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

    history = []  # Store the best fitness value found at each iteration.

    # Perform one iteration at a time
    for t in range(t_max):
        # Stop optimization if the best fitness reaches the tolerance.
        if tol is not None and f_best <= tol:
            stop_iter = t
            break

        cands = []  # List to store candidate solutions for this iteration

        for i in range(n):
            # Generate a random frequency within the specified range.
            frequency = f_min + (f_max - f_min) * rng.random()

            # Update the bat's velocity using its previous velocity
            # and the difference between its position and the global best.
            velocities[i] += (positions[i] - best) * frequency

            # Calculate the new candidate position.
            candidate = positions[i] + velocities[i]

            # Generate a random number to decide whether local search is used.
            if rng.random() > r[i]:

                # Generate a new candidate around the current global best.
                # The random perturbation is controlled by sigma and loudness.
                candidate = best + sigma * rng.standard_normal(dim) * np.mean(A)

            cands.append(candidate.copy())

            if np.any(candidate < lb) or np.any(candidate > ub):
                # If the candidate is out of bounds, clip it to the search boundaries.
                np.clip(candidate, lb, ub, out=candidate)

            # Calculate the objective function value of the candidate.
            f_new = evaluate(candidate, objective)

            # If the candidate improves the global best, accept it as the new best.
            if f_new <= f_best:
                best = candidate.copy()
                f_best = f_new

            # Accept the candidate if it improves the bat's fitness
            # and a random number is less than the bat's loudness.
            if f_new <= fitness[i] and rng.random() < A[i]:

                # Replace the bat's old position with the new candidate.
                positions[i] = candidate

                # Update the bat's fitness value.
                fitness[i] = f_new

                # Reduce the loudness of all bats over time.
                A[i] = alpha * A[i]

                # Update the pulse emission rate according to the book's equation.
                r[i] = r0[i] * (1 - np.exp(-gamma * (t + 1)))

        history.append(float(f_best))

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

        if tol is not None and f_best <= tol:
            stop_iter = len(history)
