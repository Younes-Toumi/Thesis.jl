function pso_optimize(f::Function,
                      initial_particles::Matrix{Float64},
                      bounds::Matrix{Float64};
                      max_iter::Int = 100,
                      w::Float64 = 0.8,
                      c1::Float64 = 1.5,
                      c2::Float64 = 1.5,
                      mode::Symbol = :max,
                      delta::Float64 = 1e-4,
                      track::Bool = true)

    n_particles, dim = size(initial_particles)

    # state initialization
    pos = copy(initial_particles) # current particle positions (n × d)
    vel = zeros(n_particles, dim) # particle velocities (n × d)

    # initial evaluation of objective function
    fvals = [f(pos[i, :]) for i in 1:n_particles]

    # best known position per particle (personal best)
    pbest_pos = copy(pos)
    pbest_val = copy(fvals)

    # global best initialization
    best_idx = argmax(pbest_val)
    gbest_pos = copy(pbest_pos[best_idx, :])
    gbest_val = pbest_val[best_idx]

    # optional trajectory storage (for visualization/debugging)
    history = track ? Vector{Matrix{Float64}}() : nothing

    # objective comparison rule (handles max vs min optimization)
    better(a, b) = mode == :max ? (a > b) : (a < b)

    # main PSO loop
    for iter in 1:max_iter

        # particle update loop
        for i in 1:n_particles

            # random coefficients per dimension
            r1 = rand(dim)
            r2 = rand(dim)

            # velocity + position update (standard PSO equation)
            for j in 1:dim
                vel[i, j] =
                    w * vel[i, j] +
                    c1 * r1[j] * (pbest_pos[i, j] - pos[i, j]) +
                    c2 * r2[j] * (gbest_pos[j] - pos[i, j])

                # position update + enforce box constraints
                pos[i, j] += vel[i, j]
                pos[i, j] = clamp(pos[i, j], bounds[1, j], bounds[2, j])
            end

            # objective at new particle position
            fval = f(pos[i, :])

            # update personal best if improved
            if better(fval, pbest_val[i])
                pbest_val[i] = fval
                pbest_pos[i, :] = pos[i, :]
            end

            # update global best if improved
            if better(fval, gbest_val)
                gbest_val = fval
                gbest_pos = copy(pos[i, :])
            end
        end

        # store swarm state (for visualization / debugging)
        if track
            push!(history, copy(pos))
        end

        # stopping criterion (relative improvement measure)
        f_max = maximum(pbest_val)
        f_min = minimum(pbest_val)

        # uses spread of swarm objective values for normalization
        opt_range = max(f_max - f_min, 1e-12) # prevent division by zero (collapsed swarm case)

        # normalized convergence metric
        relative_gap = mode == :max ?
            (f_max - gbest_val) / opt_range :
            (gbest_val - f_min) / opt_range

        # stop if swarm has converged AND minimum exploration time passed (20%)
        if relative_gap < delta && iter > 0.2 * max_iter
            println("Converged at iteration ", iter)
            break
        end
    end

    return gbest_pos, gbest_val, history
end