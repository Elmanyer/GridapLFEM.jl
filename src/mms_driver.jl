# ==============================================================
#  mms_driver.jl — run the analytic MMS and measure convergence rates
#
#  Kept separate from `setup_and_run` on purpose: the MMS case is a closed basin
#  with the wavemaker, sponge and relaxation zone all OFF and a forcing nothing
#  else uses, so folding it into the production driver would add MMS-only kwargs
#  to a signature that is already large. Everything below is assembled from the
#  same public building blocks a normal run uses (build_horizontal_model,
#  build_fe_spaces, build_problem, build_ode_operator, build_ode_solver,
#  run_time_loop), so the operator under test IS the production operator.
#
#  Plan: building_files/MMS_ANALYTIC_PLAN.md
# ==============================================================

"""
    run_mms_case(; nx, ny, dt, T_final, ...) → NamedTuple

One analytic-MMS solve. Returns `(h, dt, e_eta, e_u, rel_eta, rel_u, ndofs, steps)`.

Configuration is fixed to Stage 1 and deliberately not user-overridable where the
derivation depends on it:
  * `regime=:linear`, `flat_bed=true`, constant depth `d` — the strong form the
    closed-form forcing was derived from;
  * `x_wall_bc=true`, `y_wall_bc=:wall` — a CLOSED BASIN, which is what makes all
    three IBP boundary integrals vanish so no boundary forcing is needed;
  * `mu_sponge ≡ 0`, `wm_src ≡ 0`, no relaxation zone — a pure interior-operator test.

The initial condition is `u*(t0)` interpolated into the trial space; `u*` satisfies
the wall Dirichlet data exactly, so the interpolation is consistent.
"""
function run_mms_case(; nx::Int, ny::Int, dt::Float64, T_final::Float64,
                        Lx::Float64 = 1.0, Ly::Float64 = 1.0,
                        d::Float64 = 1.0, g::Float64 = g,
                        M::Int = 2, p_vert::Int = 1,
                        c_bdy = nothing,           # σ-element boundaries; nothing ⇒ resolve_cbdy(M)
                        p_horizontal::Int = 2,
                        p_eta::Int = 0,            # 0 ⇒ equal order. Set < p_horizontal for a
                                                   #   Taylor-Hood-like pairing (see build_fe_spaces).
                        field = nothing,
                        hfun = nothing,            # bathymetry h(x,y); nothing ⇒ constant `d`
                        flat_bed::Bool = true,     # selects BOTH the solver model and the forcing
                        regime::Symbol = :linear,      # ) these three select BOTH sides too —
                        nl_pressure::Symbol = :none,   # ) see ValidationTests.tex §subsec: mms model1–4
                        vert_override = nothing,   # substitute vertical tensors. Used to ISOLATE a
                                                   #   term: e.g. `merge(NamedTuple(pairs(v)),
                                                   #   (B=zeros(size(v.B)),))` drops R_P from the
                                                   #   solver AND the forcing consistently, since
                                                   #   both read B from this same object. That is
                                                   #   the experiment that attributes the measured
                                                   #   order loss (see MMS_ANALYTIC_PLAN.md §result).
                        solver_type::Symbol = :sdirk, tableau::Symbol = :SDIRK_2_2,
                        theta::Float64 = 0.5,
                        nl_tol::Float64 = 1e-12, nl_iter::Int = 50,
                        t0::Float64 = 0.0,
                        verbose::Bool = true,
                        use_ad::Bool = false,   # AD Jacobians (3-arg TransientFEOperator)
                        output_dir::String = mktempdir())
    vert = vert_override === nothing ?
           assemble_vertical_tensors(M, p_vert, resolve_cbdy(M, c_bdy)) : vert_override
    f    = field === nothing ?
           MMSField(vert.N_dof; Lx=Lx, Ly=Ly) : field

    # --- mesh + closed-basin FE spaces ------------------------------------
    domain       = ((0.0, Lx), (0.0, Ly))
    model, trian = build_horizontal_model(domain, (nx, ny))
    pe           = p_eta == 0 ? p_horizontal : p_eta
    U, V         = build_fe_spaces(model, p_horizontal, vert.N_dof;
                                   y_wall_bc = :wall, x_wall_bc = true, p_eta = pe)
    dΩh          = Measure(trian, 2*max(p_horizontal, pe) + 2)

    # --- the forcing, derived independently of the residual code ----------
    #  ONE bathymetry object feeds BOTH the forcing and the solver, and the SAME
    #  regime/flat_bed symbols select both, so the two cannot describe different
    #  models. mms_forcing additionally REJECTS flat_bed=true over a varying bed.
    hf   = hfun === nothing ? ((xx, yy) -> d) : hfun
    src  = mms_forcing(f, vert, hf, g; regime = regime, flat_bed = flat_bed,
                                       nl_pressure = nl_pressure)

    prob = build_problem(vert; g = g,
                         h_bathy     = (x -> hf(x[1], x[2])),
                         regime      = regime,       # SAME variables as the forcing above —
                         nl_pressure = nl_pressure,  # never two literals, or the two can drift
                         flat_bed    = flat_bed,
                         mu_sponge   = (x -> 0.0),    # sponge OFF
                         wm_src      = ((x, t) -> 0.0),  # wavemaker OFF
                         mms_src     = src)

    op     = use_ad ? build_ode_operator_ad(prob, U, V, trian, dΩh) :
                      build_ode_operator(prob, U, V, trian, dΩh)
    solver = build_ode_solver(dt; solver_type=solver_type, tableau=tableau,
                              theta=theta, nl_iter=nl_iter, nl_tol=nl_tol)

    # --- IC = u*(t0), which satisfies the wall Dirichlet data exactly ------
    u0 = interpolate_everywhere([mms_exact_eta(f, t0),
                                 mms_exact_ux(f, t0),
                                 mms_exact_uy(f, t0)], U)

    final = Ref{Any}(nothing)
    diags = run_time_loop(op, solver, u0, t0, T_final;
                          output_dir=output_dir, save_every=0,
                          trian=trian, Nσ=vert.N_dof,
                          # print_every must be > 0: run_time_loop does `step % print_every`,
                          # so 0 raises DivideError. typemax never matches ⇒ silent run.
                          print_every=typemax(Int), dt=dt, final_uh=final,
                          diag_every=-1, check_every=0)
    final[] === nothing && error("run_mms_case: the time loop produced no solution")

    # --- L² errors against the EXACT field, on an elevated quadrature ------
    uh   = final[]
    tF   = isempty(diags) ? t0 : diags[end].t
    dΩe  = error_measure(trian, p_horizontal)
    e_eta = l2_error(uh[1], mms_exact_eta(f, tF), trian, dΩe)
    e_ux  = l2_error(uh[2], mms_exact_ux(f, tF),  trian, dΩe)
    e_uy  = l2_error(uh[3], mms_exact_uy(f, tF),  trian, dΩe)
    e_u   = sqrt(e_ux^2 + e_uy^2)
    n_eta = l2_norm_exact(mms_exact_eta(f, tF), trian, dΩe)
    n_u   = sqrt(l2_norm_exact(mms_exact_ux(f, tF), trian, dΩe)^2 +
                 l2_norm_exact(mms_exact_uy(f, tF), trian, dΩe)^2)

    h = max(Lx/nx, Ly/ny)
    if verbose
        @printf("  nx=%-4d ny=%-4d h=%-9.5g dt=%-9.4g  t=%.4f  e_eta=%.6e  e_u=%.6e\n",
                nx, ny, h, dt, tF, e_eta, e_u)
    end
    return (h=h, dt=dt, e_eta=e_eta, e_u=e_u,
            rel_eta = e_eta/n_eta, rel_u = e_u/n_u,
            ndofs = num_free_dofs(U), steps = length(diags), t_final = tF)
end

"""
    run_mms_refinement(mode; levels, ...) → NamedTuple

`mode = :space` — halve `h` at fixed `dt`; expected slope **3** for `Q2` (`p+1`).
`mode = :time`  — halve `dt` at fixed fine mesh; expected slope **2** (SDIRK_2_2 / CN).

Returns the sequences and the fitted slopes, and prints the refinement table.

Isolating the two rates is the caller's responsibility and is the classic way this
measurement goes wrong: for `:space` the time step must be small enough that the
`O(Δt²)` contribution sits below the SMALLEST spatial error in the sequence, or the
finest level saturates and drags the fitted slope down. `mms_dt_independence`
below checks that directly.
"""
function run_mms_refinement(mode::Symbol; levels::Int = 3,
                            nx0::Int = 8, ny0::Int = 8, dt0::Float64 = 1e-3,
                            T_final::Float64 = 0.05,
                            nx_fine::Int = 32, ny_fine::Int = 32,
                            kwargs...)
    mode in (:space, :time) ||
        error("run_mms_refinement: mode must be :space or :time (got :$mode)")
    hs = Float64[]; ee = Float64[]; eu = Float64[]; rows = []
    for l in 0:levels-1
        r = mode == :space ?
            run_mms_case(; nx = nx0*2^l, ny = ny0*2^l, dt = dt0,
                           T_final = T_final, kwargs...) :
            run_mms_case(; nx = nx_fine, ny = ny_fine, dt = dt0/2^l,
                           T_final = T_final, kwargs...)
        push!(hs, mode == :space ? r.h : r.dt)
        push!(ee, r.e_eta); push!(eu, r.e_u); push!(rows, r)
    end
    #  Expectations follow the FE PAIRING actually used, and the two fields differ
    #  whenever p_eta < p_horizontal. Hardcoding "3" here (as this did) reports an
    #  unreachable target on an equal-order run and the wrong target on a mixed one.
    p_h  = get(kwargs, :p_horizontal, 2)
    p_e0 = get(kwargs, :p_eta, 0)
    p_e  = p_e0 == 0 ? p_h : p_e0
    exp_eta, exp_u = mode == :space ? (p_e + 1.0, p_h + 1.0) : (2.0, 2.0)
    label = mode == :space ?
        "SPATIAL refinement (Q$(p_h)/Q$(p_e) ⇒ eta $(p_e+1), u $(p_h+1))" *
        (p_e == p_h ? "  [EQUAL ORDER: inf-sup deficient, expect p not p+1]" : "") :
        "TEMPORAL refinement (2nd-order scheme)"
    p_eta, p_u = refinement_table(label, hs, ee, eu;
                                  expected=exp_eta, expected_u=exp_u)
    return (param=hs, e_eta=ee, e_u=eu, p_eta=p_eta, p_u=p_u,
            expected=exp_eta, expected_u=exp_u, rows=rows)
end

"""
    run_mms_case_distributed(; nx, ny, cpu_grid, ...) → NamedTuple

Distributed twin of `run_mms_case`. MUST be called from an `mpiexecjl -n px*py` context.

The forcing needs no distributed-specific code: `mms_src` flows through the *shared*
`global_residual`, whose `CellField(x -> S(x,t), trian)` pattern is exactly what the relaxation-zone
term already does with a `DistributedTriangulation`. The L² error likewise uses the distributed
`Measure`/`∫`/`sum`, which reduce across ranks.

**Tolerances matter here in a way they do not sequentially.** The sequential path factorises directly
(`LUSolver`), so nothing caps the error; the distributed path is GMRES+Jacobi, and a loose `ls_rtol`
would cap `‖e‖` and manufacture a fake rate. Defaults are therefore `ls_rtol=1e-13`, `nl_tol=1e-13`,
and the campaign verifies non-capping explicitly (Gate T).

⚠ **This function used to hard-code Model 1** — `regime=:linear, nl_pressure=:none, flat_bed=true`
as LITERALS, plus the Model-1 closed form `mms_forcing_stage1` — while accepting no model switches
at all. `run_conv_study` still built its printed tag from the *requested* switches, so a distributed
campaign over eight models returned **eight identical Model-1 studies under eight different labels,
all passing**. Fixed 2026-08-21: the switches are now parameters, the forcing comes from the general
`mms_forcing` (which itself re-selects the Model-1 closed form when the switches ask for it), and the
SAME three variables feed the forcing and the solver — never two literals. `test_mms_distributed_
parity.jl` gates the two branches against each other on a NON-Model-1 configuration so the defect
cannot recur silently.
"""
function run_mms_case_distributed(; nx::Int, ny::Int, dt::Float64, T_final::Float64,
                                    cpu_grid::Tuple{Int,Int} = (2,2),
                                    Lx::Float64 = 1.7, Ly::Float64 = 1.1,
                                    d::Float64 = 1.0, g::Float64 = g,
                                    M::Int = 2, p_vert::Int = 1,
                                    c_bdy = nothing,  # σ-element boundaries; nothing ⇒ resolve_cbdy(M)
                                    p_horizontal::Int = 2, p_eta::Int = 0,
                                    field = nothing, t0::Float64 = 0.0,
                                    hfun = nothing,            # bathymetry h(x,y); nothing ⇒ constant `d`
                                    flat_bed::Bool = true,     # ) the same three symbols select BOTH
                                    regime::Symbol = :linear,  # ) the forcing and the solver, exactly
                                    nl_pressure::Symbol = :none, # ) as in run_mms_case above
                                    vert_override = nothing,   # substitute vertical tensors (see run_mms_case)
                                    solver_type::Symbol = :sdirk, tableau::Symbol = :SDIRK_2_2,
                                    nl_tol::Float64 = 1e-13, nl_iter::Int = 50,
                                    ls_rtol::Float64 = 1e-13, ls_maxiter::Int = 5000,
                                    krylov_m::Int = 200, verbose::Bool = true,
                                    output_dir::String = mktempdir())
    vert = vert_override === nothing ?
           assemble_vertical_tensors(M, p_vert, resolve_cbdy(M, c_bdy)) : vert_override
    f    = field === nothing ? MMSField(vert.N_dof; Lx=Lx, Ly=Ly) : field
    pe   = p_eta == 0 ? p_horizontal : p_eta
    #  ONE bathymetry object and ONE set of switches for the forcing and the solver,
    #  built OUTSIDE the MPI block so every rank derives them from identical inputs.
    hf   = hfun === nothing ? ((xx, yy) -> d) : hfun
    src  = mms_forcing(f, vert, hf, g; regime = regime, flat_bed = flat_bed,
                                       nl_pressure = nl_pressure)
    with_mpi() do distribute
        ranks = distribute(LinearIndices((prod(cpu_grid),)))
        model, trian = build_horizontal_model_distributed(ranks, cpu_grid,
                            (0.0, Lx, 0.0, Ly), (nx, ny))
        U, V = build_fe_spaces(model, p_horizontal, vert.N_dof;
                               y_wall_bc=:wall, x_wall_bc=true, p_eta=pe)
        dΩh  = Measure(trian, 2*max(p_horizontal, pe) + 2)
        prob = build_problem(vert; g=g,
                             h_bathy     = (x -> hf(x[1], x[2])),
                             regime      = regime,       # SAME variables as `src` above —
                             nl_pressure = nl_pressure,  # never two literals, or the two can drift
                             flat_bed    = flat_bed,
                             mu_sponge=(x -> 0.0), wm_src=((x,t) -> 0.0),
                             mms_src=src)
        op     = build_ode_operator(prob, U, V, trian, dΩh)
        solver = build_ode_solver_distributed(dt; solver_type=solver_type, tableau=tableau,
                        nl_iter=nl_iter, nl_tol=nl_tol, ls_rtol=ls_rtol,
                        ls_maxiter=ls_maxiter, krylov_m=krylov_m)
        u0 = interpolate_everywhere([mms_exact_eta(f, t0),
                                     mms_exact_ux(f, t0),
                                     mms_exact_uy(f, t0)], U)
        final = Ref{Any}(nothing)
        diags = run_time_loop_dist(ranks, op, solver, u0, t0, T_final;
                                   output_dir=output_dir, save_every=0, trian=trian,
                                   Nσ=vert.N_dof, print_every=typemax(Int), dt=dt,
                                   final_uh=final, diag_every=-1, check_every=0)
        uh  = final[]
        tF  = isempty(diags) ? t0 : diags[end].t
        dΩe = error_measure(trian, max(p_horizontal, pe))
        e_eta = l2_error(uh[1], mms_exact_eta(f, tF), trian, dΩe)
        e_u   = sqrt(l2_error(uh[2], mms_exact_ux(f, tF), trian, dΩe)^2 +
                     l2_error(uh[3], mms_exact_uy(f, tF), trian, dΩe)^2)
        verbose && i_am_main(ranks) &&
            @printf("  [dist] nx=%-4d %s/%s/%s  e_eta=%.6e  e_u=%.6e\n",
                    nx, regime, flat_bed ? "flat" : "varbed", nl_pressure, e_eta, e_u)
        return (h=Lx/nx, dt=dt, e_eta=e_eta, e_u=e_u,
                ndofs=num_free_dofs(U), steps=length(diags), t_final=tF)
    end
end

"""
    run_conv_study(; pairing, domain, levels, mode, ...) → NamedTuple

ONE convergence study of the campaign in `building_files/MMS_CONVERGENCE_CAMPAIGN.md`:
a mixed-order pairing `Q_p` velocity / `Q_{p-1}` surface, refined over `levels` meshes,
reported against the OPTIMAL rates for that pairing.

**The two fields have different optima and the gates differ accordingly:**
  * `u`  in `Q_p`      ⇒ optimal `L²` rate `p+1`
  * `η`  in `Q_{p-1}`  ⇒ optimal `L²` rate `p`      (NOT `p+1` — do not read the gap as a defect)

`domain = :d1` builds the quasi-1D case: `k_y = 0` ⇒ `u*ʸ ≡ 0`, y-invariant solution, refinement in
`nx` only with `ny` held fixed. `domain = :d2` refines both.

`mode = :static` uses `ω = 0`, for which `∂ₜu* ≡ 0` and the temporal discretisation error is
*identically* zero — the cleanest possible isolation of the spatial operator. `mode = :transient`
integrates `nsteps` at a small `dt`.

Sequential runs need no tolerance study: `LUSolver` is direct (no linear tolerance at all) and the
linear regime makes the residual linear in `u`, so Newton hits round-off in one step.

**The VERTICAL basis is a parameter, `(M, p_vert, c_bdy)`.** It used to be hard-wired
(`assemble_vertical_tensors(M, 1, [0.0, 0.728, 1.0])`): `p_vert` was not a parameter at all and
`c_bdy` was pinned to the **M=2** node set, so `run_conv_study(M=3)` threw the
`length(c_bdy) == M+1` assertion in `assemble_vertical_tensors` — while the signature advertised
`M` as a degree of freedom. Fixed 2026-08-21; `c_bdy === nothing` now resolves through
[`resolve_cbdy`](@ref). This is what makes the vertical-basis convergence study
(`building_files/PENDING_TASKS.md` §1) runnable.

⚠ **`c_bdy` changes the error CONSTANT, never the ORDER.** A rate study may use any reasonable node
set (uniform is fine for `p ≥ 2`, where no published optimum exists); the optimised positions are a
DISPERSION-accuracy question, measured by applicable `kd`, not by a rate. Do not report one as
evidence for the other.

The returned `tag` carries the vertical configuration (`P{p_vert}LFE-{M}`) as well as the pairing
and the model, so a sweep over `(M, p)` cannot produce two studies under the same label.
"""
function run_conv_study(; p_u::Int, domain::Symbol = :d2, mode::Symbol = :static,
                          levels::Int = 4, nx0::Int = 8, ny0::Int = 8, ny_1d::Int = 3,
                          Lx::Float64 = 1.7, Ly::Float64 = 1.1, d::Float64 = 1.0,
                          M::Int = 2, p_vert::Int = 1, c_bdy = nothing,
                          dt::Float64 = 1e-4, nsteps::Int = 100,
                          nl_tol::Float64 = 1e-14,
                          #  The NONLINEAR Jacobians are quasi-Newton by design (the
                          #  pressure blocks' η-dependence is frozen — see problem.jl),
                          #  so Newton converges LINEARLY there and needs far more than
                          #  the 50 iterations a linear model uses. Raise this, don't
                          #  loosen nl_tol, or the algebraic error contaminates the rate.
                          nl_iter::Int = 50,
                          distributed::Bool = false,
                          flat_bed::Bool = true, a_b::Float64 = 0.0,
                          regime::Symbol = :linear, nl_pressure::Symbol = :none,
                          kbx::Float64 = 1.3, kby::Float64 = 0.0,
                          cpu_grid::Tuple{Int,Int} = (2,2),
                          ls_rtol::Float64 = 1e-13, ls_maxiter::Int = 5000,
                          verbose::Bool = true)
    domain in (:d1, :d2) || error("run_conv_study: domain must be :d1 or :d2")
    mode   in (:static, :transient) || error("run_conv_study: mode must be :static or :transient")
    p_e   = p_u - 1
    p_e ≥ 1 || error("run_conv_study: p_u must be ≥ 2 (surface order p_u−1 ≥ 1)")
    ω     = mode == :static ? 0.0 : 1.3
    T_fin = mode == :static ? dt*nsteps : dt*nsteps
    #  The vertical basis is a PARAMETER (see the docstring): (M, p_vert, c_bdy),
    #  resolved once here so every refinement level shares one tensor set and the
    #  rate cannot be contaminated by a changing vertical discretisation.
    cb    = resolve_cbdy(M, c_bdy)
    vert  = assemble_vertical_tensors(M, p_vert, cb)
    #  ONE bathymetry object for both branches — the sequential path used to build
    #  this inline and the distributed path had none at all (it hard-coded a flat bed).
    hfun  = flat_bed ? nothing : bathymetry_field(; d0=d, a_b=a_b, kbx=kbx, kby=kby)

    hs = Float64[]; ee = Float64[]; eu = Float64[]; nd = Int[]
    for l in 0:levels-1
        nx = nx0*2^l
        ny = domain == :d1 ? ny_1d : ny0*2^l
        f  = MMSField(vert.N_dof; Lx=Lx, Ly=Ly, omega=ω,
                      ky = domain == :d1 ? 0.0 : nothing)
        #  The two branches take the SAME vertical basis, the SAME bathymetry and the
        #  SAME three model switches. Every one of them was previously either absent
        #  from the distributed call or a literal inside it — the defect that made a
        #  distributed 8-model campaign return 8 copies of Model 1 (see A2 above).
        common = (; nx=nx, ny=ny, dt=dt, T_final=T_fin, Lx=Lx, Ly=Ly, d=d,
                    M=M, p_vert=p_vert, c_bdy=cb, vert_override=vert,
                    p_horizontal=p_u, p_eta=p_e, field=f,
                    regime=regime, nl_pressure=nl_pressure, flat_bed=flat_bed,
                    hfun=hfun, nl_tol=nl_tol, nl_iter=nl_iter, verbose=false)
        r = distributed ?
            run_mms_case_distributed(; common..., cpu_grid=cpu_grid,
                                       ls_rtol=ls_rtol, ls_maxiter=ls_maxiter) :
            run_mms_case(; common...)
        push!(hs, Lx/nx); push!(ee, r.e_eta); push!(eu, r.e_u); push!(nd, r.ndofs)
        verbose && @printf("    nx=%-4d ndofs=%-8d e_eta=%.6e  e_u=%.6e\n",
                           nx, r.ndofs, r.e_eta, r.e_u)
    end
    p_eta_fit, pw_eta = convergence_rate(hs, ee)
    p_u_fit,   pw_u   = convergence_rate(hs, eu)
    #  The vertical configuration is PART OF THE LABEL. Without it an (M,p) sweep
    #  reports every study under the same tag — which is precisely how the
    #  hard-coded-Model-1 defect stayed invisible for a whole campaign.
    tag = "P$(p_vert)LFE-$(M) Q$(p_u)/Q$(p_e) $(domain == :d1 ? "1D" : "2D") " *
          "$(distributed ? "dist" : "seq") $(mode) $(flat_bed ? "flat" : "varbed") " *
          "$(regime === :linear ? "lin" : "nl")" *
          "$(nl_pressure === :none ? "" : "/" * String(nl_pressure))"
    if verbose
        println("  ── $tag ──")
        println("    pairwise eta: ", round.(pw_eta, digits=3), "   optimal $(p_e+1)")
        println("    pairwise u  : ", round.(pw_u,   digits=3), "   optimal $(p_u+1)")
        @printf("    fitted: p_eta=%.3f (opt %d)   p_u=%.3f (opt %d)   %s\n",
                p_eta_fit, p_e+1, p_u_fit, p_u+1,
                (abs(pw_eta[end]-(p_e+1))<0.3 && abs(pw_u[end]-(p_u+1))<0.3) ? "PASS" : "CHECK")
    end
    return (tag=tag, p_u=p_u, p_e=p_e, h=hs, e_eta=ee, e_u=eu, ndofs=nd,
            pw_eta=pw_eta, pw_u=pw_u, fit_eta=p_eta_fit, fit_u=p_u_fit,
            opt_eta=Float64(p_e+1), opt_u=Float64(p_u+1),
            #  Vertical configuration, carried out so an (M,p) sweep can tabulate
            #  against Nσ directly (PENDING_TASKS.md §1 tier 3 asks exactly that).
            M=M, p_vert=p_vert, c_bdy=cb, Nsigma=vert.N_dof,
            regime=regime, nl_pressure=nl_pressure, flat_bed=flat_bed,
            domain=domain, mode=mode, distributed=distributed)
end

"""
    run_model_case(; nx, ny, dt, T_final, n_mode, ...) → NamedTuple

MODEL validation, as distinct from code verification: **no forcing at all**. The
initial condition is the exact standing mode of `standing_mode`, which is an exact
unforced solution of the linear flat-bed system satisfying the solid walls, so the
discrete solution must converge to a KNOWN ANALYTIC WAVE.

What this adds over the MMS: the MMS certifies that the discretised operator is the
intended one, but it forces the solution and so never exercises the model's own
free evolution. Here the dynamics are the model's — in particular the phase is set
by the model dispersion relation `ω = k·Cm(k)`, so a wrong dispersion shows up as a
phase drift that does NOT converge away under refinement.

Returns the same fields as `run_mms_case`, so both feed `refinement_table`.
"""
function run_model_case(; nx::Int, ny::Int, dt::Float64, T_final::Float64,
                          Lx::Float64 = 1.7, Ly::Float64 = 1.1,
                          d::Float64 = 1.0, g::Float64 = g,
                          M::Int = 2, p_vert::Int = 1,
                          c_bdy = nothing,         # σ-element boundaries; nothing ⇒ resolve_cbdy(M)
                          p_horizontal::Int = 2, n_mode::Int = 1,
                          p_eta::Int = 0,            # 0 ⇒ equal order. Set < p_horizontal for the
                                                     #   Taylor-Hood-like pairing — same meaning and
                                                     #   same default as run_mms_case, so the forced
                                                     #   and unforced studies stay comparable.
                          eta_hat::Float64 = 1.0,
                          solver_type::Symbol = :sdirk, tableau::Symbol = :SDIRK_2_2,
                          theta::Float64 = 0.5,
                          nl_tol::Float64 = 1e-12, nl_iter::Int = 50,
                          t0::Float64 = 0.0, verbose::Bool = true,
                          output_dir::String = mktempdir())
    vert = assemble_vertical_tensors(M, p_vert, resolve_cbdy(M, c_bdy))
    cbs, ω, û, k = standing_mode(vert, d, g; n=n_mode, Lx=Lx, eta_hat=eta_hat)

    domain       = ((0.0, Lx), (0.0, Ly))
    model, trian = build_horizontal_model(domain, (nx, ny))
    pe           = p_eta == 0 ? p_horizontal : p_eta
    U, V         = build_fe_spaces(model, p_horizontal, vert.N_dof;
                                   y_wall_bc = :wall, x_wall_bc = true, p_eta = pe)
    dΩh          = Measure(trian, 2*max(p_horizontal, pe) + 2)

    prob = build_problem(vert; g = g, h_bathy = (x -> d),
                         regime = :linear, nl_pressure = :none, flat_bed = true,
                         mu_sponge = (x -> 0.0), wm_src = ((x, t) -> 0.0),
                         mms_src = nothing)          # NO FORCING — that is the point
    op     = build_ode_operator(prob, U, V, trian, dΩh)
    solver = build_ode_solver(dt; solver_type=solver_type, tableau=tableau,
                              theta=theta, nl_iter=nl_iter, nl_tol=nl_tol)

    e0, ux0, uy0 = exact_cfs(cbs, vert.N_dof, t0)
    u0 = interpolate_everywhere([e0, ux0, uy0], U)

    final = Ref{Any}(nothing)
    diags = run_time_loop(op, solver, u0, t0, T_final;
                          output_dir=output_dir, save_every=0, trian=trian,
                          Nσ=vert.N_dof, print_every=typemax(Int), dt=dt,
                          final_uh=final, diag_every=-1, check_every=0)
    final[] === nothing && error("run_model_case: the time loop produced no solution")

    uh  = final[]
    tF  = isempty(diags) ? t0 : diags[end].t
    dΩe = error_measure(trian, p_horizontal)
    eF, uxF, uyF = exact_cfs(cbs, vert.N_dof, tF)
    e_eta = l2_error(uh[1], eF,  trian, dΩe)
    e_u   = sqrt(l2_error(uh[2], uxF, trian, dΩe)^2 +
                 l2_error(uh[3], uyF, trian, dΩe)^2)
    n_eta = l2_norm_exact(eF, trian, dΩe)

    h = max(Lx/nx, Ly/ny)
    if verbose
        @printf("  nx=%-4d h=%-9.5g dt=%-9.4g  t=%.4f  T=%.4f  e_eta=%.6e  e_u=%.6e\n",
                nx, h, dt, tF, 2pi/ω, e_eta, e_u)
    end
    return (h=h, dt=dt, e_eta=e_eta, e_u=e_u, rel_eta=e_eta/n_eta,
            omega=ω, period=2pi/ω, k=k, celerity=ω/k,
            ndofs=num_free_dofs(U), steps=length(diags), t_final=tF)
end

"""
    run_model_refinement(mode; levels, ...) → NamedTuple

Refinement study for `run_model_case` (unforced, exact standing mode). Same
signature and expected slopes as `run_mms_refinement`.
"""
function run_model_refinement(mode::Symbol; levels::Int = 3,
                              nx0::Int = 8, ny0::Int = 4, dt0::Float64 = 1e-3,
                              T_final::Float64 = 0.05,
                              nx_fine::Int = 32, ny_fine::Int = 8, kwargs...)
    mode in (:space, :time) ||
        error("run_model_refinement: mode must be :space or :time (got :$mode)")
    hs = Float64[]; ee = Float64[]; eu = Float64[]
    for l in 0:levels-1
        r = mode == :space ?
            run_model_case(; nx=nx0*2^l, ny=ny0*2^l, dt=dt0, T_final=T_final, kwargs...) :
            run_model_case(; nx=nx_fine, ny=ny_fine, dt=dt0/2^l, T_final=T_final, kwargs...)
        push!(hs, mode == :space ? r.h : r.dt)
        push!(ee, r.e_eta); push!(eu, r.e_u)
    end
    #  As in run_mms_refinement: the expectations are a property of the FE pairing,
    #  not a constant. See the note there.
    p_h  = get(kwargs, :p_horizontal, 2)
    p_e0 = get(kwargs, :p_eta, 0)
    p_e  = p_e0 == 0 ? p_h : p_e0
    exp_eta, exp_u = mode == :space ? (p_e + 1.0, p_h + 1.0) : (2.0, 2.0)
    label = mode == :space ?
        "MODEL standing mode — SPATIAL (Q$(p_h)/Q$(p_e) ⇒ eta $(p_e+1), u $(p_h+1))" *
        (p_e == p_h ? "  [EQUAL ORDER: inf-sup deficient, expect p not p+1]" : "") :
        "MODEL standing mode — TEMPORAL (2nd order)"
    p_eta, p_u = refinement_table(label, hs, ee, eu;
                                  expected=exp_eta, expected_u=exp_u)
    return (param=hs, e_eta=ee, e_u=eu, p_eta=p_eta, p_u=p_u,
            expected=exp_eta, expected_u=exp_u)
end

"""
    mms_dt_independence(; nx, ny, dt, T_final, kwargs...) → (e1, e2, rel_change)

Run the finest spatial level at `dt` and at `dt/2`. If the spatial error is truly
resolved, halving the time step must barely move it. A large change means the
spatial study is measuring the temporal error and its slope is meaningless.
"""
function mms_dt_independence(; nx::Int, ny::Int, dt::Float64, T_final::Float64, kwargs...)
    a = run_mms_case(; nx=nx, ny=ny, dt=dt,     T_final=T_final, verbose=false, kwargs...)
    b = run_mms_case(; nx=nx, ny=ny, dt=dt/2,   T_final=T_final, verbose=false, kwargs...)
    rel = abs(b.e_eta - a.e_eta)/a.e_eta
    @printf("  dt-independence at nx=%d: e(dt)=%.6e  e(dt/2)=%.6e  Δ=%.2f %%\n",
            nx, a.e_eta, b.e_eta, 100rel)
    return a.e_eta, b.e_eta, rel
end
