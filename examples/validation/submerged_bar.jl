# ==============================================================
#  submerged_bar.jl — regular waves over a submerged bar / shoal
#                     (Yang & Liu 2024 §4: submerged-shoal case;
#                      experimentally the Beji–Battjes / Dingemans bar)
#
#  The canonical harmonic-generation benchmark. A regular wave shoals up the
#  front slope of a smooth submerged bar; nonlinear interaction on the crest
#  pumps energy into higher harmonics; on the down-slope these are RELEASED as
#  free (dispersively separating) waves. The signature reproduced by all
#  Boussinesq / non-hydrostatic models is the growth of the 2nd (and 3rd)
#  harmonic over the bar and their persistence downstream. The variable bed
#  activates the slope-pressure package (∇h in the linear and nonlinear
#  pressure), which is why an analytic (tanh) bar is used — exact ∇h, ∇²h.
#
#  We place a line of gauges up-slope / crest / down-slope, DFT each, and print
#  the harmonic-amplitude evolution H₁(x),H₂(x),H₃(x). QUICK default; production
#  = finer mesh, more periods, and comparison to digitised experimental curves.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/validation/submerged_bar.jl
# ==============================================================

if !isdefined(Main, :GridapLFEM)
    include(joinpath(@__DIR__, "..", "..", "src", "GridapLFEM.jl"))
end
using .GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 64)
println("  submerged_bar.jl — harmonic generation over a submerged bar")
println("=" ^ 64)

g = 9.81; d0 = 3.5; hbar = 2.0; T = 2.5; A = 0.001
omega = 2π/T; k0 = find_wavenumber(omega, d0, g); lam0 = 2π/k0
xbar = 120.0; wbar = 25.0; sramp = wbar/3
h_bathy(x) = d0 - 0.5*hbar*(tanh((x[1]-(xbar-wbar))/sramp) -
                            tanh((x[1]-(xbar+wbar))/sramp))    # smooth bar, exact ∇h
@printf("  offshore d=%.1f m, bar crest depth=%.1f m, T=%.1f s, λ₀=%.1f m\n",
        d0, d0-hbar, T, lam0)

x_wm = 40.0; Lx = 250.0; Ly = 2.0
nx = round(Int, Lx/(lam0/12)); ny = 2
dt = 0.02; T_final = 30*T
x_gauges = [80.0, 110.0, xbar, 150.0, 190.0]       # up-slope, front, crest, back, down
gauges = [(xg, Ly/2) for xg in x_gauges]
@printf("  domain %.0f×%.0f, %d×%d cells, %d gauges\n", Lx,Ly,nx,ny,length(gauges))

diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,Lx),(0.0,Ly)), partition=(nx,ny),
    p_horizontal=2, h_val=d0, T_wave=T, A_wave=A, x_wm=x_wm, y_wm=nothing,
    sponge_wL=35.0, sponge_wR=35.0, mu_max=5.0, T_final=T_final, dt=dt,
    h_bathy=h_bathy, save_every=0, gauges=gauges,
    regime=:nonlinear, nl_pressure=:full, flat_bed=false,   # variable bathymetry (the bar): ∇h terms ON
    print_every=200)

ts = [d.t for d in diags]; i0 = length(ts) ÷ 2; tv = ts[i0:end]
amp(gv, w) = 2*abs(sum(gv .* exp.(-im .* w .* tv))) / length(gv)
println("\n   x [m]    depth    H₁ [m]     H₂ [m]     H₃ [m]    H₂/H₁")
for (i, xg) in enumerate(x_gauges)
    gv = [d.gauge_vals[i] for d in diags][i0:end]
    H1 = amp(gv, omega); H2 = amp(gv, 2omega); H3 = amp(gv, 3omega)
    @printf("  %6.1f   %5.2f   %.3e  %.3e  %.3e   %.3f\n",
            xg, h_bathy((xg,0.0)), H1, H2, H3, H2/H1)
end
println("\n  Expected: H₂ grows toward/over the crest and persists down-slope")
println("  (released free harmonic). Compare H₁,₂,₃(x) to the reference/experiment.")
println("=" ^ 64)
