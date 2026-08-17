# ==============================================================
#  ring_spreading.jl — point-source ring waves: cylindrical spreading + symmetry
#                      (gated version of examples/ring_wave.jl)
#
#  A Gaussian point source emits circular fronts. Two exact geometric properties
#  are checked quantitatively:
#    (1) RADIAL SYMMETRY — equal amplitude at equal radius along +x, +y and the
#        diagonal (isotropy of the 2-D operator);
#    (2) CYLINDRICAL SPREADING — far-field amplitude decays as 1/√r (energy flux
#        conservation on an expanding circle), i.e. amp·√r ≈ const.
#  The near field deviates from 1/√r; the fit is taken over the far-field gauges.
#
#  QUICK default (8 periods, 100×100). Production: L=200, 200×200, 20 periods.
#
#  RUN:  julia --project=. GridapLFEM.jl/examples/validation/ring_spreading.jl
# ==============================================================

using GridapLFEM
using Printf, LinearAlgebra

println("=" ^ 64)
println("  ring_spreading.jl — radial symmetry & 1/√r cylindrical spreading")
println("=" ^ 64)

g = 9.81; d = 3.5; T = 1.6; A = 0.001
L = 100.0; n_side = 100
x_wm, y_wm = L/2, L/2
r_gauges = [6.0, 10.0, 14.0, 18.0, 22.0]                     # radial line +x
gauges = [(x_wm + r, y_wm) for r in r_gauges]
push!(gauges, (x_wm, y_wm + 14.0))                          # +y at r=14
push!(gauges, (x_wm + 14/sqrt(2), y_wm + 14/sqrt(2)))       # diagonal at r=14

diags, _, _ = setup_and_run(
    M=2, c_bdy=[0.0,0.728,1.0], domain=((0.0,L),(0.0,L)), partition=(n_side,n_side),
    p_horizontal=2, h_val=d, T_wave=T, A_wave=A, x_wm=x_wm, y_wm=y_wm,
    sponge_wL=12.0, sponge_wR=12.0, sponge_wB=12.0, sponge_wT=12.0, mu_max=10.0,
    T_final=8*T, dt=T/30, save_every=0, gauges=gauges,
    regime=:linear, nl_pressure=:none, flat_bed=true,   # constant depth ⇒ ∇h≡0
    print_every=40)

steadyamp(i) = (gv = [d.gauge_vals[i] for d in diags]; n2 = length(gv)÷2;
                maximum(abs.(gv[n2:end])))
amps = [steadyamp(i) for i in eachindex(r_gauges)]
amp_x14 = amps[3]                                           # +x at r=14 (index 3)
amp_y14 = steadyamp(length(r_gauges)+1)
amp_d14 = steadyamp(length(r_gauges)+2)

println("\n   r [m]    amp [m]    amp·√r")
for (r, a) in zip(r_gauges, amps)
    @printf("  %5.1f   %.6f   %.6f\n", r, a, a*sqrt(r))
end
@printf("\n  symmetry at r=14:  +x %.6f   +y %.6f   diag %.6f\n", amp_x14, amp_y14, amp_d14)

# symmetry error (max spread across the three directions, normalised)
sym3 = [amp_x14, amp_y14, amp_d14]
sym_err = (maximum(sym3) - minimum(sym3)) / maximum(sym3)
# far-field 1/√r: amp·√r should be ~constant over the outer gauges
prod = amps .* sqrt.(r_gauges)
far  = prod[3:end]                                         # drop near field
spread = (maximum(far) - minimum(far)) / maximum(far)
@printf("  symmetry error = %.1f%%    far-field 1/√r spread = %.1f%%\n",
        100*sym_err, 100*spread)

n_fail = 0
(sym_err < 0.05) ? println("  PASS  radial symmetry (<5% across +x/+y/diag)") :
                   (println("  FAIL  anisotropic ($sym_err)"); global n_fail+=1)
(spread < 0.20) ? println("  PASS  far-field ≈ 1/√r (amp·√r const within 20%)") :
                  (println("  FAIL  far-field decay off ($spread)"); global n_fail+=1)
println("=" ^ 64)
n_fail == 0 ? println("  Ring waves are isotropic and spread cylindrically.") :
              println("  (enlarge domain / move gauges to the far field)")
