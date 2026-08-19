# ==============================================================
#  test_mms_forcing_nonlinear.jl — forcing-level gates for the NONLINEAR
#  analytic-MMS models (no FE solve — seconds, not minutes)
#
#  Specification: ValidationTests.tex §subsec: mms model3 / §subsec: mms model4.
#  Plan:          building_files/MMS_NONLINEAR_PLAN.md §2.1.
#
#  WHY GATE THE FORCING SEPARATELY FROM THE RATE.
#  A convergence study conflates two things: a wrong FORCING and a wrong SOLVER
#  both show up as a collapsed rate, and a study costs tens of minutes. These
#  gates check the forcing alone, in seconds, by bracketing each new model
#  between two ALREADY-VERIFIED ones:
#
#      Model 4 at constant h   ==  Model 3          (flat-bed limit)
#      Model 3 as amplitude→0  ->  Model 1 + O(ε²)  (small-amplitude limit)
#
#  A wrong term must break one of those two, whichever model it lives in.
#
#  RUN:  julia --project=. test/test_mms_forcing_nonlinear.jl
# ==============================================================

using GridapLFEM
using Printf

println("=" ^ 76)
println("  test_mms_forcing_nonlinear.jl — nonlinear MMS forcing gates")
println("=" ^ 76)

const G  = GridapLFEM.g
const D0 = 2.5      # ≠ 1: a unit depth makes the H-weighting invisible
const AB = 0.2      # > 0: else Model 4 degenerates to Model 3
const AE = 0.8      # a_eta ≤ h_min/3 keeps H = h+η* > 0 with margin

h_slope(x, y) = D0 * (1 + AB * sin(1.3x))
h_flat(x, y)  = D0

vert = assemble_vertical_tensors(2, 1, [0.0, 0.728, 1.0])
fld  = MMSField(vert.N_dof; Lx = 1.7, Ly = 1.1, omega = 1.3, a_eta = AE)
cbs  = field_callables(fld)
x0, y0, t0 = 0.37, 0.61, 0.23

n_pass = 0; n_fail = 0
function check(name, cond, extra = "")
    global n_pass, n_fail
    cond ? (println("  PASS  $name $extra"); n_pass += 1) :
           (println("  FAIL  $name $extra"); n_fail += 1)
end

SR(reg, fb, np, hf) = strong_residual_model(cbs, vert, hf, G, x0, y0, t0;
                                            regime = reg, flat_bed = fb, nl_pressure = np)
mx(r)      = max(abs(r[1]), maximum(abs, r[2]), maximum(abs, r[3]))
df(r1, r2) = max(abs(r1[1]-r2[1]), maximum(abs, r1[2].-r2[2]), maximum(abs, r1[3].-r2[3]))

# ---- N1: flat-bed limit — Model 4 at constant h must EQUAL Model 3 ----------
#  This is an exact identity, not an approximation: flat_bed zeroes ∇h at a single
#  control point, and a constant h has ∇h = 0 anyway, so the two must agree bitwise.
println("\nN1  flat-bed limit  (Model 4 @ const h  ≡  Model 3)")
let r3 = SR(:nonlinear, true, :none, h_flat), r4 = SR(:nonlinear, false, :none, h_flat)
    rel = df(r3, r4) / mx(r3)
    check("N1  nonlinear :none", rel < 1e-12, @sprintf("(rel %.2e)", rel))
end
let l1 = SR(:linear, true, :none, h_flat), l2 = SR(:linear, false, :none, h_flat)
    rel = df(l1, l2) / mx(l1)
    check("N1b linear (Model 2 @ const h ≡ Model 1)", rel < 1e-12, @sprintf("(rel %.2e)", rel))
end

# ---- N2: small-amplitude limit — the nonlinear excess scales as ε² ----------
#  Every term Model 3 adds to Model 1 is O(ε²) (rows C4, M2, M4, M5–M9, M11, M12,
#  M15 of tab: term classification), so halving the amplitude must quarter the gap.
#  This is the gate that would catch a term added at the WRONG ORDER.
println("\nN2  small-amplitude limit  (nonlinear − linear  ~  ε²)")
function nl_gap(scale)
    ff = MMSField(vert.N_dof; Lx = 1.7, Ly = 1.1, omega = 1.3, a_eta = AE*scale,
                  alpha = [(1.0 + 0.25j)*scale for j in 1:vert.N_dof],
                  beta  = [(0.7 - 0.16j)*scale for j in 1:vert.N_dof])
    cb = field_callables(ff)
    rn = strong_residual_model(cb, vert, h_flat, G, x0, y0, t0;
                               regime = :nonlinear, flat_bed = true, nl_pressure = :none)
    rl = strong_residual_model(cb, vert, h_flat, G, x0, y0, t0;
                               regime = :linear, flat_bed = true, nl_pressure = :none)
    df(rn, rl)
end
let ratio = nl_gap(1.0) / nl_gap(0.5)
    check("N2  halving the amplitude quarters the gap", abs(ratio - 4) < 0.35,
          @sprintf("(ratio %.3f, expect 4)", ratio))
end

# ---- N3: non-triviality — each model must actually differ from its parent ---
#  Without this, N1/N2 could both pass on a forcing that silently dropped the new
#  terms entirely. Always ask: if this term were missing, would a gate notice?
println("\nN3  non-triviality  (the added terms actually change the forcing)")
let r3 = SR(:nonlinear, true, :none, h_flat), l1 = SR(:linear, true, :none, h_flat)
    rel = df(r3, l1) / mx(r3)
    check("N3  nonlinear ≠ linear on a flat bed", rel > 1e-3, @sprintf("(rel %.2e)", rel))
end
let n4 = SR(:nonlinear, false, :none, h_slope), l2 = SR(:linear, false, :none, h_slope)
    rel = df(n4, l2) / mx(n4)
    check("N3b nonlinear ≠ linear over a slope", rel > 1e-3, @sprintf("(rel %.2e)", rel))
end
let n4 = SR(:nonlinear, false, :none, h_slope), r3 = SR(:nonlinear, true, :none, h_slope)
    rel = df(n4, r3) / mx(n4)
    check("N3c Model 4 ≠ Model 3 over a slope (∇h rows live)", rel > 1e-3,
          @sprintf("(rel %.2e)", rel))
end

# ---- N7: positivity — the nonlinear models divide by H ----------------------
println("\nN7  H = h + η* > 0 is enforced, not assumed")
let ok = false
    try
        bad = MMSField(vert.N_dof; Lx = 1.7, Ly = 1.1, a_eta = 5.0)   # a_eta ≫ h
        mms_forcing(bad, vert, h_flat, G; regime = :nonlinear, flat_bed = true)
    catch e
        ok = occursin("H = h+η*", sprint(showerror, e))
    end
    check("N7  a_eta > h is rejected with a clear message", ok)
end

# ---- N6: independence — the forcing must never touch the residual code ------
println("\nN6  independence from problem.jl")
let src = read(joinpath(@__DIR__, "..", "src", "mms.jl"), String)
    banned = ["global_residual", "jacobian_u", "build_problem"]
    hits = [b for b in banned if occursin(b, replace(src, r"#[^\n]*" => ""))]
    #  Build the detail string with string concatenation, NOT interpolation: an
    #  escaped double quote inside a $(...) interpolation is a parse error in
    #  Julia, and it made this whole file unparseable (so the gates never ran).
    check("N6  src/mms.jl references no residual code", isempty(hits),
          isempty(hits) ? "" : "(found: " * join(hits, ", ") * ")")
end

# ---- N8..N11: the 𝓝 TIERS (available since 2026-08-18) ---------------------
#  These four gates replaced a gate that asserted the tiers REFUSE. They refused
#  because of a Julia closure variable-capture collision inside strong_residual_model
#  (Nvec's `H`/`ukx`/`uky` aliased the enclosing function's locals), NOT because of
#  the nested-ForwardDiff limit that was recorded for two days. See the note at the
#  𝓝 block in src/mms.jl. The refutation was that components {7,8} are FIRST order
#  and failed identically to {1,2,4,5} — a fact no derivative-depth story explains.
println("\nN8  the 𝓝 tiers evaluate and are finite")
for (fb, hf, nm) in ((true, h_flat, "flat"), (false, h_slope, "slope")), np in (:native, :full)
    ok = false
    try
        r = SR(:nonlinear, fb, np, hf)
        ok = all(isfinite, vcat(r[1], r[2], r[3])) && mx(r) > 0
    catch
        ok = false
    end
    check("N8  :$np over a $nm bed evaluates finitely", ok)
end

# ---- N9: the flat_bed control point still holds under 𝓝 --------------------
#  Components {3,6} are ∇h-gated INSIDE Nvec, and flat_bed zeroes ∇h at one control
#  point. Over a constant bed the two routes must agree BITWISE, exactly as N1 does
#  for :none. This is what would catch a 𝓝 component that forgot its ∇h guard.
println("\nN9  flat-bed limit under 𝓝  (flat_bed=true ≡ flat_bed=false @ const h)")
for np in (:native, :full)
    a = SR(:nonlinear, true, np, h_flat); b = SR(:nonlinear, false, np, h_flat)
    rel = df(a, b) / mx(a)
    check("N9  :$np", rel < 1e-12, @sprintf("(rel %.2e)", rel))
end

# ---- N10: non-triviality — each tier must actually add something ------------
println("\nN10 tier ordering  (each tier changes the forcing)")
let n = SR(:nonlinear, false, :none,   h_slope),
    v = SR(:nonlinear, false, :native, h_slope),
    f = SR(:nonlinear, false, :full,   h_slope)
    r1 = df(v, n) / mx(n);  r2 = df(f, v) / mx(v)
    check("N10  :native ≠ :none", r1 > 1e-3, @sprintf("(rel %.2e)", r1))
    check("N10b :full ≠ :native", r2 > 1e-3, @sprintf("(rel %.2e)", r2))
end

# ---- N11: the 𝓝 excess is QUADRATIC in the state amplitude -----------------
#  MEASURED 2026-08-18, all four models: 1.927 / 1.998 / 1.968 / 1.967 at the
#  smallest amplitude pair — i.e. O(A²), NOT the O(A³) once claimed for the whole
#  package. That matches the standing correction that the 𝓟 block is O(A²) and
#  dominates. This is the gate that would catch a 𝓝 term added at the wrong order.
println("\nN11 the 𝓝 excess scales as A²")
function n_excess(scale, np, fb, hf)
    ff = MMSField(vert.N_dof; Lx = 1.7, Ly = 1.1, omega = 1.3, a_eta = AE*scale,
                  alpha = [(1.0 + 0.25j)*scale for j in 1:vert.N_dof],
                  beta  = [(0.7 - 0.16j)*scale for j in 1:vert.N_dof])
    cb = field_callables(ff)
    df(strong_residual_model(cb, vert, hf, G, x0, y0, t0;
                             regime = :nonlinear, flat_bed = fb, nl_pressure = np),
       strong_residual_model(cb, vert, hf, G, x0, y0, t0;
                             regime = :nonlinear, flat_bed = fb, nl_pressure = :none))
end
for (np, fb, hf, nm) in ((:native, true,  h_flat,  "flat"),
                         (:native, false, h_slope, "slope"),
                         (:full,   true,  h_flat,  "flat"),
                         (:full,   false, h_slope, "slope"))
    #  small amplitudes only: at a_eta/h ≈ 0.3 the higher-order terms are still visible
    e1 = n_excess(0.125, np, fb, hf);  e2 = n_excess(0.0625, np, fb, hf)
    ord = log2(e1 / e2)
    check("N11 :$np over a $nm bed is O(A²)", abs(ord - 2) < 0.2,
          @sprintf("(order %.3f, expect 2)", ord))
end

println()
println("=" ^ 76)
@printf("  Results: %d PASS,  %d FAIL\n", n_pass, n_fail)
println("=" ^ 76)
n_fail > 0 ? error("test_mms_forcing_nonlinear: $n_fail failed!") :
             println("  Nonlinear MMS forcing gates OK (:none, :native and :full tiers).")
