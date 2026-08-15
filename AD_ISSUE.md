# Gridap.jl transient multifield AD — ROOT CAUSE FOUND, and how to fix it

> **Revision 2 (2026-08-15).** Revision 1 was a hypothesis document written without the Gridap
> sources. They are now vendored at `Gridap.jl/` and the failure has been traced to a specific
> defect at a specific line. **Sections 1–3 below supersede the earlier hypotheses A/B/C, all three
> of which were wrong.** The original speculative material is kept in §7 for the record, clearly
> marked, because knowing *why* those guesses missed is useful.

---

## 1. THE ROOT CAUSE — one line, in Gridap, not in user code

`Gridap.jl/src/ODEs/TransientCellFields.jl:213`

```julia
function time_derivative(f::TransientMultiFieldCellField)
  cellfield, derivatives = first_and_tail(f.derivatives)

  single_field_derivatives = map(cellfield, derivatives...) do cellfield, derivatives...
    TransientSingleFieldCellField(cellfield, derivatives)
  end

  TransientMultiFieldCellField(          # <-- 3-arg inner constructor
    cellfield, derivatives,
    single_field_derivatives             # <-- may be a Tuple; struct demands a Vector
  )
end
```

against the struct at `:113`

```julia
struct TransientMultiFieldCellField{A} <: TransientCellField
  cellfield::A
  derivatives::Tuple
  transient_single_fields::Vector{<:TransientCellField}   # <-- VECTOR, not Tuple
end
```

**`map` returns a `Tuple` when its input is a `Tuple`, and a `Vector` when its input is an array-like
`MultiFieldCellField`.** When `cellfield` (i.e. `f.derivatives[1]`) is a plain `Tuple` of single
fields, `single_field_derivatives` is a `Tuple`, the third positional argument fails the
`Vector{<:TransientCellField}` field type, and Julia reports **no applicable constructor**.

### 1.1 The observed error decodes exactly onto this call

```
MethodError: no method matching Gridap.ODEs.TransientMultiFieldCellField(
  ::Tuple{GenericCellField, SingleFieldFEFunction, SingleFieldFEFunction},          # arg1 cellfield
  ::Tuple{},                                                                        # arg2 derivatives
  ::Tuple{TransientSingleFieldCellField, TransientSingleFieldCellField,
          TransientSingleFieldCellField})                                           # arg3 <-- TUPLE
```

* **arg1** is a `Tuple` of our three fields `[η, 𝖴x, 𝖴y]` ⇒ `first_and_tail` returned a Tuple, so
  `map` produced a Tuple. Confirms the mechanism.
* **arg2** is `Tuple{}` ⇒ the tail is empty, consistent with a single time-derivative order.
* **arg3** is a `Tuple` where the struct requires `Vector{<:TransientCellField}`. **This is the
  mismatch.** Everything else dispatches fine (`A` is unconstrained, `derivatives::Tuple` matches).

### 1.2 Why the HAND-Jacobian path works and only AD fails

The residual calls `∂t(u)` (= `time_derivative`) in both paths — so why only AD?

* **Hand path.** We build `TransientCellField(uh, (uht,))` with `uht::MultiFieldFEFunction`. So
  `f.derivatives[1]` is a `MultiFieldFEFunction`, which is array-like ⇒ `map` returns a **Vector**
  ⇒ the constructor matches ⇒ works.
* **AD path.** Gridap re-seeds the state and hands `time_derivative` a `Tuple` of single fields
  rather than a MultiField container ⇒ `map` returns a **Tuple** ⇒ MethodError.

**⇒ The bug is a type instability in `time_derivative`, not anything about `ForwardDiff.Dual`.**
Note the error message contains **no `Dual` types at all** — a detail Revision 1 overlooked, and the
single strongest clue that the Dual-centred hypotheses were wrong.

---

## 2. THE FIX

### 2.1 Minimal, lowest-risk (recommended): accept a Tuple and normalise

Add one outer constructor. Purely additive — no existing call site changes behaviour.

```julia
# Gridap.jl/src/ODEs/TransientCellFields.jl, after the struct definition
function TransientMultiFieldCellField(
  cellfield, derivatives::Tuple, transient_single_fields::Tuple
)
  TransientMultiFieldCellField(
    cellfield, derivatives,
    collect(TransientCellField, transient_single_fields)
  )
end
```

*Why this and not the alternatives:* it changes no types already in use, cannot alter the hand-Jacobian
path (which never produces a Tuple there), and is a strictly wider dispatch surface.

### 2.2 Alternative: relax the struct field type

```julia
transient_single_fields::Union{Vector{<:TransientCellField},Tuple{Vararg{TransientCellField}}}
```
Wider blast radius — it changes the struct's layout/type parameters and every method that touches
the field. **Prefer 2.1.**

### 2.3 Alternative: fix at source, force a Vector in `time_derivative`

```julia
single_field_derivatives = collect(TransientCellField, map(...) do ... end)
```
Also correct and arguably the most honest fix, but it edits a function body rather than adding a
method, so a future Gridap upgrade silently reverts it. **2.1 survives upgrades better as a local
patch and is the easier upstream PR.**

### 2.4 How to apply it in THIS repository

The vendored `Gridap.jl/` is a clone, so the clean route is `Pkg.develop(path="Gridap.jl")`, apply
2.1, and rebuild. **Check first** whether the pinned registry version matches the clone — the
project runs Gridap 0.20.8 and `[compat]` admits `"0.19, 0.20"`. A `develop` switches the whole
project onto the clone, so re-run the reference tests afterwards (`test_basic` should give max η
0.00410, gauge 0.00212, 240 Newton).

---

## 3. VERIFICATION PLAN — prove the fix, then use it as an oracle

1. **Reproduce.** `run_mms_case(...; use_ad=true, flat_bed=true)` must currently throw the
   MethodError above. (Confirmed 2026-08-14 on Gridap 0.20.8.)
2. **Apply 2.1, re-run.** *Prediction:* the operator builds and the solve converges.
3. **Non-regression.** `test_basic`, `test_dispersion`, `test_sloshing`, `test_shallow_water` — the
   patch touches only a constructor that the hand path never hits, so all four must be **unchanged**.
4. **THE PAYOFF — use AD as the Jacobian oracle.** Run `use_ad=true` against `use_ad=false` on
   `regime=:linear, flat_bed=false`, the configuration where the hand Jacobian is known wrong:
   *Prediction:* **AD converges, hand fails.** That confirms the residual is correct and localises the
   defect to the hand Jacobian, which is precisely the question that has been open since 2026-08-14.
5. **Make it a standing gate.** Assemble both Jacobians on the same state and compare entrywise.
   This is the gate the task-5 set lacked: all four gates there tested the AD-derived *forcing*;
   none tested residual↔Jacobian agreement.

---

## 4. CONNECTION TO THE OPEN `sAK` BUG

Independent finite-difference work (`output/local/logs/termjac*.log`) already localised the failure
to the `𝓐/𝓚` bed-slope package (`sAK`):

```
flat_bed=T lin_press=F (control)  dR/du=2.219e-10  dR/du_t=1.531e-09   MATCH
flat_bed=F lin_press=F (no sAK)   dR/du=1.997e-10  dR/du_t=2.008e-09   MATCH
flat_bed=F lin_press=T (w/ sAK)   dR/du=8.153e-02  dR/du_t=1.553e-01   FAIL
```

**An unexplained detail that AD would settle immediately:** `∂R/∂u` fails too, although `sAK` depends
only on `u̇` and therefore contributes nothing to `∂R/∂u`. Since FD differentiates the *assembled
residual*, enabling `lin_pressure` apparently changes the residual's `u`-dependence — which the
source says is impossible. Either the residual is malformed at assembly level, or the FD harness is
misleading. **A working AD Jacobian distinguishes these in one run**, which is the strongest
practical argument for fixing AD before continuing to chase `sAK` by hand.

---

## 5. SOURCE MAP (verified against the vendored clone, not from memory)

| what | where |
|---|---|
| `TransientMultiFieldCellField` struct | `src/ODEs/TransientCellFields.jl:113` |
| 2-arg outer constructor (Vector path, works) | `:120` |
| `TransientCellField(::Tuple, ::NTuple{N,MultiFieldTypes})` | `:134` |
| `getindex` range → 3-arg call (Vector, fine) | `:200` |
| **`time_derivative` → 3-arg call (Tuple, BUG)** | **`:213–224`** |
| `_to_transient_single_fields` (always returns a Vector) | `:278–297` |
| AD jacobian entry points (`u0 = TransientCellField(y, u.derivatives)`) | `src/ODEs/TransientFEOperators.jl:279, 290, 439, 452, 615, 627` |

**Correction to Revision 1's source map:** it listed
`/src/FESpaces/FEOperatorsFromWeakForm.jl` as the ForwardDiff integration point for this path. The
relevant transient AD entry points are in `src/ODEs/TransientFEOperators.jl` at the lines above.

---

## 6. STATUS

* Root cause: **identified and located** (§1).
* Fix: **specified, not yet applied** (§2.1).
* Verification: **planned, not yet run** (§3).

---

## 7. SUPERSEDED — Revision 1 hypotheses, and why each was wrong

Kept deliberately: the failure modes of the guesses are instructive.

* **Hypothesis A — "missing constructor for Dual-wrapped tuples."** Directionally closest, but wrong
  in its cause. The constructor is missing for **plain `Tuple` vs `Vector`**; `Dual` is irrelevant.
  The tell was in the error message all along — **it mentions no `Dual` type whatsoever.**
* **Hypothesis B — "inconsistent velocity/state Dual promotion."** No evidence. The failure occurs
  when *constructing* the transient container, before any residual evaluation or Dual promotion.
* **Hypothesis C — "hardcoded types in the user residual."** Ruled out. `global_residual` is fully
  duck-typed (`r(t, u, v)` with no type annotations) and the same residual works on the hand path.

**Method note for future issues of this kind:** read the failing `MethodError`'s argument types
against the actual struct definition *first*. Here that alone identifies the culprit — argument 3 is
a `Tuple` where the field is declared `Vector` — and would have skipped three wrong hypotheses.
