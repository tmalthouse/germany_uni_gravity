# Gravity Analyses: Findings and Caveats

Student-university gravity analyses, 1800–1848. This is the research log
carried over from the original project (`analysis/gravity/README.md`), with
its text unchanged. Script names below are the original ones; the README's
"Where the old scripts went" table maps each to its module here. Everything is
rebuilt with `make` rather than `run_all.py`.

---

## Data

### `build_od.py` → `od.parquet` (era grid), `od_year.parquet` (year grid)

Builds origin–destination flow grids from `data/students_final.parquet`
at the repo root (`../../data/` relative to this directory)
(132,999 enrollment spells at 13 German universities — carried as 14
destination labels, since Munich appears under both its seats — geocoded,
`in_germany_broad` sample). One row per enrollment spell (`spell_id` dedup —
schools recording every semester would otherwise be overrepresented ~2×).

- **`od.parquet`** (era grid): one row per origin × dest × era (era = decade of
  first enrollment, 1 = 1800s … 5 = 1840s). Zero-padded; 1,022,770 rows
  (14,611 origins × 14 destination labels × 5 eras).
- **`od_year.parquet`** (year grid): one row per origin × dest × true
  enrollment year. Zero-padded; 10,023,146 rows. **All annual/event-study
  analyses use this grid.**

Both carry: great-circle `dist_km`, `log_dist` (floored at 0.5 km for
same-city cells, flagged by `same_city`), `same_state`/`same_polity`
(shared ruling polity: state = polity//100), confession vars via
`CONFESSION_MAP` (origin polity) and `UNI_CONFESSION` (destination).

**Implementation decisions & caveats**

- *Historical bug (fixed):* the era grid's `first_year` is the modal year
  within a decade cell — never use it for annual analysis. The year grid is
  exact (verified 0 mismatches vs student-level counts).
- The 1840 Göttingen recording gap was a source-level bug, fixed upstream in
  cleaning (Hannover-city 1839/1840 entries were misassigned).
- Zero-padding: era cells before a university existed (Berlin/Bonn in eras
  1–2) appear as observed zeros. Harmless for specs that give Berlin/Bonn no
  era-specific terms, but not for ones that do: Design B's dynamic spec
  therefore **drops these 58,372 cells** (all zero flow) before estimating.
  Kept, the era-specific Berlin/Bonn coefficients are fit purely to
  pre-founding zeros — which is what produced the old, spurious Bonn
  "maturation" reading (see §4–5).
- 132 university-year cells have zero total flow (true single-year gaps at
  year level, invisible at era level). Descriptive weighted means must filter
  `flow > 0` (see design B figure).
- `CONFESSION_MAP` classifies by ruling house / territorial church — a
  ruling-elite proxy, not population confession. **Exception:** Württemberg is
  coded `protestant` (its Lutheran state church), not by its Catholic ruling
  house — see §1–2.
- *Unmapped origins are not actually labelled `unknown`.* `pl.Series.replace`
  leaves unmatched values **unchanged**, so origins whose polity name is the
  empty string (boundary-gap geocodes) keep `''`; `fill_null` never sees them.
  Consequently the `unmapped` warning in `attach_covariates` is dead code that
  can never fire, and these origins are **not excluded** from
  `same_confession` — they fall into `ss_dc` or the omitted `ds_dc` baseline.
  That is 13,042 of 132,999 flows (9.8%).
- For the same reason `mixed` origins (35,991 flows, 27%) count as
  *not* same-confession. So `ss_dc` means "same state, not **verified**
  same confession" — a pool of genuinely-different, mixed and unknown — not
  "same state, different confession". Read the confession split accordingly.

**Presentation order below follows the paper arc: market description →
fragmentation → the new universities → reputation shock.**

---

## §1–2. Baseline gravity — `estimate_gravity.py` → `gravity_results.csv`

**What is estimated.** PPML (`pyfixest.fepois`) flow gravity on the era grid:

```
flow(o,d,e) ~ log_dist + same_city + same_state + same_polity | origin_id + dest
```

plus era-interaction specs (`dist_x_era`, `state_x_era`, `both_x_era`,
reference era 1) and a confession split. SEs clustered by origin.

**Headline estimates.** Distance elasticity −1.37 (SE 0.064); same-state
+1.65 (≈ ×5.2 odds); `same_polity` n.s. (+0.22, SE 0.20) — once state is
controlled, finer polity identity adds nothing.

Confession split: the border effect survives with **no shared confession at
all** (`ss_dc` +1.68, ≈ ×5.3), so borders are not confessional proxies. Within
states, shared confession adds a further +0.22 (`ss_sc` +1.90 vs `ss_dc`
+1.68) that is **not significant** (Wald p = 0.13) but imprecise — 95% CI
[−0.07, +0.52], so a sizeable premium cannot be ruled out. Across states,
shared confession buys +0.37 (SE 0.11). **Do not write that same-state pairs
"cost nothing confessionally"** — that reading leaned on the Württemberg
miscoding below.

*Coding note (Württemberg).* A pure ruling-house rule would code Württemberg
`catholic`; since 2026-09-14 it is coded `protestant` (Lutheran state church
and population). That moves Württemberg→Tübingen — 7,898 flows, ~21% of the
old `ss_dc` cell — from "same state, different confession" into `ss_sc`.
Effect of the recode: `ss_sc` 1.86 → 1.90, `ss_dc` 1.79 → 1.68, `ds_sc`
0.56 → 0.37, and the within-state gap 0.07 (p = 0.63) → 0.22 (p = 0.13).
See also the `ss_dc` composition caveat above.

**Caveats.**
- Era grid: `era` is the correct unit here; fine.
- Confession is a ruling-house classification (see above), with Württemberg
  the one deliberate exception.
- Cluster-robust SEs by origin; no higher-level clustering (polity) tested.

## §3. Fragmentation over time & the 1819 event study — `estimate_gravity.py` (`main_event_study`), `plot_era_coeffs.py`, `plot_event_study.py`

**What is estimated.** (i) `same_state:C(era)` and `log_dist:C(era)` on the
era grid; (ii) a Carlsbad (1819) event study on the **year grid**:
`same_state:C(first_year)` with origin + dest + year×dest FEs, 1801–1848.

**Findings.** Border effect strengthens from ×3.8 (1800s) through ×4.1
(1810s) to a **peak of ×5.9 in the 1820s**, then eases slightly to ×5.1
(1830s) and ×5.0 (1840s) — read the peak as the 1820s, not the 1840s;
distance gradient weakens modestly (+0.14 to +0.18 log points by
era 5 in the joint spec). 1819 event study: border effect drifts smoothly
upward (~1.2 → ~2.0 log points by 1824) with **no 1819 discontinuity** —
slow institutional forces, not a repression shock.

**Caveats.**
- The 1819 event study ran on the era grid's modal years before the fix;
  current version is on true annual data. Joint Wald on year interactions is
  significant by construction (48 years); read the path, not the Wald.
- `era^dest` FEs absorb destination time trends; exposure-style claims must
  not be read off these specs.

## §4–5. Design B: composition of new-university flows — `design_b_composition.py` → `design_b_composition.png`

**What is estimated.** On the era grid, Berlin/Bonn split from "new":

```
flow ~ log_dist + same_city + is_berlin:log_dist + is_bonn:log_dist
       + is_berlin:C(era) + is_bonn:C(era) | origin_id + dest + era
```

and a dynamic version, estimated with the pre-founding Berlin/Bonn cells
(eras 1–2) dropped:

```
flow ~ log_dist + same_city + is_berlin:log_dist:C(era) + is_bonn:log_dist:C(era)
       | origin_id + dest^era
```

Its Berlin/Bonn coefficients are **relative** to the old universities'
gradient (`log_dist`): positive = less distance decay than the old
universities. `dest^era` FEs absorb each school's enrollment level in each
era, so level growth cannot leak into the era gradients; `same_city` absorbs
the arbitrary 0.5 km same-town floor from `build_od.py` (never interpret its
own coefficient). The script also prints an equal-era Wald test per school.
This spec replaced one with no base `log_dist`, `dest + era` FEs and the
structural zeros kept — see the ⚠ note under Findings.
The figure is the flow-weighted mean origin distance by year and school
(Berlin / Bonn / old-university average), **from the year grid**.

**Findings.**
- Static: `is_berlin:log_dist` +0.74 (SE 0.14) → Berlin's distance elasticity
  −1.26 vs old universities' −2.00; Bonn only +0.18 (−1.82). (Note this −2.00
  is not the §1–2 baseline −1.37: this spec adds era FEs and the Berlin/Bonn
  interactions. They are different objects — do not quote them as one.)
- *Is Berlin's reach about Berlin? (2026-09-15, `design_b_berlin_checks.py`)*
  The static Berlin premium does not come from East Prussia, rural origins or
  coarse geocoding. Without East and West Prussia it is +0.71 (SE 0.14);
  without all eastern provinces (adding Posen and Silesia) +0.68 (0.15); for
  mapped, non-eastern origins alone (by-region split) +0.77 (0.15). Small
  origins (<10 students) send fewer students to Berlin but with the same
  gradient (difference −0.11, p = 0.49). Berlin has far more coarse geocodes
  than any other school — 27% of its spells are placed at a territory,
  district or country point, against 3–5% at most schools, some far from
  where students lived (Pommern at 361 km, Schlesien at 463 km) — yet keeping
  only settlement-level geocodes makes the premium larger, not smaller:
  +0.81 (0.11), or +0.89 (0.12) also without eastern and unmapped origins.
  With same-state and same-polity controls it is +0.65 (0.14); adding
  Berlin × same state, +0.57 (0.15), with Berlin's own border effect weaker
  than the other universities' (−0.68, SE 0.29, p = 0.02). Part of the
  premium is Prussia, but most of it is not, and Berlin is not mainly a
  Prussian draw. Note that most of Berlin's apparent East/West Prussian
  intake is two province-level geocodes (hometown "Berlin", region Pommern
  or Ostpreußen), and its genuine East Prussian origins are towns (Danzig,
  Königsberg, Elbing, Thorn), not rural hinterland.
- Dynamic (gradients relative to the old universities' −2.00; eras 3/4/5 =
  1820s/30s/40s):
  - **Berlin** +0.69 / +0.76 / +0.77 (SE 0.13 / 0.14 / 0.16); equal-era
    Wald p = 0.25. Berlin's distance elasticity is ≈ −1.3 against −2.0 for the
    old universities **in every era it is observed**, and equality across eras
    cannot be rejected. Its reach advantage is a **level effect present from
    first observation**, not a growth process.
  - **Bonn** +0.11 / +0.21 / +0.19 (SE 0.08 / 0.08 / 0.10); equal-era
    p = 0.0001. Close to the old universities in the 1820s (n.s.), then a
    modestly wider reach from the 1830s — a real but small shift (≈0.1 log
    points). A slight widening, **not** convergence from a much narrower base.
  - Cite the equal-era Wald tests, not overlapping CIs: the era coefficients
    are correlated ≈0.99, so level SEs are wide (≈0.13–0.16) while era
    *differences* are precise (SE ≈0.01–0.02).
  - ⚠ **Superseded readings — do not repeat.** The previous dynamic spec had
    (i) no base `log_dist`, so it estimated *absolute* gradients and gave
    old-university cells no distance term at all; (ii) `dest + era` FEs, so
    school level growth leaked into the gradients — with base `log_dist` but
    without `dest^era`, Berlin's gradient spuriously appears to change across
    eras (p < 0.001); and (iii) the pre-founding zeros kept. Together these
    produced a Bonn "maturation" story (−2.66 in eras 1–2 → −1.5 in eras 3–5)
    estimated entirely off structural zeros — Bonn has no data before 1825 —
    and absolute Berlin figures (−0.84/−0.76/−0.79) mislabelled as relative.
- Annual figure: old universities' mean distance *falls* from ~183 km in 1800
  to ~150 km by the mid-1820s; Berlin starts at ~283 km in 1821 and climbs.
  *There is no 1837 dip in Berlin's catchment* — an earlier draft claimed a
  "349 → 247 km" fall at 1837 and read it as the Göttingen shock. Berlin's
  series runs 314 (1836) → 315 (1837) → 320 (1838) → 328 (1839): it rises
  straight through. The 349/247 pair is 1833 (349.8) and 1831 (247.8) — wrong
  years, and the reverse direction. Do not repeat the claim. If anything
  Berlin's catchment *widens* after 1837, which is consistent with §8e.

**Implementation decisions.**
- The pooled "new" category was abandoned after the split: it averaged two
  schools with very different gradients (Berlin ≈0.7 less steep than the old
  universities, Bonn ≈0.1–0.2), and the earlier "growing gradient" it showed
  leaned on the structural-zero artifact described under Findings.
- The **dynamic** spec drops the 58,372 pre-founding Berlin/Bonn cells and
  uses `dest^era` FEs (see above). The **static** spec keeps those cells; its
  degenerate `is_berlin:C(era)` / `is_bonn:C(era)` era-1/2 terms are never
  quoted (see Caveats), and dropping the cells would not change the
  coefficients it reports.
- Figure filters `flow > 0` before weighted means (132 zero-flow
  university-year cells otherwise poison the old-university line with NaN).

**Caveats.**
- Static spec: era main effects for Berlin are numerically degenerate (SE
  ~10⁵) due to collinearity with dest FEs — never quote them. Bonn's era-1/2
  terms look precise (≈ −16.6, SE ≈ 0.1) but are equally meaningless: they are
  fit to pre-founding zeros.
- ⚠ *Correction (2026-09-14):* this caveat used to read "field composition
  is untestable (fields unrecorded at Berlin/Bonn/Tübingen/Jena)". The fields
  are recorded; the register cleaners computed the field groups for these four
  schools only on intermediate tables that were never used, so `field` was
  null there. It is now populated (see README, "Fixes since the port"). Field
  composition at the new universities has not been analysed yet; until it is,
  the Heidelberg proxy analyses (`heidelberg_tests.py`,
  `robustness_student_religion.py`) are directional support only. Heidelberg
  religion is the only student-level religion data.
  ⚠ Corrections to those analyses (2026-09-14):
  - Both scripts used **Tübingen's coordinates** for Heidelberg (101 km off;
    11% of students on the wrong side of the 300 km line). Corrected
    Protestant >300 km shares by decade: 0.41 / 0.39 / 0.36 / 0.35 / 0.40 —
    a decline to the 1830s and a 1840s rebound, which fits neither a
    1810s–20s reputation peak nor a flat structural pattern. **Interpretation
    pending.**
  - `heidelberg_tests.py`'s field-composition panel normalised across near/far
    instead of across fields (bars summed to ~3.6 and ~1.4) and, because
    `pivot` orders columns by first appearance, had the near/far labels
    swapped. Corrected: law/admin is 64% of near vs 81% of far Protestants;
    theology 16% vs 4%; medicine 15% vs 11%.
  - `robustness_student_religion.py`'s PPML was fixed on 2026-09-14. It used to
    cluster on 3 groups that were also its fixed effect, silently drop the
    collinear `rel_catholic`, and include Jewish students despite its
    docstring. It now uses Catholic and Protestant students only, no
    `rel_catholic`, and heteroskedasticity-robust SEs (SEs clustered by origin
    polity are printed as a check), on 102 polity × religion cells. Result:
    `log_dist` −0.32 (SE 0.15), Baden origin +1.66 (SE 0.24), and
    **Catholic × Baden +0.53 (SE 0.40, p = 0.19; polity-clustered p = 0.095) —
    not significant at the 5% level.** Catholics are descriptively far more often from Baden
    (30% vs 12% of Protestants), but distance and the Baden main effect account
    for that; there is no significant *extra* Catholic–Baden premium. The old
    version's apparently precise interaction came from its invalid SEs (and the
    wrong Heidelberg coordinates) and must not be cited.
- Bonn's data begins 1825, Berlin's 1821 — "level effect from first
  observation" cannot speak to 1810–1820.

## §6. Design A: substitution DD — **REMOVED**

`design_a_substitution.py` was removed from the pipeline (2026-09-14) and must
not be cited. The destination-level exposure design is not identified:

- **Leave-one-destination-out** moves `expo:post` from **+0.68** (drop Giessen)
  to **−0.71** (drop Munich) — flipping sign and significance. Five single
  deletions shift it by 0.33–0.80. Restricting to the eight complete-coverage
  destinations gives −0.729; without Munich, −1.381.
- **The SEs were computed at the wrong level.** `expo:post` varies across 11
  destinations (12 distinct values) but was clustered by origin (14,593
  clusters). Re-clustering by destination inflates the SE from 0.163 to
  **0.990** — every swing above fits inside one correct standard error.
- **Cause:** the spec spends a destination FE *and* a destination-specific
  trend on each of ~11 destinations, then asks those ~11 destination-level
  points to identify one destination-level interaction.

**Do not write "no evidence of cannibalization."** The earlier −0.15 was one
arbitrary point in a range spanning +0.68 to −1.38, indistinguishable from
zero in any direction. This design had no power to detect substitution either
way — it was never a null.

**If substitution is revisited,** change the level of variation, not the
sample: use **origin-level exposure** — each origin's choice-set change when
Berlin/Bonn open (distance to the nearest new university, or an
inverse-distance market-access term). That varies *within* destination, is not
absorbed by destination FEs, and yields thousands of effective units instead
of 11. Adding Halle as a 12th destination does not address the problem.

The script is retained on disk, marked deprecated, for reference only.

---

## §8. The Göttingen Seven — five scripts

Historical treatment: King Ernst August dismissed seven Göttingen professors
(incl. the Grimm brothers) in December 1837 for protesting the revoked
Hanoverian constitution. All analyses use the **year grid**, PPML, manual year
dummies with **1837 omitted as explicit reference**, origin + dest + year FEs,
SEs clustered by origin, window **1834–1847** — it ends in 1847 because Berlin,
Bonn and Jena have no 1848 records (see §8a).

**Spec note (important).** A `goettingen:trend` term is exactly collinear
with a complete Göttingen×year dummy set — the original trend-controlled spec
was only estimable because the 1840 gap broke the collinearity by accident.
After the 1840 fix, the trend term is degenerate and was dropped. The plain
event study's pre-1837 path is itself the pre-trend diagnostic.

### §8a. Pooled event study — `goettingen7_event_study.py`

Flat 1834–35 (−0.02, −0.01), −0.31 in 1836, then **−0.52 (SE 0.11) in 1838**,
and **negative in every year through 1847** (post-1838–47 mean −0.41): no
recovery, and a further decline of about a quarter log point in the
mid-1840s (−0.36 in 1843, −0.58 in 1846, −0.63 in 1847). Joint Wald over
1838–1847 p ≈ 7e-16.

⚠ **Coverage corrections — three earlier claims withdrawn:**
- *"+0.84 in 1848 (political liberalization wave)"* is an artifact. Berlin,
  Bonn and Jena have **no records in 1848** (1,159 / 409 / 222 students in
  1847), Kiel's 1848 looks partial (49 vs ~150/yr), and total recorded flow
  falls from 3,361 to 1,441 across 7 reporting destinations instead of 10.
  Göttingen's share of recorded flow jumps from 8.7% to 27.6% because its
  largest competitors drop out of the records. **The §8 window therefore ends
  in 1847** (since 2026-09-14). Doing so left every 1834–1847 coefficient
  unchanged, since 1848 had its own year effect.
- *"Joint post-1837 Wald p ≈ 1e-45"* was inflated by that 1848 artifact;
  1838–1847 alone gives p ≈ 7e-16.
- *"Declining further to −0.67 by 1846"* overstated 1846 specifically: that
  year was inflated by Heidelberg's dating error (937 enrollments in 1846 vs
  532 in 1845 and 338 in 1847; see the cross-cutting caveats), now corrected
  at the source. On the corrected data 1846 is −0.58 and 1847 −0.63. An
  interim version of this note went further and called the path "persistent,
  not deepening"; that rested on dropping Heidelberg from the *uncorrected*
  data and is also withdrawn — the decline holds (below).

*Robustness (window 1834–1847, corrected Heidelberg dates).*

| Göttingen | 1838 | 1846 | 1847 | post-1838–47 mean | mean 1845–47 − mean 1840–42 |
|---|---|---|---|---|---|
| all destinations (shipped) | −0.52 | −0.58 | −0.63 | −0.41 | −0.27 |
| stable coverage (8: drops Erlangen, Jena, München) | −0.56 | −0.54 | −0.63 | −0.40 | −0.29 |
| stable coverage without Heidelberg (7) | −0.52 | −0.42 | −0.53 | −0.33 | −0.24 |

The 1838 drop, its persistence and the mid-1840s decline are robust (the 1836
dip also remains, −0.34, pre-trend p = 0.003 in the stable set; see §8c). The
post-period *magnitude* depends somewhat on whether Heidelberg, a fast-growing
comparison school, is in the pool.

### §8b. Near/far heterogeneity — `goettingen7_heterogeneity.py`

Origins split at 200 km from Göttingen (haversine; verified against the
grid's `dist_km`, max deviation 0). **Far origins (≥200 km) drive the
collapse**: −0.55 to −1.26 log points in 1838–1847. Near origins:
milder, −0.13 to −0.50. The political shock repriced Göttingen for mobile,
cross-border students — the same students Design B shows choosing Berlin.

### §8c. The 1836 pre-trend dip — diagnosis (in-text, no script)

The joint pre-trend test is marginally significant (p = 0.016), driven by
1836 (−0.31). Decomposition: the dip is **near-origin** (far flow flat:
111/116/128 across 1835–37; near 356→229→340), concentrated in **Hannover
city's cohort volume** (total outbound fell 227→156, no offsetting
destination), reversed in 1837. Reading: cohort-timing noise in the
Hanoverian pipeline, not a demand-side shift toward other universities —
no destination-side control absorbs it (see §8d), and it is orthogonal to
the far-origin margin the shock operates on. *Footnote this; do not hide it.*

### §8d. Donor-pool robustness — `goettingen7_matched.py`

Same spec against restricted pools: old Protestant (6: Kiel, Marburg,
Giessen, Erlangen, Jena, Tübingen) and west Protestant (4: Kiel, Marburg,
Giessen, Erlangen). **The 1836 dip survives every pool** (−0.31, −0.31,
−0.22) — it is an origin-side volume shock, not differential university
trends. Post-period estimates are noisier in tighter pools (4–6 controls)
but the 1838 impact and persistent decline hold in all three.

⚠ **The pools are thinner than their labels.** Marburg has no flow anywhere
in 1834–1848 and Erlangen none after 1843; Jena lapses in 1845 (17 students)
and is absent in 1848. So "old Protestant (6)" is effectively five schools
through 1843 and three to four after, and "west Protestant (4)" is three
through 1843 and **only two (Kiel, Giessen) from 1844**. Read the tight pools'
late-period estimates accordingly.

### §8e. Reallocation — `goettingen7_reallocation.py`

Where did Göttingen's far students go? Treated origins: 100 far origins with
Göttingen flow 1834–37; destination shares pre (1834–37) vs post (1838–41):
Göttingen 21.4%→8.3% (−13.1 pp); **Berlin 42.3%→51.5% (+9.1 pp)**; smaller
gains at Tübingen (+1.7), München (+1.7), Bonn (+1.4); Heidelberg flat.
PPML: Göttingen×post −1.02 (SE 0.14); new-schools×post +0.13 (n.s. — Berlin
was already the dominant destination; the shift is share reallocation).
Near-origin comparison: same direction, weaker (−10.5/+5.0 pp).

**Interpretation.** The displaced far flow went *to Berlin*, not out of the
market — the mobile-student population the research universities cultivated
was precisely the population the political shock reallocated.

**Caveats.** Descriptive share shift, not an identified causal effect on
Berlin (Berlin's share was already rising). The causal weight stays on the
Göttingen coefficients; the table establishes *where the flow went*.

### §8f. Placebo test — `goettingen7_placebo.py`

The same event-study spec applied to each other university (1837 reference,
same 1834–1847 window). The figure and summary use the **seven
well-identified placebos** — Berlin, Bonn, Freiburg, Giessen, Heidelberg,
Kiel, Tübingen; the CSV keeps all ten paths.

**What distinguishes Göttingen is the treatment date.** In 1838 Göttingen is
at −0.52; the seven placebos are at +0.08 (Berlin), +0.05 (Bonn), +0.23
(Freiburg), +0.18 (Giessen), +0.39 (Heidelberg), −0.02 (Kiel) and +0.06
(Tübingen) — none meaningfully negative. Dips of similar size occur
elsewhere, but later (Kiel −0.70 in 1839; Bonn −0.35 in 1841).

**Depth and persistence do not distinguish it.** Kiel dips deeper than
Göttingen (−0.70 vs −0.63, Göttingen's lowest point, in 1847). Kiel's 1838–47 mean (−0.41) equals Göttingen's, Kiel is negative in
every year 1839–1847, and its pre-period is already negative. Make the claim
about timing, not durability.

**Caveats.**
- **Excluded as near-unidentified:** Erlangen (no records after 1843), Jena
  (a lapse in 1845, 17 students) and München (a lapse in 1842, 1 student).
  Their year dummies are fit to near-empty years, so their paths fall to
  −13.3, −2.4 and −4.7. Berlin and Bonn were excluded while the window ran to
  1848, where they have no records; within 1834–1847 they are fully covered.
  Marburg and Würzburg have no year-level flow in the window and are not in
  the placebo set.
- Placebos are real universities with real histories (e.g. Kiel and the
  Schleswig-Holstein question) — the claim is that dips are common but a dip
  *dated to a documented shock* is not.

---

## Presentation-order summary

| Order | Script | Output | Paper role |
|---|---|---|---|
| 1 | `estimate_gravity.py` | gravity_results.csv | baseline market structure (Table 1) |
| 2 | `estimate_gravity.py` + `plot_era_coeffs.py` | era_coefficients.png | fragmentation (Fig. 1) |
| 3 | `estimate_gravity.py` + `plot_event_study.py` | event_study_1819.png | no-1819-shock robustness |
| 4 | `design_b_composition.py` | design_b_composition.png | Berlin-led composition (Fig. 2, headline) |
| 5 | `goettingen7_event_study.py` | goettingen7_event_study.csv/.png | shock (Fig. 3) |
| 6 | `goettingen7_heterogeneity.py` | …csv/.png | far-origin mechanism (Fig. 4) |
| 7 | `goettingen7_reallocation.py` | …csv/.png | flow went to Berlin (Fig. 5; the Munich row is now "muenchen", the post-1826 seat — the 1834–41 windows are entirely after the move) |
| 8 | `goettingen7_matched.py` | …csv/.png | donor-pool robustness |
| 9 | `goettingen7_placebo.py` | …csv/.png | placebo robustness |
| — | `heidelberg_tests.py`, `robustness_student_religion.py` | …png | religion mechanism aside |

## Cross-cutting caveats

- **Linkage-id join (pipeline bug, fixed 2026-09-14).** `clean_final.py`
  attached `spell_id`/`student_id` by joining on (last name, first names,
  school, first year), which is not unique: *n* records sharing that key
  became *n²* rows. `students_final` had 3,960 duplicate rows (Kiel +41%, Bonn
  +2.5%, Berlin +1.5%, every other school under 1%), and
  `data/student_counts.parquet` was inflated accordingly. Namesakes also picked
  up each other's `spell_id`s, so after `build_od.py`'s one-row-per-spell
  dedup 590 spells kept another person's hometown (0.29% of flow in the wrong
  origin cell) and the gravity sample counted 133,219 spells instead of
  132,999. The join now uses the record key (`index` = linkage `rec_id`) and is
  checked one-to-one. **Any figure or table built from `student_counts.parquet`
  before this date — Kiel especially — needs regenerating.**
- **Spell-level dedup** is applied in `build_od.py`, so the OD grids count
  enrollment spells. The Heidelberg scripts (`heidelberg_tests.py`,
  `robustness_student_religion.py`) read `students_final.parquet` directly and
  do **not** dedup — they count recorded rows. At Heidelberg this is harmless: the register
  records one row per spell (15,528 rows = 15,528 spells in the religion
  sample). The 0.6% gap reported here earlier (15,650 rows vs 15,550 spells)
  was entirely the linkage-id join duplication above, not re-recorded
  semesters. Dedup would matter only if these scripts were pointed at a school
  that records every semester (Berlin, Bonn, Jena, Tübingen).
- **Single-observation windows**: Berlin pre-1821 and Bonn pre-1825 do not
  exist; era-1–2 claims about the new schools are structural zeros.
- **Clustering**: all SEs clustered by origin place; no two-way (origin ×
  year) clustering tested.
  ⚠ Origin clustering is only defensible where the regressor varies at the
  origin–destination level. It is **wrong for Design A**, whose `expo:post`
  varies across 11 destinations only: clustering by destination inflates that
  SE from 0.163 to 0.990 (see §6). Any future destination-level treatment
  needs the same check before its SEs are quoted.
- **`muenchen_old` vs `muenchen`** (pipeline bug, fixed): one university, two
  labels — the 1826 move from Landshut to Munich. `muenchen_old` = Landshut
  years (1801–1825), `muenchen` = Munich years (1826–1848), and they are
  distinct destinations with distinct seats.
  *Until this was corrected the split did not exist.* The rule in
  `geocode_pipeline/geocode/resolve.py` read `first_year < 1926` — a
  transposed digit — and since the corpus ends in 1848 it was true for every
  Munich row. All 6,293 were relabelled `muenchen_old` and geocoded at
  Landshut (TGN 7077307), 62 km from Munich; `school_fixed == 'muenchen'`
  never occurred, so Munich's own seat (TGN 7004333) was never used. 3,610
  post-1826 enrollments carried the wrong seat, and `checks.py` could not
  catch it because it only asserts that each emitted variant exists in
  `uni_tgn_mapping.csv`. Any result produced before the fix has Munich's
  `dist_km` / `log_dist` / `same_city` wrong; `same_state` and confession are
  unaffected (both seats are Bayern and Catholic).
  Do not sum the two labels as one institution across the move without
  deciding how to treat the relocation itself.
- **Annual series are noisy**: single-year flow-weighted means swing with
  cohort composition (Hannover city alone sends 130–235 students/year to
  Göttingen). Prefer era-level claims except where the annual event study is
  the point.
- **Source coverage is not constant inside the §8 window.** Berlin, Bonn and
  Jena have no 1848 records; Erlangen ends after 1843; Marburg and Würzburg
  have none in 1834–1848; Jena (1845) and München (1842) have one-year
  lapses. (Heidelberg's apparent 1846 spike, 937 vs ~330–530, was an
  extraction dating error, since corrected — see below.) Year fixed effects
  absorb total volume but not changes in *which* schools report, so any
  destination-relative annual coefficient near these years can move for
  purely clerical reasons. Check coverage before interpreting a single-year
  coefficient. **Heidelberg dating error (found and corrected 2026-09-14).** Where an
  entry's date in the register was a ditto mark, the extractor took the year
  from the scan's footer, which names the *volume*, so in `heidelberg_1846`
  372 entries on 1847–48 pages were dated 1846. The page years read from the
  running heads (`heidelberg_date_fix/`) should have caught this, but three
  things stopped them: the old correction only acted on gaps of more than 2
  years; the page-year table had no 1846 entry (its generator dropped each
  volume's opening year, and one misread head also dropped 1843); and page
  years were forward-filled across volumes. All three are fixed. For
  `heidelberg_1846`, whose page boundaries were confirmed against the register
  (1846 = pp. 10–43), no entry may now predate its page's year. Raw extraction
  rows per year, 1840–1848, before → after: 494 / 430 / 543 / 387 / 471 / 544 /
  **955** / 347 / 179 → 451 / 430 / 433 / 540 / 409 / 485 / **645** / 598 /
  339.
  **Open question:** the same rule applied to `heidelberg_1807` would move
  1,263 entries by 1–3 years (all later), reshaping Heidelberg's 1834–45
  series. The data cannot tell whether those entries are misdated or that
  volume's detected page starts are a spread or two off; it needs a spot-check
  against the register, so it has not been applied.
- Everything regenerable via `run_all.py`; grids are deterministic from
  `students_final.parquet`.