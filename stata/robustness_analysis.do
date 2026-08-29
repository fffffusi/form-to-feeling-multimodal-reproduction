/*=============================================================================
 robustness_analysis.do
 -----------------------------------------------------------------------------
 Supplementary robustness analyses:
   1) 14 component-specific indirect associations with 5,000 bootstrap draws;
   2) minimum-review thresholds of 50 and 100 collected review texts;
   3) proxy review-volume weighted regressions with robust standard errors.

 Data:    data_anonymized/product_master.dta
 Output:  stata/output_robustness/

 Measurement note:
 n_reviews_total is the number of collected review texts. It has not been
 verified as the denominator used by the platform to calculate favorable rate.
 The weighted regressions are therefore a proxy sensitivity analysis, not
 exact inverse-variance weighting of the platform favorable-rate estimator.
=============================================================================*/

version 18
clear all
set more off
set linesize 255
set seed 42

capture mkdir "stata/output_robustness"

*=============================================================================
* Part 0. Load public data and reproduce the main-analysis transformations
*=============================================================================

use "data_anonymized/product_master.dta", clear

capture drop lnprice lnreviews
gen lnprice   = ln(price + 1)
gen lnreviews = ln(n_reviews_total + 1)

foreach v in y_favrate_only m_e_avg m_f_avg ///
             z_pc1 z_pc2 z_pc3 z_pc4 z_pc5 z_pc6 z_pc7 ///
             lnprice lnreviews has_discount free_shipping ///
             ln_imperfection ln_brand_avg_price recency_inv {
    capture confirm variable `v'
    if _rc {
        display as error "Required variable not found: `v'"
        exit 111
    }
    capture drop `v'_s
    egen `v'_s = std(`v')
}

global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

isid product_id
assert !missing(n_reviews_total)
assert n_reviews_total > 0
assert !missing($Y, $ME, $MF)

quietly count
local n_all = r(N)
quietly count if n_reviews_total >= 20
local n_20 = r(N)
quietly count if n_reviews_total >= 50
local n_50 = r(N)
quietly count if n_reviews_total >= 100
local n_100 = r(N)

tempfile analysis_data
save `analysis_data', replace

*=============================================================================
* Part 1. Fourteen indirect associations and bootstrap BCa intervals
*=============================================================================

capture program drop med_parallel_robustness
program define med_parallel_robustness, rclass
    syntax [if] [in]
    marksample touse

    quietly sem ($ME <- $X $CTRL) ///
        ($MF <- $X $CTRL) ///
        ($Y  <- $X $ME $MF $CTRL) if `touse'

    forvalues i = 1/7 {
        scalar aE`i' = _b[$ME:z_pc`i'_s]
        scalar aF`i' = _b[$MF:z_pc`i'_s]
    }
    scalar bE = _b[$Y:$ME]
    scalar bF = _b[$Y:$MF]

    forvalues i = 1/7 {
        return scalar indE_pc`i' = aE`i' * bE
        return scalar indF_pc`i' = aF`i' * bF
    }
end

bootstrap ///
    indE_pc1=r(indE_pc1) indE_pc2=r(indE_pc2) indE_pc3=r(indE_pc3) ///
    indE_pc4=r(indE_pc4) indE_pc5=r(indE_pc5) indE_pc6=r(indE_pc6) ///
    indE_pc7=r(indE_pc7) ///
    indF_pc1=r(indF_pc1) indF_pc2=r(indF_pc2) indF_pc3=r(indF_pc3) ///
    indF_pc4=r(indF_pc4) indF_pc5=r(indF_pc5) indF_pc6=r(indF_pc6) ///
    indF_pc7=r(indF_pc7), ///
    reps(5000) seed(42) bca ties ///
    saving("stata/output_robustness/bootstrap_parallel_14paths.dta", replace): ///
    med_parallel_robustness

capture log close indirectci
log using "stata/output_robustness/indirect_14paths_bootstrap_ci.log", ///
    name(indirectci) replace text
estat bootstrap, percentile bca
log close indirectci

tempname indirectpost
postfile `indirectpost' str12 path_id str12 channel byte pc ///
    double estimate boot_se z_normal p_normal_diagnostic using ///
    "stata/output_robustness/indirect_14paths_raw.dta", replace

foreach channel in E F {
    forvalues i = 1/7 {
        local parameter "ind`channel'_pc`i'"
        scalar b_now  = _b[`parameter']
        scalar se_now = _se[`parameter']
        scalar z_now  = b_now / se_now
        scalar p_now  = 2 * normal(-abs(z_now))
        post `indirectpost' ("`parameter'") ("`channel'") (`i') ///
            (b_now) (se_now) (z_now) (p_now)
    }
}
postclose `indirectpost'

preserve
    use "stata/output_robustness/indirect_14paths_raw.dta", clear
    export delimited using ///
        "stata/output_robustness/indirect_14paths_raw.csv", replace
restore

*=============================================================================
* Part 2. Minimum-review-threshold sensitivity analyses
*=============================================================================

use `analysis_data', clear

tempname senspost
postfile `senspost' str18 specification int cutoff byte weighted ///
    str10 model str28 path str16 term double b robust_se p_value ///
    int N double r2 using ///
    "stata/output_robustness/review_volume_sensitivity_long.dta", replace

foreach cutoff in 50 100 {
    preserve
        keep if n_reviews_total >= `cutoff'
        local spec "reviews_ge`cutoff'"

        quietly reg $ME $X $CTRL, vce(robust)
        estimates store thr`cutoff'_me
        scalar p_now = 2 * ttail(e(df_r), abs(_b[z_pc3_s] / _se[z_pc3_s]))
        post `senspost' ("`spec'") (`cutoff') (0) ("ME") ///
            ("PC3 -> affective") ("z_pc3_s") (_b[z_pc3_s]) ///
            (_se[z_pc3_s]) (p_now) (e(N)) (e(r2))

        quietly reg $MF $X $CTRL, vce(robust)
        estimates store thr`cutoff'_mf
        scalar p_now = 2 * ttail(e(df_r), abs(_b[z_pc6_s] / _se[z_pc6_s]))
        post `senspost' ("`spec'") (`cutoff') (0) ("MF") ///
            ("PC6 -> functional") ("z_pc6_s") (_b[z_pc6_s]) ///
            (_se[z_pc6_s]) (p_now) (e(N)) (e(r2))

        quietly reg $Y $X $CTRL, vce(robust)
        estimates store thr`cutoff'_ytotal

        quietly reg $Y $X $ME $MF $CTRL, vce(robust)
        estimates store thr`cutoff'_yfull
        foreach term in m_e_avg_s m_f_avg_s z_pc3_s z_pc6_s {
            scalar p_now = 2 * ttail(e(df_r), abs(_b[`term'] / _se[`term']))
            local pathlabel "`term'"
            if "`term'" == "m_e_avg_s" local pathlabel "Affective -> favorable rate"
            if "`term'" == "m_f_avg_s" local pathlabel "Functional -> favorable rate"
            if "`term'" == "z_pc3_s"   local pathlabel "PC3 direct -> favorable rate"
            if "`term'" == "z_pc6_s"   local pathlabel "PC6 direct -> favorable rate"
            post `senspost' ("`spec'") (`cutoff') (0) ("Y_full") ///
                ("`pathlabel'") ("`term'") (_b[`term']) (_se[`term']) ///
                (p_now) (e(N)) (e(r2))
        }

        esttab thr`cutoff'_me thr`cutoff'_mf thr`cutoff'_ytotal thr`cutoff'_yfull ///
            using "stata/output_robustness/Table_review_threshold_`cutoff'.rtf", replace ///
            b(%7.3f) se(%7.3f) star(* 0.10 ** 0.05 *** 0.01) ///
            scalar(N r2 r2_a F) compress nogaps label ///
            mtitles("Affective" "Functional" "Y total" "Y full") ///
            addnote("Heteroskedasticity-robust standard errors in parentheses." ///
                    "The sample is restricted by collected review-text volume." ///
                    "Review-text volume is not a verified favorable-rate denominator.")
    restore
}

*=============================================================================
* Part 3. Proxy review-volume weighted sensitivity analysis
*=============================================================================

quietly reg $ME $X $CTRL [aweight=n_reviews_total], vce(robust)
estimates store wt_me
scalar p_now = 2 * ttail(e(df_r), abs(_b[z_pc3_s] / _se[z_pc3_s]))
post `senspost' ("review_weighted") (0) (1) ("ME") ///
    ("PC3 -> affective") ("z_pc3_s") (_b[z_pc3_s]) ///
    (_se[z_pc3_s]) (p_now) (e(N)) (e(r2))

quietly reg $MF $X $CTRL [aweight=n_reviews_total], vce(robust)
estimates store wt_mf
scalar p_now = 2 * ttail(e(df_r), abs(_b[z_pc6_s] / _se[z_pc6_s]))
post `senspost' ("review_weighted") (0) (1) ("MF") ///
    ("PC6 -> functional") ("z_pc6_s") (_b[z_pc6_s]) ///
    (_se[z_pc6_s]) (p_now) (e(N)) (e(r2))

quietly reg $Y $X $CTRL [aweight=n_reviews_total], vce(robust)
estimates store wt_ytotal

quietly reg $Y $X $ME $MF $CTRL [aweight=n_reviews_total], vce(robust)
estimates store wt_yfull
foreach term in m_e_avg_s m_f_avg_s z_pc3_s z_pc6_s {
    scalar p_now = 2 * ttail(e(df_r), abs(_b[`term'] / _se[`term']))
    local pathlabel "`term'"
    if "`term'" == "m_e_avg_s" local pathlabel "Affective -> favorable rate"
    if "`term'" == "m_f_avg_s" local pathlabel "Functional -> favorable rate"
    if "`term'" == "z_pc3_s"   local pathlabel "PC3 direct -> favorable rate"
    if "`term'" == "z_pc6_s"   local pathlabel "PC6 direct -> favorable rate"
    post `senspost' ("review_weighted") (0) (1) ("Y_full") ///
        ("`pathlabel'") ("`term'") (_b[`term']) (_se[`term']) ///
        (p_now) (e(N)) (e(r2))
}

esttab wt_me wt_mf wt_ytotal wt_yfull ///
    using "stata/output_robustness/Table_review_volume_weighted.rtf", replace ///
    b(%7.3f) se(%7.3f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    mtitles("Affective" "Functional" "Y total" "Y full") ///
    addnote("Analytic weights are collected review-text counts." ///
            "Heteroskedasticity-robust standard errors are in parentheses." ///
            "The weights are a proxy, not a verified favorable-rate denominator.")

postclose `senspost'

preserve
    use "stata/output_robustness/review_volume_sensitivity_long.dta", clear
    sort weighted cutoff model term
    export delimited using ///
        "stata/output_robustness/review_volume_sensitivity_long.csv", replace
restore

*=============================================================================
* Part 4. Concise public audit log (no machine path, timestamp, or table output)
*=============================================================================

capture file close audit
file open audit using "stata/robustness_analysis.log", write replace
file write audit "Robustness analysis audit log" _n
file write audit "Data: data_anonymized/product_master.dta" _n
file write audit "Unit of analysis: product" _n
file write audit "Products: `n_all'" _n
file write audit "Products with n_reviews_total >= 20: `n_20'" _n
file write audit "Products with n_reviews_total >= 50: `n_50'" _n
file write audit "Products with n_reviews_total >= 100: `n_100'" _n
file write audit "Indirect associations: 14" _n
file write audit "Bootstrap replications: 5000" _n
file write audit "Bootstrap seed: 42" _n
file write audit "Bootstrap interval: BCa 95%" _n
file write audit "Threshold regressions: n_reviews_total >= 50 and >= 100" _n
file write audit "Weighted regression: analytic weight = n_reviews_total (proxy only)" _n
file write audit "Regression standard errors: heteroskedasticity robust" _n
file write audit "Status: completed without error" _n
file close audit

display as result "Robustness analyses completed."
