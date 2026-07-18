/*=============================================================================
 main_analysis.do
 -----------------------------------------------------------------------------
 Reproduces the regression tables of the paper.

 Primary outcome: y_favrate_only (platform favorable rate, logit-transformed,
   z-scored) — source-separated from the text-extracted mediators.

 Controls (pre-specified conceptual blocks):
   Block A (retail conditions):     ln(price), ln(reviews), has_discount,
                                    free_shipping
   Block B (seller characteristics): ln_imperfection
   Block C (brand positioning):     ln(brand average price)
   Block D (review timing):         recency_inv

 Run from the repository root:
   do stata/main_analysis.do

 Data:  data_anonymized/product_master.dta  (N = 200 products)
 Output: stata/output/
=============================================================================*/

version 18
clear all
set more off
set seed 42
capture log close
capture mkdir "stata/output"
log using "stata/main_analysis.log", replace text


*==============================================================================
* Part 0. Load data and standardize
*==============================================================================
use "data_anonymized/product_master.dta", clear
display "Loaded product_master.dta. N = " _N

* Control variables (ln transforms)
capture drop lnprice lnreviews
gen lnprice   = ln(price + 1)
gen lnreviews = ln(n_reviews_total + 1)

* Main-analysis subsample flag
capture drop included_main
gen byte included_main = (n_reviews_total >= 20)

* z-score standardize all regression variables
foreach v in y_50_50 y_30_70 y_70_30 y_sentiment_only y_favrate_only ///
             m_e_avg m_f_avg ///
             z_pc1 z_pc2 z_pc3 z_pc4 z_pc5 z_pc6 z_pc7 ///
             me1_z me2_z me3_z me4_z me5_z ///
             mf1_z mf2_z mf3_z mf4_z mf5_z mf6_z mf7_z ///
             lnprice lnreviews ///
             has_discount free_shipping ///
             ln_imperfection ln_brand_avg_price recency_inv {
    capture confirm variable `v'
    if !_rc {
        capture drop `v'_s
        egen `v'_s = std(`v')
    }
}

label var y_50_50_s              "Y satisfaction (50:50)"
label var y_30_70_s              "Y satisfaction (30:70)"
label var y_70_30_s              "Y satisfaction (70:30)"
label var y_sentiment_only_s     "Y sentiment only"
label var y_favrate_only_s       "Y favorable-rate only (PRIMARY)"
label var m_e_avg_s              "M_E Kansei evaluation"
label var m_f_avg_s              "M_F functional evaluation"
label var z_pc1_s "PC1 Minimalist Lightness"
label var z_pc2_s "PC2 Refined Tech-Modern"
label var z_pc3_s "PC3 Non-standard Premium Form"
label var z_pc4_s "PC4 Heavy Rack Architecture"
label var z_pc5_s "PC5 Dark Rough Texture"
label var z_pc6_s "PC6 Leg-form Variation"
label var z_pc7_s "PC7 Light-Wood Accessory Cues"
label var lnprice_s              "ln(Price)"
label var lnreviews_s            "ln(N reviews)"
label var has_discount_s         "Has discount (1=yes)"
label var free_shipping_s        "Free shipping (1=yes)"
label var ln_imperfection_s      "ln(Reputation gap)"
label var ln_brand_avg_price_s   "ln(Brand avg price)"
label var recency_inv_s          "Review recency (inverse)"

capture drop id
egen id = group(product_id)

* NOTE: the standardized dataset is not saved back to disk.

*==============================================================================
* Shared globals
*==============================================================================
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"


*==============================================================================
* Part 1. Descriptive statistics, correlations, and VIF
*==============================================================================
estpost tabstat y_favrate_only y_50_50 y_sentiment_only m_e_avg m_f_avg ///
    z_pc1 z_pc2 z_pc3 z_pc4 z_pc5 z_pc6 z_pc7 ///
    price n_reviews_total has_discount free_shipping ///
    ln_imperfection ln_brand_avg_price recency_inv, ///
    stats(N mean sd min p50 max) columns(statistics)
esttab using "stata/output/T1_descriptive.rtf", replace ///
    cells("count(fmt(0)) mean(fmt(3)) sd(fmt(3)) min(fmt(3)) p50(fmt(3)) max(fmt(3))") ///
    nonumber nomtitle nonote label ///
    title("Table 1. Descriptive Statistics (N=200)")

estpost correlate y_favrate_only_s m_e_avg_s m_f_avg_s ///
    z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s ///
    lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s, matrix
esttab using "stata/output/T1B_correlations.rtf", replace ///
    unstack not noobs nonumber compress label ///
    title("Table 1B. Pearson Correlations (standardized)") b(%6.3f)

* VIF
reg y_favrate_only_s z_pc1_s-z_pc7_s m_e_avg_s m_f_avg_s $CTRL
estat vif


*==============================================================================
* Part 2. Main parallel mediation (4 columns)
*==============================================================================
reg $ME $X $CTRL, vce(robust)
est store m1_aE

reg $MF $X $CTRL, vce(robust)
est store m2_aF

reg $Y $X $CTRL, vce(robust)
est store m3_total

reg $Y $X $ME $MF $CTRL, vce(robust)
est store m4_full

esttab m1_aE m2_aF m3_total m4_full ///
    using "stata/output/T3_main_mediation.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    mtitles("M_E (a-path)" "M_F (a-path)" "Y total (c)" "Y full (c'+b)") ///
    title("Table 3. Main Parallel Mediation (N=200)") ///
    addnote("Primary outcome: platform favorable rate. Full control specification. Robust SE." ///
            "*** p<0.01, ** p<0.05, * p<0.10")


*==============================================================================
* Part 3. Bootstrap indirect effects (5000 reps)
*==============================================================================
capture program drop med_parallel
program med_parallel, rclass
    sem ($ME <- $X $CTRL) ///
        ($MF <- $X $CTRL) ///
        ($Y  <- $X $ME $MF $CTRL)

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

    scalar tE = abs(aE1*bE)+abs(aE2*bE)+abs(aE3*bE)+abs(aE4*bE)+abs(aE5*bE)+abs(aE6*bE)+abs(aE7*bE)
    scalar tF = abs(aF1*bF)+abs(aF2*bF)+abs(aF3*bF)+abs(aF4*bF)+abs(aF5*bF)+abs(aF6*bF)+abs(aF7*bF)
    return scalar totE_abs = tE
    return scalar totF_abs = tF
    return scalar diff_EF  = tE - tF
    return scalar bE = bE
    return scalar bF = bF
end

bootstrap ///
    indE_pc1=r(indE_pc1) indE_pc2=r(indE_pc2) indE_pc3=r(indE_pc3) ///
    indE_pc4=r(indE_pc4) indE_pc5=r(indE_pc5) indE_pc6=r(indE_pc6) indE_pc7=r(indE_pc7) ///
    indF_pc1=r(indF_pc1) indF_pc2=r(indF_pc2) indF_pc3=r(indF_pc3) ///
    indF_pc4=r(indF_pc4) indF_pc5=r(indF_pc5) indF_pc6=r(indF_pc6) indF_pc7=r(indF_pc7) ///
    bE=r(bE) bF=r(bF) ///
    totE_abs=r(totE_abs) totF_abs=r(totF_abs) diff_EF=r(diff_EF), ///
    reps(5000) seed(42): med_parallel
est store boot_indirect

esttab boot_indirect using "stata/output/T4_indirect_effects.rtf", replace ///
    b(%7.4f) ci(%7.4f) star(* 0.10 ** 0.05 *** 0.01) ///
    title("Table 4. Indirect Effects (5000 Bootstrap, 95% CI)") ///
    label nonotes


*==============================================================================
* Part 4. Serial mediation Path A: Visual -> M_E -> M_F -> Y
*==============================================================================
capture program drop serial_A
program serial_A, rclass
    sem ($ME <- $X $CTRL) ///
        ($MF <- $X $ME $CTRL) ///
        ($Y  <- $X $ME $MF $CTRL)
    scalar d_EF = _b[$MF:$ME]
    scalar bF_A = _b[$Y:$MF]
    scalar bE_A = _b[$Y:$ME]
    forvalues i = 1/7 {
        scalar aE`i' = _b[$ME:z_pc`i'_s]
        return scalar serialA_pc`i' = aE`i' * d_EF * bF_A
    }
    return scalar d_EF = d_EF
    return scalar bE_serialA = bE_A
    return scalar bF_serialA = bF_A
    return scalar serialA_total = (aE1+aE2+aE3+aE4+aE5+aE6+aE7) * d_EF * bF_A
end

bootstrap ///
    serialA_pc1=r(serialA_pc1) serialA_pc2=r(serialA_pc2) serialA_pc3=r(serialA_pc3) ///
    serialA_pc4=r(serialA_pc4) serialA_pc5=r(serialA_pc5) serialA_pc6=r(serialA_pc6) ///
    serialA_pc7=r(serialA_pc7) ///
    d_EF=r(d_EF) bE_serialA=r(bE_serialA) bF_serialA=r(bF_serialA) ///
    serialA_total=r(serialA_total), ///
    reps(2000) seed(42): serial_A
est store boot_serialA

esttab boot_serialA using "stata/output/T5A_serial_path_A.rtf", replace ///
    b(%7.4f) ci(%7.4f) star(* 0.10 ** 0.05 *** 0.01) ///
    title("Table 5A. Serial Path A: Visual -> M_E -> M_F -> Y") ///
    label nonotes


*==============================================================================
* Part 5. Serial mediation Path B: Visual -> M_F -> M_E -> Y
*==============================================================================
capture program drop serial_B
program serial_B, rclass
    sem ($MF <- $X $CTRL) ///
        ($ME <- $X $MF $CTRL) ///
        ($Y  <- $X $ME $MF $CTRL)
    scalar d_FE = _b[$ME:$MF]
    scalar bE_B = _b[$Y:$ME]
    scalar bF_B = _b[$Y:$MF]
    forvalues i = 1/7 {
        scalar aF`i' = _b[$MF:z_pc`i'_s]
        return scalar serialB_pc`i' = aF`i' * d_FE * bE_B
    }
    return scalar d_FE = d_FE
    return scalar bE_serialB = bE_B
    return scalar bF_serialB = bF_B
    return scalar serialB_total = (aF1+aF2+aF3+aF4+aF5+aF6+aF7) * d_FE * bE_B
end

bootstrap ///
    serialB_pc1=r(serialB_pc1) serialB_pc2=r(serialB_pc2) serialB_pc3=r(serialB_pc3) ///
    serialB_pc4=r(serialB_pc4) serialB_pc5=r(serialB_pc5) serialB_pc6=r(serialB_pc6) ///
    serialB_pc7=r(serialB_pc7) ///
    d_FE=r(d_FE) bE_serialB=r(bE_serialB) bF_serialB=r(bF_serialB) ///
    serialB_total=r(serialB_total), ///
    reps(2000) seed(42): serial_B
est store boot_serialB

esttab boot_serialB using "stata/output/T5B_serial_path_B.rtf", replace ///
    b(%7.4f) ci(%7.4f) star(* 0.10 ** 0.05 *** 0.01) ///
    title("Table 5B. Serial Path B: Visual -> M_F -> M_E -> Y") ///
    label nonotes


*==============================================================================
* Part 6. Dimensional decomposition (12 mediators)
*==============================================================================
reg $Y $X ///
    me1_z_s me2_z_s me3_z_s me4_z_s me5_z_s ///
    mf1_z_s mf2_z_s mf3_z_s mf4_z_s mf5_z_s mf6_z_s mf7_z_s ///
    $CTRL, vce(robust)
est store m_decomp

esttab m_decomp using "stata/output/T6_decomposition.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    title("Table 6. Dimensional Decomposition (12 mediators, N=200)") nonotes

* Export b-path p-values for Python FDR correction
capture file close fh
file open fh using "stata/output/T6_decomp_pvalues.csv", write replace
file write fh "mediator,b,se,t,p_raw" _n
foreach m in me1_z_s me2_z_s me3_z_s me4_z_s me5_z_s ///
              mf1_z_s mf2_z_s mf3_z_s mf4_z_s mf5_z_s mf6_z_s mf7_z_s {
    scalar bv = _b[`m']
    scalar sv = _se[`m']
    scalar tv = bv/sv
    scalar pv = 2*(1-normal(abs(tv)))
    file write fh "`m'," %9.6f (bv) "," %9.6f (sv) "," %9.6f (tv) "," %9.6f (pv) _n
}
file close fh


*==============================================================================
* Part 7. Robustness: N=185 subsample (>= 20 reviews)
*==============================================================================
preserve
    keep if included_main == 1
    display "Subsample N = " _N

    reg $ME $X $CTRL, vce(robust)
    est store r1
    reg $MF $X $CTRL, vce(robust)
    est store r2
    reg $Y $X $CTRL, vce(robust)
    est store r3
    reg $Y $X $ME $MF $CTRL, vce(robust)
    est store r4

    esttab r1 r2 r3 r4 ///
        using "stata/output/T7_robust_N185.rtf", replace ///
        b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
        scalar(N r2 r2_a F) compress nogaps label ///
        mtitles("M_E" "M_F" "Y total" "Y full") ///
        title("Table 7. Robustness: N=185 (n_reviews>=20)")
restore


*==============================================================================
* Part 8. Robustness: alternative Y operationalizations
*==============================================================================
reg y_favrate_only_s   $X $ME $MF $CTRL, vce(robust)
est store y1
reg y_50_50_s          $X $ME $MF $CTRL, vce(robust)
est store y2
reg y_30_70_s          $X $ME $MF $CTRL, vce(robust)
est store y3
reg y_70_30_s          $X $ME $MF $CTRL, vce(robust)
est store y4
reg y_sentiment_only_s $X $ME $MF $CTRL, vce(robust)
est store y5

esttab y1 y2 y3 y4 y5 ///
    using "stata/output/T8_robust_Y_variants.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    drop($X $CTRL _cons) ///
    mtitles("Y favrate (PRIMARY)" "Y 50:50" "Y 30:70" "Y 70:30" "Y sentiment") ///
    title("Table 8. Robustness: Alternative Y Operationalizations") ///
    addnote("Col 1 = primary source-separated outcome. Cols 2-5 = robustness alternatives.")


*==============================================================================
* Wrap up
*==============================================================================
log close
display ""
display "=========================================="
display "  Done. All outputs in stata/output/"
display "=========================================="
display "Next: python scripts_stata_prep/fdr_correction.py"
