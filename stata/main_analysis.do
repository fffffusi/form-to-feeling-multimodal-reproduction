/*=============================================================================
 main_analysis.do
 -----------------------------------------------------------------------------
 Full mediation analysis with platform-favorable-rate as the primary
 satisfaction outcome (source-separated from the text-extracted mediators).

 The primary outcome is constructed solely from the platform favorable-rate
 measure (logit-transformed), so it does not share a text source with M_E,
 M_F, or the LLM sentiment measure.

 Controls (pre-specified conceptual blocks):
   Block A (retail conditions):     ln(price), ln(reviews), has_discount,
                                    free_shipping
   Block B (seller characteristics): ln_imperfection
   Block C (brand positioning):     ln(brand average price)
   Block D (review timing):         recency_inv

 Primary outcome: y_favrate_only_s
 Data: data_anonymized/product_master.dta
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
* Part 0. 加载 + 标准化
*==============================================================================
use "data_anonymized/product_master.dta", clear
display "Loaded data_anonymized/product_master.dta. N = " _N

* 控制变量 (ln transforms)
capture drop lnprice lnreviews
gen lnprice   = ln(price + 1)
gen lnreviews = ln(n_reviews_total + 1)

* 主分析样本标志
capture drop included_main
gen included_main = (n_reviews_total >= 20)

* z-score 标准化所有变量
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

* 变量标签
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

tempfile analysis_data
save `analysis_data', replace


*==============================================================================
* Part 1. 描述统计 + 相关矩阵 + VIF
*==============================================================================
use `analysis_data', clear

** 1.1 描述统计
estpost tabstat y_favrate_only y_50_50 y_sentiment_only m_e_avg m_f_avg ///
    z_pc1 z_pc2 z_pc3 z_pc4 z_pc5 z_pc6 z_pc7 ///
    price n_reviews_total has_discount free_shipping ///
    ln_imperfection ln_brand_avg_price recency_inv, ///
    stats(N mean sd min p50 max) columns(statistics)
esttab using "stata/output/T1_descriptive.rtf", replace ///
    cells("count(fmt(0)) mean(fmt(3)) sd(fmt(3)) min(fmt(3)) p50(fmt(3)) max(fmt(3))") ///
    nonumber nomtitle nonote label ///
    title("Table 1. Descriptive Statistics (N=200, primary Y = favorable-rate only)")

** 1.2 相关矩阵 (主Y = y_favrate_only_s)
estpost correlate y_favrate_only_s y_50_50_s y_sentiment_only_s ///
    m_e_avg_s m_f_avg_s ///
    z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s ///
    lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s, matrix
esttab using "stata/output/T1B_correlations.rtf", replace ///
    unstack not noobs nonumber compress label ///
    title("Table 1B. Pearson Correlations (standardized; Y = favorable-rate only)") b(%6.3f)

** 1.3 VIF 多重共线性诊断 (3 个关键规格 + 导出)
*    阈值: VIF < 5 良好, 5-10 可接受, > 10 严重共线性需处理
*    检查规格:
*      (a) 仅控制变量 — 检查新增控制变量之间是否共线
*      (b) 全中介模型 (PCs + ME + MF + 控制变量) — 主回归的共线性
*      (c) 维度分解 (7 PCs + 12 mediators + 控制变量) — 共线性最严重的规格

display ""
display "========================================"
display "  VIF Diagnostics (Y = favorable-rate only)"
display "========================================"

* --- 导出 VIF 结果到 CSV ---
capture file close vif_fh
file open vif_fh using "stata/output/VIF_diagnostics.csv", write replace
file write vif_fh "spec,variable,vif,1/vif,flag" _n

* --- (a) 仅 7 个控制变量 ---
display ""
display "--- Spec (a): Controls only ---"
reg y_favrate_only_s lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s
estat vif
* 捕获 VIF 矩阵写入 CSV
matrix VA = r(table)
* estat vif 不直接返回简洁矩阵, 用循环逐变量计算并写文件
local controls_vif lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s
foreach v of local controls_vif {
    local rhs : list controls_vif - v
    quietly reg `v' `rhs'
    scalar _vif_val = 1 / (1 - e(r2))
    scalar _vif_inv = 1 - e(r2)
    local _flag = ""
    if _vif_val >= 10 local _flag = "SEVERE(>=10)"
    else if _vif_val >= 5 local _flag = "elevated(5-10)"
    file write vif_fh "(a)controls,`v'," %9.4f (_vif_val) "," %9.4f (_vif_inv) ",`_flag'" _n
}

* --- (b) 全中介模型 (主回归 Table 5 column 4) ---
display ""
display "--- Spec (b): Full mediation model (PCs + ME + MF + controls) ---"
reg y_favrate_only_s z_pc1_s-z_pc7_s m_e_avg_s m_f_avg_s ///
    lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s
estat vif
local full_vif z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s ///
    m_e_avg_s m_f_avg_s lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s
foreach v of local full_vif {
    local rhs : list full_vif - v
    quietly reg `v' `rhs'
    scalar _vif_val = 1 / (1 - e(r2))
    scalar _vif_inv = 1 - e(r2)
    local _flag = ""
    if _vif_val >= 10 local _flag = "SEVERE(>=10)"
    else if _vif_val >= 5 local _flag = "elevated(5-10)"
    file write vif_fh "(b)full_mediation,`v'," %9.4f (_vif_val) "," %9.4f (_vif_inv) ",`_flag'" _n
}

* --- (c) 维度分解 (Table 7, 共线性最严重: 7PCs + 12mediators + 7controls) ---
display ""
display "--- Spec (c): Dimensional decomposition (26 predictors) ---"
reg y_favrate_only_s z_pc1_s-z_pc7_s ///
    me1_z_s me2_z_s me3_z_s me4_z_s me5_z_s ///
    mf1_z_s mf2_z_s mf3_z_s mf4_z_s mf5_z_s mf6_z_s mf7_z_s ///
    lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s
estat vif
local decomp_vif z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s ///
    me1_z_s me2_z_s me3_z_s me4_z_s me5_z_s ///
    mf1_z_s mf2_z_s mf3_z_s mf4_z_s mf5_z_s mf6_z_s mf7_z_s ///
    lnprice_s lnreviews_s has_discount_s free_shipping_s ///
    ln_imperfection_s ln_brand_avg_price_s recency_inv_s
foreach v of local decomp_vif {
    local rhs : list decomp_vif - v
    quietly reg `v' `rhs'
    scalar _vif_val = 1 / (1 - e(r2))
    scalar _vif_inv = 1 - e(r2)
    local _flag = ""
    if _vif_val >= 10 local _flag = "SEVERE(>=10)"
    else if _vif_val >= 5 local _flag = "elevated(5-10)"
    file write vif_fh "(c)decomposition,`v'," %9.4f (_vif_val) "," %9.4f (_vif_inv) ",`_flag'" _n
}

file close vif_fh
display ""
display "VIF 表已导出: stata/output/VIF_diagnostics.csv"
display "阈值参考: VIF<5 良好, 5-10 可接受, >=10 严重"


*==============================================================================
* Part 2. 主中介 4 列回归 (Y = y_favrate_only_s)
*==============================================================================
use `analysis_data', clear

global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

* (1) X -> M_E
reg $ME $X $CTRL, vce(robust)
est store m1_aE

* (2) X -> M_F
reg $MF $X $CTRL, vce(robust)
est store m2_aF

* (3) X -> Y (总效应 c)
reg $Y $X $CTRL, vce(robust)
est store m3_total

* (4) X + M_E + M_F -> Y (满模型 c'+b)
reg $Y $X $ME $MF $CTRL, vce(robust)
est store m4_full

esttab m1_aE m2_aF m3_total m4_full ///
    using "stata/output/T3_main_mediation.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    mtitles("M_E (a-path)" "M_F (a-path)" "Y total (c)" "Y full (c'+b)") ///
    title("Table 3. Main Parallel Mediation – Y = favorable-rate only (N=200)") ///
    addnote("Primary outcome: platform favorable rate (logit-transformed, z-scored)." ///
            "The outcome does not share a text source with the extracted evaluations." ///
            "Full control specification. Robust SE in parentheses." ///
            "*** p<0.01, ** p<0.05, * p<0.10")


*==============================================================================
* Part 3. Parallel Bootstrap — 14 indirect + 总效应 + 路径强弱比较
*==============================================================================
use `analysis_data', clear
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

capture program drop med_parallel
program med_parallel, rclass
    syntax [if] [in]
    marksample touse
    sem ($ME <- $X $CTRL) ///
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
    reps(5000) seed(42) bca ties ///
    saving("stata/output/bootstrap_parallel.dta", replace): med_parallel
est store boot_parallel

* Report genuine percentile and BCa confidence intervals.
* Do not use esttab, ci() here: it reports the default normal-based interval.
log using "stata/output/T4_indirect_effects.txt", ///
    name(bootci_parallel) replace text
estat bootstrap, percentile bca
log close bootci_parallel


*==============================================================================
* Part 4. Serial Mediation Path A: Visual -> M_E -> M_F -> Y
*==============================================================================
use `analysis_data', clear
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

capture program drop serialA
program serialA, rclass
    syntax [if] [in]
    marksample touse
    sem ($ME <- $X $CTRL) ///
        ($MF <- $X $ME $CTRL) ///
        ($Y  <- $X $ME $MF $CTRL) if `touse'

    scalar d_EF  = _b[$MF:$ME]
    scalar bF_A  = _b[$Y:$MF]
    scalar bE_A  = _b[$Y:$ME]

    forvalues i = 1/7 {
        scalar aE`i' = _b[$ME:z_pc`i'_s]
        return scalar serialA_pc`i' = aE`i' * d_EF * bF_A
    }

    return scalar d_EF       = d_EF
    return scalar bF_serialA = bF_A
    return scalar bE_serialA = bE_A
    return scalar serialA_total = (aE1+aE2+aE3+aE4+aE5+aE6+aE7) * d_EF * bF_A
end

bootstrap ///
    serialA_pc1=r(serialA_pc1) serialA_pc2=r(serialA_pc2) serialA_pc3=r(serialA_pc3) ///
    serialA_pc4=r(serialA_pc4) serialA_pc5=r(serialA_pc5) serialA_pc6=r(serialA_pc6) ///
    serialA_pc7=r(serialA_pc7) ///
    d_EF=r(d_EF) bE_serialA=r(bE_serialA) bF_serialA=r(bF_serialA) ///
    serialA_total=r(serialA_total), ///
    reps(2000) seed(42) bca ties ///
    saving("stata/output/bootstrap_serialA.dta", replace): serialA
est store boot_serialA

log using "stata/output/T5A_serial_path_A.txt", ///
    name(bootci_serialA) replace text
estat bootstrap, percentile bca
log close bootci_serialA


*==============================================================================
* Part 5. Serial Mediation Path B: Visual -> M_F -> M_E -> Y
*==============================================================================
use `analysis_data', clear
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

capture program drop serialB
program serialB, rclass
    syntax [if] [in]
    marksample touse
    sem ($MF <- $X $CTRL) ///
        ($ME <- $X $MF $CTRL) ///
        ($Y  <- $X $ME $MF $CTRL) if `touse'

    scalar d_FE  = _b[$ME:$MF]
    scalar bE_B  = _b[$Y:$ME]
    scalar bF_B  = _b[$Y:$MF]

    forvalues i = 1/7 {
        scalar aF`i' = _b[$MF:z_pc`i'_s]
        return scalar serialB_pc`i' = aF`i' * d_FE * bE_B
    }

    return scalar d_FE       = d_FE
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
    reps(2000) seed(42) bca ties ///
    saving("stata/output/bootstrap_serialB.dta", replace): serialB
est store boot_serialB

log using "stata/output/T5B_serial_path_B.txt", ///
    name(bootci_serialB) replace text
estat bootstrap, percentile bca
log close bootci_serialB


*==============================================================================
* Part 6. 分维度探索 (12 mediators) + FDR 校正所需 p 值
*==============================================================================
use `analysis_data', clear
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

reg y_favrate_only_s $X ///
    me1_z_s me2_z_s me3_z_s me4_z_s me5_z_s ///
    mf1_z_s mf2_z_s mf3_z_s mf4_z_s mf5_z_s mf6_z_s mf7_z_s ///
    $CTRL, vce(robust)
est store m_decomp

esttab m_decomp using "stata/output/T6_decomposition.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    title("Table 6. Dimensional Decomposition — Y = favorable-rate only (12 mediators, N=200)") ///
    nonotes

* 导出 p 值供 Python FDR 校正
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
display "P-values exported to stata/output/T6_decomp_pvalues.csv"


*==============================================================================
* Part 7. 稳健性 N=185 (n_reviews >= 20)
*==============================================================================
use `analysis_data', clear
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global Y    "y_favrate_only_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

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

esttab r1 r2 r3 r4 using "stata/output/T7_robust_N185.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    mtitles("M_E" "M_F" "Y total" "Y full") ///
    title("Table 7. Robustness: Subsample N=185 (n_reviews>=20), Y=favorable-rate only") nonotes


*==============================================================================
* Part 8. 稳健性 — 5 种 Y 变体 (现在把 composite 等作为稳健性对照)
*==============================================================================
use `analysis_data', clear
global X    "z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s"
global ME   "m_e_avg_s"
global MF   "m_f_avg_s"
global CTRL "lnprice_s lnreviews_s has_discount_s free_shipping_s ln_imperfection_s ln_brand_avg_price_s recency_inv_s"

* Col 1: PRIMARY (favorable-rate only)
reg y_favrate_only_s   $X $ME $MF $CTRL, vce(robust)
est store y1
* Col 2-5: alternative operationalizations (now robustness)
reg y_50_50_s          $X $ME $MF $CTRL, vce(robust)
est store y2
reg y_30_70_s          $X $ME $MF $CTRL, vce(robust)
est store y3
reg y_70_30_s          $X $ME $MF $CTRL, vce(robust)
est store y4
reg y_sentiment_only_s $X $ME $MF $CTRL, vce(robust)
est store y5

esttab y1 y2 y3 y4 y5 using "stata/output/T8_robust_Y_variants.rtf", replace ///
    b(%6.3f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) ///
    scalar(N r2 r2_a F) compress nogaps label ///
    drop(z_pc1_s z_pc2_s z_pc3_s z_pc4_s z_pc5_s z_pc6_s z_pc7_s ///
         lnprice_s lnreviews_s has_discount_s free_shipping_s ///
         ln_imperfection_s ln_brand_avg_price_s recency_inv_s _cons) ///
    mtitles("Y favrate (PRIMARY)" "Y 50:50" "Y 30:70" "Y 70:30" "Y sentiment") ///
    title("Table 8. Alternative Y operationalizations (full controls, N=200)") ///
    addnote("Col 1 = primary source-separated outcome. Cols 2-5 = alternative operationalizations for robustness." ///
            "Checks if M_E (+) and M_F (-) signs hold across different Y definitions.")


*==============================================================================
* 收尾
*==============================================================================
log close
display ""
display "============================================================"
display "  Analysis complete (Y = favorable-rate only). Tables in stata/output/"
display "============================================================"
display ""
display "Primary Y: y_favrate_only_s (platform favorable rate, source-separated)"
display "Full control set (7 variables, pre-specified blocks)."
display ""
display "Tables:"
display "  T1_descriptive.rtf               Descriptive statistics"
display "  T1B_correlations.rtf             Correlation matrix"
display "  VIF_diagnostics.csv              VIF multicollinearity diagnostics (3 specs)"
display "  T3_main_mediation.rtf       Main mediation (4 columns)"
display "  T4_indirect_effects.txt     Percentile and BCa indirect associations"
display "  T5A_serial_path_A.txt       Serial Path A: X->ME->MF->Y"
display "  T5B_serial_path_B.txt       Serial Path B: X->MF->ME->Y"
display "  T6_decomposition.rtf        Dimensional decomposition (12 mediators)"
display "  T6_decomp_pvalues.csv       Raw p-values for FDR"
display "  T7_robust_N185.rtf          N=185 subsample"
display "  T8_robust_Y_variants.rtf    5 Y variants (favrate = primary)"
