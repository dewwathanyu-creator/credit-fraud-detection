import streamlit as st
import pandas as pd
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💳 Fraud Detection App")
st.markdown(
    "Enter transaction details below and click **Predict** to check whether "
    "the transaction is likely fraudulent."
)
st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "DEPOSIT"],
        help="Fraud only occurs in TRANSFER and CASH_OUT transactions."
    )
    amount = st.number_input(
        "Transaction Amount", min_value=0.0, value=10_000.0, step=100.0,
        format="%.2f"
    )
    oldbalanceOrg = st.number_input(
        "Sender — Balance Before", min_value=0.0, value=50_000.0, step=100.0,
        format="%.2f"
    )
    newbalanceOrig = st.number_input(
        "Sender — Balance After", min_value=0.0, value=40_000.0, step=100.0,
        format="%.2f"
    )

with col2:
    st.markdown("&nbsp;")   # vertical spacing
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    st.markdown("&nbsp;")
    oldbalanceDest = st.number_input(
        "Receiver — Balance Before", min_value=0.0, value=0.0, step=100.0,
        format="%.2f"
    )
    newbalanceDest = st.number_input(
        "Receiver — Balance After", min_value=0.0, value=0.0, step=100.0,
        format="%.2f"
    )

st.divider()

# ── Derived features (must match notebook engineering) ─────────────────────────
balance_diff_orig    = oldbalanceOrg - newbalanceOrig
balance_diff_dest    = newbalanceDest - oldbalanceDest
is_account_drained   = int(oldbalanceOrg > 0 and newbalanceOrig == 0)

with st.expander("Computed features (auto-calculated)"):
    st.write({
        "balanceDiffOrig":  round(balance_diff_orig, 2),
        "balanceDiffDest":  round(balance_diff_dest, 2),
        "isAccountDrained": is_account_drained,
    })

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "type":             transaction_type,
        "amount":           amount,
        "oldbalanceOrg":    oldbalanceOrg,
        "newbalanceOrig":   newbalanceOrig,
        "oldbalanceDest":   oldbalanceDest,
        "newbalanceDest":   newbalanceDest,
        "balanceDiffOrig":  balance_diff_orig,
        "balanceDiffDest":  balance_diff_dest,
        "isAccountDrained": is_account_drained,
    }])

    prediction = model.predict(input_data)[0]
    proba      = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **This transaction is likely FRAUDULENT**  \n"
            f"Fraud probability: **{proba * 100:.1f}%**"
        )
    else:
        st.success(
            f"✅ **This transaction appears LEGITIMATE**  \n"
            f"Fraud probability: **{proba * 100:.1f}%**"
        )

    st.progress(float(proba), text=f"Fraud risk: {proba * 100:.1f}%")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Random Forest trained on 6.3M financial transactions · "
    "Metric optimised for high Recall (minimise missed fraud)"
)
