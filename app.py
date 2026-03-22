import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Marketing Funnel Dashboard", layout="wide")

st.title("📊 Marketing Funnel & Conversion Analysis")
st.write("Upload the bank marketing dataset to analyze funnel drop-offs, conversion performance, and prediction.")

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #f7f7f9;
    border: 1px solid #e6e6e6;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(uploaded_file, sep=";")
    df.columns = df.columns.str.strip()

    # Clean important text columns
    text_cols = ["job", "marital", "education", "default", "housing", "loan",
                 "contact", "month", "day_of_week", "poutcome", "y"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    st.markdown("## 📂 Raw Data")
    st.dataframe(df.head(20), use_container_width=True)

    # -----------------------------
    # Sidebar filters
    # -----------------------------
    st.sidebar.header("🔍 Filters")

    job_options = sorted(df["job"].dropna().unique().tolist())
    contact_options = sorted(df["contact"].dropna().unique().tolist())
    month_options = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    month_present = [m for m in month_options if m in df["month"].unique()]

    job_filter = st.sidebar.multiselect("Select Job", options=job_options, default=job_options)
    contact_filter = st.sidebar.multiselect("Select Contact Type", options=contact_options, default=contact_options)
    month_filter = st.sidebar.multiselect("Select Month", options=month_present, default=month_present)

    df_filtered = df[
        (df["job"].isin(job_filter)) &
        (df["contact"].isin(contact_filter)) &
        (df["month"].isin(month_filter))
    ].copy()

    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # -----------------------------
    # Funnel metrics
    # -----------------------------
    total_contacted = len(df_filtered)
    engaged = (df_filtered["duration"] > 0).sum()
    interested = (df_filtered["duration"] > 100).sum()
    high_intent = ((df_filtered["duration"] > 100) & (df_filtered["campaign"] < 3)).sum()
    converted = (df_filtered["y"] == "yes").sum()

    overall_conversion = converted / total_contacted if total_contacted else 0

    funnel_df = pd.DataFrame({
        "Stage": ["Contacted", "Engaged", "Interested", "High Intent", "Converted"],
        "Count": [total_contacted, engaged, interested, high_intent, converted]
    })

    # -----------------------------
    # KPI cards
    # -----------------------------
    st.markdown("## 📌 Key Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 Total Contacted", f"{total_contacted:,}")
    k2.metric("💰 Converted", f"{converted:,}")
    k3.metric("📈 Conversion Rate", f"{overall_conversion:.2%}")
    k4.metric("📞 Avg Campaign", f"{df_filtered['campaign'].mean():.2f}")

    # -----------------------------
    # Funnel and drop-off
    # -----------------------------
    st.markdown("## 📊 Funnel & Drop-off")

    fig_funnel = px.funnel(
        funnel_df,
        x="Count",
        y="Stage",
        title="Behavioral Funnel"
    )

    drop_off = {
        "Contacted→Engaged": 1 - (engaged / total_contacted if total_contacted else 0),
        "Engaged→Interested": 1 - (interested / engaged if engaged else 0),
        "Interested→High Intent": 1 - (high_intent / interested if interested else 0),
        "High Intent→Converted": 1 - (converted / high_intent if high_intent else 0),
    }

    drop_df = pd.DataFrame({
        "Stage": list(drop_off.keys()),
        "Drop-off Rate": list(drop_off.values())
    }).sort_values("Drop-off Rate", ascending=False)

    fig_drop = px.bar(
        drop_df,
        x="Stage",
        y="Drop-off Rate",
        title="Stage-wise Drop-off Rate"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_funnel, use_container_width=True)
        st.dataframe(funnel_df, use_container_width=True)
    with col2:
        st.plotly_chart(fig_drop, use_container_width=True)
        st.dataframe(drop_df, use_container_width=True)

    worst_stage = drop_df.iloc[0]["Stage"]

    # -----------------------------
    # Channel performance
    # -----------------------------
    channel_perf = (
        df_filtered.groupby("contact")["y"]
        .apply(lambda x: (x == "yes").mean())
        .reset_index(name="Conversion Rate")
        .sort_values("Conversion Rate", ascending=False)
    )

    fig_channel = px.bar(
        channel_perf,
        x="contact",
        y="Conversion Rate",
        title="Conversion Rate by Contact Type"
    )

    # -----------------------------
    # Month performance
    # -----------------------------
    month_perf = (
        df_filtered.groupby("month")["y"]
        .apply(lambda x: (x == "yes").mean())
        .reset_index(name="Conversion Rate")
    )
    month_perf["month"] = pd.Categorical(month_perf["month"], categories=month_options, ordered=True)
    month_perf = month_perf.sort_values("month")

    fig_month = px.line(
        month_perf,
        x="month",
        y="Conversion Rate",
        markers=True,
        title="Conversion Rate by Month"
    )

    st.markdown("## 📊 Performance Insights")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_channel, use_container_width=True)
        st.dataframe(channel_perf, use_container_width=True)
    with col4:
        st.plotly_chart(fig_month, use_container_width=True)
        st.dataframe(month_perf, use_container_width=True)

    best_channel = channel_perf.iloc[0]["contact"]

    # -----------------------------
    # Job performance
    # -----------------------------
    job_perf = (
        df_filtered.groupby("job")["y"]
        .apply(lambda x: (x == "yes").mean())
        .reset_index(name="Conversion Rate")
        .sort_values("Conversion Rate", ascending=False)
    )

    fig_job = px.bar(
        job_perf,
        x="job",
        y="Conversion Rate",
        title="Conversion Rate by Job Segment"
    )

    # -----------------------------
    # ML model
    # -----------------------------
    df_ml = df_filtered.copy()
    df_ml["y"] = df_ml["y"].map({"yes": 1, "no": 0})

    X = pd.get_dummies(df_ml.drop("y", axis=1), drop_first=True)
    y = df_ml["y"]

    # Check class availability after filtering
    if y.nunique() < 2:
        st.warning("The selected filters leave only one class in target 'y'. Adjust filters to run the ML model.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        random_state=42,
        n_estimators=200
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Features Driving Conversion"
    )

    st.markdown("## 🤖 Model & Segments")
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig_job, use_container_width=True)
        st.dataframe(job_perf, use_container_width=True)
    with col6:
        st.metric("✅ Model Accuracy", f"{accuracy:.4f}")
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(importance_df.head(10), use_container_width=True)

    best_job_row = job_perf.iloc[0]
    top_feature = importance_df.iloc[0]["Feature"]

    # -----------------------------
    # Prediction UI
    # -----------------------------
    st.markdown("## 🎯 Predict Customer Conversion")

    input_data = {}
    feature_cols = [c for c in df_filtered.columns if c != "y"]

    c_left, c_right = st.columns(2)
    half = (len(feature_cols) + 1) // 2

    for i, col in enumerate(feature_cols):
        target_col = c_left if i < half else c_right

        with target_col:
            if df_filtered[col].dtype == "object":
                options = sorted(df_filtered[col].dropna().unique().tolist())
                input_data[col] = st.selectbox(f"{col}", options=options, key=f"input_{col}")
            else:
                input_data[col] = st.number_input(
                    f"{col}",
                    value=float(df_filtered[col].median()),
                    key=f"input_{col}"
                )

    if st.button("Predict Conversion"):
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        if prediction == 1:
            st.success(f"✅ Likely to Convert (Probability: {probability:.2%})")
        else:
            st.error(f"❌ Not Likely to Convert (Probability: {probability:.2%})")

    # -----------------------------
    # Insights and recommendations
    # -----------------------------
    st.markdown("## 🧠 Key Insights")
    st.write(f"❌ Highest drop-off occurs at **{worst_stage}**.")
    st.write(f"✅ Best-performing contact channel is **{best_channel}**.")
    st.write(f"📌 Overall conversion rate is **{overall_conversion:.2%}**.")
    st.write(f"👔 Best job segment is **{best_job_row['job']}** with **{best_job_row['Conversion Rate']:.2%}** conversion.")
    st.write(f"🔑 Most important ML feature is **{top_feature}**.")

    if not month_perf["Conversion Rate"].dropna().empty:
        best_month_row = month_perf.dropna().sort_values("Conversion Rate", ascending=False).iloc[0]
        st.write(f"🗓️ Best month is **{best_month_row['month']}** with **{best_month_row['Conversion Rate']:.2%}** conversion.")

    st.markdown("## 💡 Recommendations")
    st.write("""
    - Prioritize the highest-converting contact channel in future campaigns
    - Improve the weakest funnel stage with better targeting and call strategy
    - Focus on the most responsive job segments and months
    - Reduce excessive repeat contacts for low-response prospects
    - Use the top ML features to optimize campaign planning and outreach
    """)

else:
    st.info("Please upload a semicolon-separated CSV file to begin.")