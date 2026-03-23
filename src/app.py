from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import altair as alt
import pandas as pd
import streamlit as st

try:
    from . import config
except ImportError:  # pragma: no cover - allows `streamlit run src/app.py`
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import src.config as config


ACTION_ORDER = [
    "priority_outreach",
    "offer_incentive",
    "product_education",
    "monitor_only",
]

ACTION_LABELS = {
    "priority_outreach": "Priority Outreach",
    "offer_incentive": "Offer Incentive",
    "product_education": "Product Education",
    "monitor_only": "Monitor Only",
}

ACTION_COLORS = {
    "Priority Outreach": "#B42318",
    "Offer Incentive": "#F79009",
    "Product Education": "#1570EF",
    "Monitor Only": "#667085",
}

PLAYBOOKS = {
    "priority_outreach": "Escalate to an account manager within 24 hours.",
    "offer_incentive": "Prepare a targeted save offer with budget controls.",
    "product_education": "Schedule onboarding or feature adoption follow-up.",
    "monitor_only": "Monitor signals and re-score on the next portfolio refresh.",
}


@st.cache_data(show_spinner=False)
def _load_artifact(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_action(action: str) -> str:
    return ACTION_LABELS.get(action, action.replace("_", " ").title())


def _build_action_reason(row: pd.Series) -> str:
    thresholds = config.ACTION_THRESHOLDS
    risk_text = f"{row['churn_probability']:.0%} churn risk"

    if row["recommended_action"] == "priority_outreach":
        drivers = []
        if row["monthly_gpv"] >= thresholds["priority_gpv"]:
            drivers.append("high GPV")
        if row["chargeback_rate"] >= thresholds["severe_chargeback_rate"]:
            drivers.append("elevated chargebacks")
        if row["inactivity_days"] >= thresholds["high_inactivity_days"]:
            drivers.append("extended inactivity")
        driver_text = ", ".join(drivers[:2]) if drivers else "material merchant risk"
        return f"{risk_text} with {driver_text} warrants immediate human outreach."

    if row["recommended_action"] == "offer_incentive":
        return (
            f"{risk_text} and ${row['monthly_gpv']:,.0f} monthly GPV justify a commercial save motion."
        )

    if row["recommended_action"] == "product_education":
        return (
            f"{risk_text} with only {int(row['product_adoption_count'])} adopted products points to an enablement gap."
        )

    return f"{risk_text} stays below action thresholds, so the merchant remains on watch."


def _enrich_scored_df(scored_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_df.copy()
    df["action_label"] = df["recommended_action"].map(_format_action)
    df["needs_action"] = df["recommended_action"] != "monitor_only"
    df["priority_bucket"] = df["recommended_action"].replace(
        {
            "priority_outreach": "Immediate save",
            "offer_incentive": "Immediate save",
            "product_education": "Enablement queue",
            "monitor_only": "Watchlist",
        }
    )
    df["action_reason"] = df.apply(_build_action_reason, axis=1)
    df["recommended_playbook"] = df["recommended_action"].map(PLAYBOOKS)
    return df


def _build_action_summary(filtered_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        filtered_df.groupby(["recommended_action", "action_label"], as_index=False)
        .agg(
            merchant_count=("merchant_id", "count"),
            observed_churn_rate=("churned", "mean"),
            average_churn_probability=("churn_probability", "mean"),
            total_expected_retention_value=("expected_retention_value", "sum"),
        )
    )

    for action in ACTION_ORDER:
        label = _format_action(action)
        if label not in summary["action_label"].values:
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "recommended_action": action,
                                "action_label": label,
                                "merchant_count": 0,
                                "observed_churn_rate": 0.0,
                                "average_churn_probability": 0.0,
                                "total_expected_retention_value": 0.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    summary["action_sort"] = summary["recommended_action"].apply(ACTION_ORDER.index)
    return summary.sort_values("action_sort").reset_index(drop=True)


def _kpi_card(label: str, value: str, detail: str, tone: str) -> None:
    st.markdown(
        f"""
        <div style="
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            background: {tone};
            border: 1px solid rgba(16, 24, 40, 0.08);
            min-height: 132px;
        ">
            <div style="font-size: 0.82rem; color: #475467; text-transform: uppercase; letter-spacing: 0.04em;">
                {label}
            </div>
            <div style="font-size: 1.9rem; font-weight: 700; color: #101828; margin-top: 0.3rem;">
                {value}
            </div>
            <div style="font-size: 0.92rem; color: #344054; margin-top: 0.35rem;">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_action_value_chart(action_summary: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(action_summary)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X(
                "total_expected_retention_value:Q",
                title="Total Expected Retention Value",
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y(
                "action_label:N",
                sort=[_format_action(action) for action in ACTION_ORDER],
                title=None,
            ),
            color=alt.Color(
                "action_label:N",
                scale=alt.Scale(
                    domain=list(ACTION_COLORS.keys()),
                    range=list(ACTION_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("action_label:N", title="Action"),
                alt.Tooltip("merchant_count:Q", title="Merchants", format=","),
                alt.Tooltip(
                    "average_churn_probability:Q",
                    title="Avg Churn Probability",
                    format=".1%",
                ),
                alt.Tooltip(
                    "observed_churn_rate:Q",
                    title="Observed Churn Rate",
                    format=".1%",
                ),
                alt.Tooltip(
                    "total_expected_retention_value:Q",
                    title="Expected Retention Value",
                    format="$,.2f",
                ),
            ],
        )
        .properties(height=280)
    )


def _build_risk_distribution_chart(filtered_df: pd.DataFrame) -> alt.Chart:
    thresholds = config.ACTION_THRESHOLDS
    threshold_df = pd.DataFrame(
        [
            {
                "threshold": thresholds["education_risk"],
                "action_label": "Product Education threshold",
            },
            {
                "threshold": thresholds["offer_risk"],
                "action_label": "Offer Incentive threshold",
            },
            {
                "threshold": thresholds["priority_risk"],
                "action_label": "Priority Outreach threshold",
            },
        ]
    )

    color_scale = alt.Scale(
        domain=[
            "Product Education threshold",
            "Offer Incentive threshold",
            "Priority Outreach threshold",
        ],
        range=[
            ACTION_COLORS["Product Education"],
            ACTION_COLORS["Offer Incentive"],
            ACTION_COLORS["Priority Outreach"],
        ],
    )

    histogram = (
        alt.Chart(filtered_df)
        .mark_bar(color="#98A2B3", opacity=0.7)
        .encode(
            x=alt.X(
                "churn_probability:Q",
                title="Churn Probability",
                bin=alt.Bin(maxbins=24),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y("count():Q", title="Merchant Count"),
            tooltip=[
                alt.Tooltip("count():Q", title="Merchants", format=","),
            ],
        )
    )

    rules = (
        alt.Chart(threshold_df)
        .mark_rule(strokeDash=[6, 4], size=2)
        .encode(
            x=alt.X("threshold:Q"),
            color=alt.Color("action_label:N", scale=color_scale, title=None),
            tooltip=[
                alt.Tooltip("action_label:N", title="Threshold"),
                alt.Tooltip("threshold:Q", title="Value", format=".0%"),
            ],
        )
    )

    return (histogram + rules).properties(height=280)


def _build_risk_value_chart(filtered_df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(filtered_df)
        .mark_circle(opacity=0.72, stroke="white", strokeWidth=0.4)
        .encode(
            x=alt.X(
                "monthly_gpv:Q",
                title="Monthly GPV (log scale)",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(format="$,.0f"),
            ),
            y=alt.Y(
                "churn_probability:Q",
                title="Churn Probability",
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "action_label:N",
                scale=alt.Scale(
                    domain=list(ACTION_COLORS.keys()),
                    range=list(ACTION_COLORS.values()),
                ),
                title="Recommended Action",
            ),
            size=alt.Size(
                "expected_retention_value:Q",
                title="Expected Retention Value",
                scale=alt.Scale(range=[30, 900]),
            ),
            tooltip=[
                alt.Tooltip("merchant_id:N", title="Merchant"),
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("monthly_gpv:Q", title="Monthly GPV", format="$,.2f"),
                alt.Tooltip("churn_probability:Q", title="Churn Probability", format=".1%"),
                alt.Tooltip("recommended_action:N", title="Action"),
                alt.Tooltip(
                    "expected_retention_value:Q",
                    title="Expected Retention Value",
                    format="$,.2f",
                ),
                alt.Tooltip("product_adoption_count:Q", title="Products Adopted"),
                alt.Tooltip("inactivity_days:Q", title="Inactivity Days"),
            ],
        )
        .properties(height=360)
        .interactive()
    )


def _build_coefficient_chart(coefficients_df: pd.DataFrame) -> alt.Chart:
    coeff_df = coefficients_df.copy()
    coeff_df["direction"] = coeff_df["coefficient"].apply(
        lambda value: "Raises churn" if value > 0 else "Reduces churn"
    )
    coeff_df = coeff_df.head(8).iloc[::-1]

    return (
        alt.Chart(coeff_df)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("coefficient:Q", title="Logistic Regression Coefficient"),
            y=alt.Y("feature:N", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises churn", "Reduces churn"],
                    range=["#B42318", "#1570EF"],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("coefficient:Q", title="Coefficient", format=".3f"),
                alt.Tooltip("direction:N", title="Direction"),
            ],
        )
        .properties(height=260)
    )


def _question_block(question: str, answer: str, accent: str) -> None:
    st.markdown(
        f"""
        <div style="
            border-left: 4px solid {accent};
            padding: 0.9rem 1rem;
            background: #FCFCFD;
            border-radius: 10px;
            margin-bottom: 0.7rem;
        ">
            <div style="font-weight: 700; color: #101828; margin-bottom: 0.35rem;">{question}</div>
            <div style="color: #344054; line-height: 1.55;">{answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_download_narrative(
    filtered_df: pd.DataFrame,
    action_summary: pd.DataFrame,
    metrics: pd.Series,
    visible_merchants: int,
    needs_action_count: int,
    immediate_save_count: int,
    total_retention_value: float,
    top_action_row: pd.Series,
    priority_queue: pd.DataFrame,
) -> str:
    top_merchant = priority_queue.iloc[0]
    action_lines = "\n".join(
        [
            (
                f"- **{row['action_label']}**: {int(row['merchant_count'])} merchants, "
                f"{row['average_churn_probability']:.1%} average predicted churn, "
                f"${row['total_expected_retention_value']:,.2f} expected retention value."
            )
            for _, row in action_summary.iterrows()
        ]
    )

    return dedent(
        f"""
        # Merchant Retention Decision Engine Operator Guide

        This document is written for someone using the dashboard to understand the merchant portfolio and decide where to act.

        It is intentionally different from the repository README:
        - the **README** explains the project, architecture, and implementation
        - this **operator guide** explains how to read the outputs and use them in practice

        ## In one sentence

        This system estimates which merchants are at risk of churn, recommends the most appropriate retention action, and ranks merchants by expected business value.

        ## What this dashboard is answering

        In the current filtered view, the system is looking at **{visible_merchants:,} merchants**. Of those, **{needs_action_count:,} merchants** need some level of intervention, and **{immediate_save_count:,} merchants** sit in the most urgent save queue.

        The total expected retention value in this view is **${total_retention_value:,.2f}** over a {config.RETENTION_HORIZON_MONTHS}-month horizon. This is not guaranteed revenue. It is an estimate of retained gross margin if the recommended action improves outcomes as assumed in the prototype.

        The practical questions answered here are:
        - Which merchants appear most at risk of churn?
        - Which merchants are worth acting on first?
        - What action does the system recommend?
        - Why did the system recommend that action?
        - Where is the largest retention opportunity in the portfolio?

        ### 1. Which merchants appear most at risk of churn?

        The churn model assigns every merchant a probability between 0 and 1. Higher values mean the merchant has more signals associated with churn.

        Signals that generally increase churn in this prototype include:
        - negative GPV trend
        - higher inactivity
        - elevated chargeback rate
        - more support tickets
        - lower product adoption

        Signals that generally reduce churn include:
        - longer tenure
        - broader product adoption
        - healthier GPV patterns

        The current model's ROC-AUC is **{metrics['roc_auc']:.3f}**, which means it has useful separation power for ranking merchants by risk. This project should be read primarily as a prioritization system, not as a strict yes/no churn alarm.

        ### 2. Where is the retention opportunity?

        The dashboard does not stop at predicting risk. It translates risk into expected business value.

        A merchant can be commercially important even if its churn probability is not the highest. That is why the system combines:
        - churn probability
        - merchant monthly GPV
        - action-specific expected uplift
        - incentive cost where applicable

        In the current view, the action category with the largest opportunity is **{top_action_row['action_label']}**, representing **${top_action_row['total_expected_retention_value']:,.2f}** in expected retention value.

        Action summary:
        {action_lines}

        ### 3. Which merchants should the team work first?

        The ranked queue sorts merchants by expected retention value, then uses churn probability and monthly GPV to break ties.

        The current top merchant is **{top_merchant['merchant_id']}** in the **{top_merchant['segment']}** segment.
        - Recommended action: **{top_merchant['action_label']}**
        - Churn probability: **{top_merchant['churn_probability']:.1%}**
        - Monthly GPV: **${top_merchant['monthly_gpv']:,.2f}**
        - Expected retention value: **${top_merchant['expected_retention_value']:,.2f}**

        This means the queue is optimized for practical intervention order rather than raw model score alone.

        ### 4. Why is each merchant receiving a specific action?

        The decision engine uses explicit rules so the recommendations are explainable.

        - **Priority Outreach**: used for the highest-risk merchants when the account is valuable or showing severe distress.
        - **Offer Incentive**: used when churn risk is high enough that a commercial save motion may be justified.
        - **Product Education**: used when the merchant is at moderate risk and the clearest intervention lever is low product adoption.
        - **Monitor Only**: used when the merchant is below action thresholds and should remain on watch.

        This makes the system easier to understand than a pure black-box recommendation engine.

        ### 5. How should an unfamiliar user read the visuals?

        - **Top KPI cards** summarize the size of the visible portfolio, how many merchants need action, how many belong in the immediate save queue, and how much retention value is visible.
        - **Retention Opportunity by Action** shows where the expected business value is concentrated. This is the best chart for resource planning.
        - **Churn Risk Distribution** shows how merchants are spread across low, medium, and high risk. The vertical lines mark the action thresholds that trigger education, incentives, and outreach.
        - **Risk and Revenue Map** shows why some merchants rank higher than others. Higher on the chart means more risk, farther right means more GPV, and larger bubbles mean more expected value.
        - **Top Priority Merchants** is the actionable work queue. Start there if the goal is to take action.

        ## How the system reaches its answer

        The implementation works in four linked steps:

        1. **Generate merchant signals** that look like a realistic payments portfolio.
        2. **Estimate churn probability** with a transparent logistic regression model.
        3. **Translate risk into action** with readable business rules.
        4. **Rank merchants by expected retention value** so the team knows where to spend effort first.

        That progression is the core of the project. It moves from prediction to decision to operational prioritization.

        ## What this is and is not

        This prototype is useful for:
        - explaining retention strategy in a portfolio setting
        - demonstrating how model outputs become business actions
        - showing an end-to-end local analytics product

        This prototype is not:
        - a production treatment optimization system
        - a causal estimate of true intervention lift
        - a substitute for real merchant history or experimentation

        ## Recommended reading order for a new user

        1. Look at the KPI cards to understand the scale of the problem.
        2. Check the action-value chart to see where the retention opportunity sits.
        3. Use the risk and revenue map to understand the tradeoff between merchant value and risk.
        4. Work the top-priority table from the top down.
        5. Review diagnostics only if you want to understand model quality.
        """
    ).strip()


st.set_page_config(page_title="Merchant Retention Decision Engine", layout="wide")
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(16, 24, 40, 0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

scored_df = _load_artifact(str(config.MERCHANT_OUTPUT_PATH))
metrics_df = _load_artifact(str(config.METRICS_OUTPUT_PATH))
coefficients_df = _load_artifact(str(config.COEFFICIENTS_OUTPUT_PATH))

if scored_df is None or metrics_df is None or coefficients_df is None:
    st.warning("Run `python -m src.pipeline` first to generate the dashboard artifacts.")
    st.stop()

scored_df = _enrich_scored_df(scored_df)
metrics = metrics_df.iloc[0]

st.title("Merchant Retention Decision Engine")
st.caption("Turn merchant risk signals into a ranked retention queue with clear next actions.")

st.sidebar.header("Filter The Portfolio")
selected_segments = st.sidebar.multiselect(
    "Merchant segments",
    options=sorted(scored_df["segment"].unique()),
    default=sorted(scored_df["segment"].unique()),
)
selected_actions = st.sidebar.multiselect(
    "Recommended actions",
    options=[_format_action(action) for action in ACTION_ORDER],
    default=[_format_action(action) for action in ACTION_ORDER],
)
min_churn_probability = st.sidebar.slider(
    "Minimum churn probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
    format="%.2f",
)
min_retention_value = st.sidebar.slider(
    "Minimum expected retention value",
    min_value=0.0,
    max_value=float(scored_df["expected_retention_value"].max()),
    value=0.0,
    step=25.0,
)
top_n = st.sidebar.slider(
    "Rows in priority table",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
)

filtered_df = scored_df[
    scored_df["segment"].isin(selected_segments)
    & scored_df["action_label"].isin(selected_actions)
    & (scored_df["churn_probability"] >= min_churn_probability)
    & (scored_df["expected_retention_value"] >= min_retention_value)
].copy()

if filtered_df.empty:
    st.warning("No merchants match the current filters. Widen the selection in the sidebar.")
    st.stop()

action_summary = _build_action_summary(filtered_df)
priority_queue = filtered_df.sort_values("priority_rank").head(top_n)

action_summary_display = action_summary.assign(
    observed_churn_pct=action_summary["observed_churn_rate"] * 100,
    average_churn_pct=action_summary["average_churn_probability"] * 100,
)
priority_queue_display = priority_queue.assign(
    churn_probability_pct=priority_queue["churn_probability"] * 100,
)

visible_merchants = len(filtered_df)
needs_action_count = int(filtered_df["needs_action"].sum())
needs_action_share = needs_action_count / visible_merchants
immediate_save_count = int(
    filtered_df["recommended_action"].isin(["priority_outreach", "offer_incentive"]).sum()
)
total_retention_value = filtered_df["expected_retention_value"].sum()
top_action_row = action_summary.sort_values("total_expected_retention_value", ascending=False).iloc[0]
narrative_markdown = _build_download_narrative(
    filtered_df=filtered_df,
    action_summary=action_summary,
    metrics=metrics,
    visible_merchants=visible_merchants,
    needs_action_count=needs_action_count,
    immediate_save_count=immediate_save_count,
    total_retention_value=total_retention_value,
    top_action_row=top_action_row,
    priority_queue=priority_queue,
)

st.caption(
    f"Showing {visible_merchants:,} of {len(scored_df):,} merchants based on the current filters."
)

st.sidebar.divider()
st.sidebar.markdown("### Operator Guide")
st.sidebar.caption(
    "Download the end-user guide for this dashboard. It explains the outputs and workflow; the README stays focused on the project itself."
)
st.sidebar.download_button(
    "Download Operator Guide",
    data=narrative_markdown,
    file_name="merchant_retention_operator_guide.md",
    mime="text/markdown",
    width="stretch",
)

with st.container(border=True):
    st.markdown("### Project Guidelines")
    st.caption(
        "This section is the quickest way to understand what the page is showing and how to use it."
    )
    tab_overview, tab_signals, tab_actions = st.tabs(
        ["What This Page Answers", "How To Read The Signals", "How To Use The Queue"]
    )
    with tab_overview:
        st.markdown(
            """
            This page is built to answer three questions:
            - Where is the retention opportunity?
            - Which merchants should the team work first?
            - Why did the system recommend that action?

            Read the KPI row first for scale, then the action-value chart for resource planning, then the ranked table for execution.
            """
        )
    with tab_signals:
        st.markdown(
            f"""
            - **Churn Probability** is the model's estimate of churn risk. Use it for ranking, not as a hard yes/no verdict.
            - **Expected Retention Value** is an estimated {config.RETENTION_HORIZON_MONTHS}-month retained gross margin opportunity.
            - **Priority Rank** is the order in which merchants should generally be worked.
            - **Recommended Action** is the rule engine's suggested intervention based on risk, merchant value, and product usage.
            """
        )
    with tab_actions:
        st.markdown(
            """
            - Start with **Priority Outreach** and **Offer Incentive** if the team needs an immediate save queue.
            - Use **Product Education** for enablement or onboarding motions.
            - Leave **Monitor Only** merchants in the watchlist unless their signals worsen.
            - In the table, the `Why This Merchant` and `Recommended Next Step` columns are the fastest way to brief someone unfamiliar with the account.
            """
        )

metric_cols = st.columns(5)
with metric_cols[0]:
    _kpi_card(
        "Visible Merchants",
        f"{visible_merchants:,}",
        "Merchants included in the current view.",
        "#F8FAFC",
    )
with metric_cols[1]:
    _kpi_card(
        "Need Action",
        f"{needs_action_count:,}",
        f"{needs_action_share:.1%} of the filtered portfolio requires intervention.",
        "#F5F3FF",
    )
with metric_cols[2]:
    _kpi_card(
        "Immediate Save Queue",
        f"{immediate_save_count:,}",
        "Priority outreach plus incentive cases to work first.",
        "#FFF7ED",
    )
with metric_cols[3]:
    _kpi_card(
        "Retention Value In View",
        f"${total_retention_value:,.0f}",
        "Expected 6-month retained gross margin from the visible merchants.",
        "#ECFDF3",
    )
with metric_cols[4]:
    _kpi_card(
        "Largest Opportunity",
        top_action_row["action_label"],
        f"${top_action_row['total_expected_retention_value']:,.0f} in expected value.",
        "#FEF3F2",
    )

overview_col, distribution_col = st.columns((1.15, 1))

with overview_col:
    st.markdown("### Retention Opportunity By Action")
    st.caption(
        "Question answered: Where is the business opportunity concentrated, and what kind of team response does the portfolio need?"
    )
    st.altair_chart(_build_action_value_chart(action_summary), width="stretch")
    st.dataframe(
        action_summary_display[
            [
                "action_label",
                "merchant_count",
                "observed_churn_pct",
                "average_churn_pct",
                "total_expected_retention_value",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "action_label": st.column_config.TextColumn("Action"),
            "merchant_count": st.column_config.NumberColumn("Merchants", format="%d"),
            "observed_churn_pct": st.column_config.NumberColumn(
                "Observed Churn",
                format="%.1f%%",
            ),
            "average_churn_pct": st.column_config.NumberColumn(
                "Avg Predicted Churn",
                format="%.1f%%",
            ),
            "total_expected_retention_value": st.column_config.NumberColumn(
                "Expected Value",
                format="$%.2f",
            ),
        },
    )

with distribution_col:
    st.markdown("### Churn Risk Distribution")
    st.caption(
        "Question answered: How much of the portfolio falls below, near, or above the intervention thresholds?"
    )
    st.altair_chart(_build_risk_distribution_chart(filtered_df), width="stretch")
    st.markdown(
        f"""
        Decision thresholds:
        `Product Education` at **{config.ACTION_THRESHOLDS['education_risk']:.0%}**,
        `Offer Incentive` at **{config.ACTION_THRESHOLDS['offer_risk']:.0%}**,
        `Priority Outreach` at **{config.ACTION_THRESHOLDS['priority_risk']:.0%}**.
        """
    )

st.markdown("### Risk And Revenue Map")
st.caption(
    "Question answered: Which merchants combine enough risk and enough commercial importance to deserve action now? Farther right means more GPV, higher means more churn risk, and larger bubbles mean more expected value."
)
st.altair_chart(_build_risk_value_chart(filtered_df), width="stretch")

st.markdown("### Top Priority Merchants")
st.caption(
    "Question answered: Which merchants should the team work first, what should they do, and why? Priority rank is computed across the full portfolio, so filters narrow the list without re-ranking it."
)
st.dataframe(
    priority_queue_display[
        [
            "priority_rank",
            "merchant_id",
            "segment",
            "priority_bucket",
            "monthly_gpv",
            "churn_probability_pct",
            "action_label",
            "expected_retention_value",
            "recommended_playbook",
            "action_reason",
        ]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "priority_rank": st.column_config.NumberColumn("Rank", format="%d"),
        "merchant_id": st.column_config.TextColumn("Merchant"),
        "segment": st.column_config.TextColumn("Segment"),
        "priority_bucket": st.column_config.TextColumn("Queue"),
        "monthly_gpv": st.column_config.NumberColumn("Monthly GPV", format="$%.2f"),
        "churn_probability_pct": st.column_config.ProgressColumn(
            "Churn Probability",
            min_value=0.0,
            max_value=100.0,
            format="%.1f%%",
        ),
        "action_label": st.column_config.TextColumn("Action"),
        "expected_retention_value": st.column_config.NumberColumn(
            "Expected Value",
            format="$%.2f",
        ),
        "recommended_playbook": st.column_config.TextColumn("Recommended Next Step", width="medium"),
        "action_reason": st.column_config.TextColumn("Why This Merchant", width="large"),
    },
)

with st.expander("Model Diagnostics", expanded=False):
    diag_cols = st.columns(5)
    diag_cols[0].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    diag_cols[1].metric("Precision", f"{metrics['precision']:.3f}")
    diag_cols[2].metric("Recall", f"{metrics['recall']:.3f}")
    diag_cols[3].metric("F1", f"{metrics['f1']:.3f}")
    diag_cols[4].metric("Avg Predicted Churn", f"{metrics['average_predicted_churn']:.1%}")

    st.markdown(
        "Use these metrics as a quality check. This prototype is designed primarily as a ranking and prioritization tool, not a hard yes/no churn classifier."
    )
    st.altair_chart(_build_coefficient_chart(coefficients_df), width="stretch")
