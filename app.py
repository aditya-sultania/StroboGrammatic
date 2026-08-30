import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw, load_master, load_inference, ensure_data
from src.analytics import state_ranking, top_hazards, trend, master_kpis, filtered_master, recommendations_for_hazard
from src.assistant import answer

st.set_page_config(page_title="OIL SIF Sentinel", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

DEMO_USERS = {
    "worker": ("worker123", "Worker"),
    "hse": ("hse123", "HSE Safety Officer"),
    "manager": ("manager123", "Management"),
}


def inject_css():
    st.markdown("""
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(135deg,#0b1220,#16243a); color:white; margin-bottom:1rem;}
    .hero h1 {margin:0; font-size:2.1rem;}
    .hero p {margin:.4rem 0 0; color:#cbd5e1;}
    .badge {display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem; font-weight:700; background:#1e293b; color:#e2e8f0;}
    .risk {font-size:1.4rem; font-weight:800;}
    </style>
    """, unsafe_allow_html=True)


def login():
    st.markdown("<div class='hero'><h1>🛡️ OIL SIF Sentinel</h1><p>Safety Intelligence &amp; Serious Injury/Fatality precursor decision platform</p></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Prototype access")
        username = st.text_input("Username", placeholder="worker / hse / manager")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", type="primary", use_container_width=True):
            if username in DEMO_USERS and password == DEMO_USERS[username][0]:
                st.session_state.role = DEMO_USERS[username][1]
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid prototype credentials.")
    with c2:
        st.info("Demo accounts")
        st.code("worker   / worker123\nhse      / hse123\nmanager  / manager123")
        st.caption("These credentials are for the local prototype only. Replace with enterprise authentication before production deployment.")


def sidebar(role):
    st.sidebar.title("OIL SIF Sentinel")
    st.sidebar.caption(f"Signed in as **{role}**")
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.clear(); st.rerun()
    if role == "Worker":
        return st.sidebar.radio("Workspace", ["Safety Home", "My Safety View"])
    if role == "HSE Safety Officer":
        return st.sidebar.radio("Workspace", ["HSE Command Center", "SIF Queue", "Trend & Precursor Lab", "AI Safety Assistant"])
    return st.sidebar.radio("Workspace", ["Executive Overview", "State Risk Ranking", "State Drill-down", "Decision Brief"])


def header(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def worker_view(page, raw):
    header("🦺 Worker Safety Dashboard", "Actionable safety intelligence without sensitive management information.")
    hazards = top_hazards(raw, n=6)
    if page == "Safety Home":
        a,b,c,d = st.columns(4)
        a.metric("Incident records", f"{len(raw):,}")
        a.metric("States represented", f"{raw['State'].nunique():,}")
        b.metric("Top hazard", hazards.iloc[0]['hazard'] if len(hazards) else "—")
        c.metric("Top hazard records", f"{int(hazards.iloc[0]['incidents']):,}" if len(hazards) else "0")
        d.metric("Latest record", str(raw['EventDate'].max().date()) if raw['EventDate'].notna().any() else "—")
        st.subheader("⚠️ Common hazards")
        st.dataframe(hazards, use_container_width=True, hide_index=True)
        st.subheader("🛡️ Life-Saving control reminders")
        for item in ["Verify energy isolation before maintenance.", "Stay out of line-of-fire zones.", "Use fall protection for work at height.", "Separate pedestrians from mobile equipment.", "Confirm hot-work controls before ignition."]:
            st.info(item)
    else:
        states = ["All states"] + sorted(raw["State"].dropna().unique().tolist())
        state = st.selectbox("Select your site/state", states)
        hz = top_hazards(raw, state if state != "All states" else None, 8)
        st.subheader(f"Safety patterns — {state}")
        st.dataframe(hz, use_container_width=True, hide_index=True)
        if len(hz):
            st.warning(f"Most frequent observed hazard/event: **{hz.iloc[0]['hazard']}** ({int(hz.iloc[0]['incidents']):,} records).")
            st.subheader("Recommended precautions")
            for r in recommendations_for_hazard(hz.iloc[0]['hazard']): st.write("•", r)


def hse_view(page, master, raw):
    header("🛡️ HSE Safety Officer Command Center", "Full SIF analytics, precursor intelligence, review queue and explainable recommendations.")
    k = master_kpis(master)
    if page == "HSE Command Center":
        a,b,c,d = st.columns(4)
        a.metric("Analyzed reports", f"{k['total']:,}")
        b.metric("SIF potential", f"{k['sif']:,}", f"{k['sif_rate']:.1f}%")
        c.metric("Review suggested", f"{k['review']:,}")
        d.metric("Mean SIF probability", f"{k['mean_probability']:.1%}")
        col1,col2=st.columns(2)
        with col1:
            p=master['sif_prediction'].value_counts().rename_axis('prediction').reset_index(name='records')
            st.plotly_chart(px.bar(p,x='prediction',y='records',title='SIF prediction distribution'),use_container_width=True)
        with col2:
            q=master['precursor'].fillna('Unknown').value_counts().head(10).rename_axis('precursor').reset_index(name='records')
            st.plotly_chart(px.bar(q,x='records',y='precursor',orientation='h',title='Top precursor patterns'),use_container_width=True)
    elif page == "SIF Queue":
        activities=["All"]+sorted(master['activity_group'].dropna().unique().tolist())
        precursors=["All"]+sorted(master['precursor'].dropna().unique().tolist())
        c1,c2,c3=st.columns(3)
        activity=c1.selectbox("Activity",activities)
        precursor=c2.selectbox("Precursor",precursors)
        min_prob=c3.slider("Minimum SIF probability",0.0,1.0,0.30,0.01)
        d=filtered_master(master,activity,precursor,min_prob)
        cols=[c for c in ['report_id','sif_probability','sif_prediction','activity_group','precursor','life_saving_rule','confidence_band','hse_review_status','priority_score'] if c in d.columns]
        st.dataframe(d[cols].head(500),use_container_width=True,hide_index=True)
        st.caption(f"Showing up to 500 of {len(d):,} matching records.")
    elif page == "Trend & Precursor Lab":
        d=master.copy(); d['month']=d['date_of_incident'].dt.to_period('M').dt.to_timestamp()
        tr=d.groupby('month',as_index=False).agg(reports=('report_id','count'),sif=('sif_prediction',lambda s:(s=='SIF Potential').sum()))
        st.plotly_chart(px.line(tr,x='month',y=['reports','sif'],title='Report and SIF-potential trend'),use_container_width=True)
        c1,c2=st.columns(2)
        with c1: st.dataframe(master['activity_group'].value_counts().head(10).rename_axis('activity').reset_index(name='records'),use_container_width=True,hide_index=True)
        with c2: st.dataframe(master['life_saving_rule'].value_counts().head(10).rename_axis('rule').reset_index(name='records'),use_container_width=True,hide_index=True)
    else:
        st.subheader("Ask the HSE AI Safety Assistant")
        st.caption("This prototype assistant is grounded in the supplied datasets and uses transparent analytical rules. It does not invent operational facts.")
        q=st.text_area("Question",placeholder="Which states have the highest incident burden? What are the top SIF precursors? How can we prevent the most common hazard?")
        if st.button("Analyze",type="primary"):
            st.success(answer(q,raw,master))


def manager_view(page, raw, master):
    header("👔 Management Decision Dashboard", "State-level safety burden, emerging hazards and action-oriented decision support.")
    ranking=state_ranking(raw)
    if page == "Executive Overview":
        top=ranking.head(10).copy(); top.index=range(1,len(top)+1)
        a,b,c,d=st.columns(4)
        a.metric("States",f"{len(ranking):,}")
        b.metric("Incident records",f"{len(raw):,}")
        c.metric("Highest-burden state",ranking.iloc[0]['State'].title() if len(ranking) else '—')
        d.metric("Master SIF potential",f"{(master['sif_prediction']=='SIF Potential').sum():,}")
        st.subheader("Top states by observed incident burden + severity")
        st.dataframe(top[['State','incidents','hospitalized','amputations','loss_of_eye','severity_rate_per_100','risk_index','risk_band']],use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(ranking.head(15),x='risk_index',y='State',orientation='h',color='risk_band',title='State risk index — transparent prototype score'),use_container_width=True)
    elif page == "State Risk Ranking":
        st.subheader("US state ranking")
        st.caption("Ranking combines record burden (65%) and severity-rate percentile (35%). It is an analytical indicator, not an inherent safety label for a state.")
        st.dataframe(ranking[['State','incidents','employers','hospitalized','amputations','loss_of_eye','severe_events','severity_rate_per_100','risk_index','risk_band']],use_container_width=True,hide_index=True)
    elif page == "State Drill-down":
        state=st.selectbox("State",ranking['State'].tolist())
        r=ranking[ranking['State']==state].iloc[0]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Incidents",f"{int(r.incidents):,}"); c2.metric("Hospitalized",f"{int(r.hospitalized):,}"); c3.metric("Amputations",f"{int(r.amputations):,}"); c4.metric("Risk index",f"{r.risk_index:.1f}")
        hz=top_hazards(raw,state,12)
        st.subheader(f"Top hazards/events — {state.title()}")
        st.dataframe(hz,use_container_width=True,hide_index=True)
        if len(hz):
            st.subheader("Priority controls")
            for rec in recommendations_for_hazard(hz.iloc[0]['hazard']): st.write("•",rec)
        d=raw[raw['State']==state].dropna(subset=['EventDate']).copy(); d['month']=d['EventDate'].dt.to_period('M').dt.to_timestamp(); tr=d.groupby('month').size().reset_index(name='incidents')
        st.plotly_chart(px.line(tr,x='month',y='incidents',title=f'{state.title()} incident trend'),use_container_width=True)
    else:
        st.subheader("Decision brief")
        r=ranking.iloc[0]
        hz=top_hazards(raw,r.State,5)
        st.markdown(f"### Current attention state: {r.State.title()}")
        st.write(f"The supplied OSHA dataset contains **{int(r.incidents):,}** records for this state. The prototype risk index is **{r.risk_index:.1f}**. The leading observed hazards/events are **{', '.join(hz.hazard.head(3))}**.")
        st.markdown("### Recommended management actions")
        for x in ["Prioritize HSE review of high-probability SIF records.", "Fund controls against the leading recurring hazard/event patterns.", "Track state-level trends monthly rather than using a single-period snapshot.", "Validate the prototype score against company-specific exposure/denominator data before operational decisions."]:
            st.write("•",x)


def main():
    inject_css()
    if 'role' not in st.session_state:
        login(); return
    role=st.session_state.role
    try:
        ensure_data()
        raw=load_raw(); master=load_master()
    except Exception as e:
        st.error("Dataset loading failed.")
        st.exception(e)
        st.stop()
    page=sidebar(role)
    if role == "Worker": worker_view(page,raw)
    elif role == "HSE Safety Officer": hse_view(page,master,raw)
    else: manager_view(page,raw,master)

if __name__ == "__main__":
    main()
