import io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='ACL | Water Sort Research Evaluation',page_icon='📊',layout='wide')
BASE=Path(__file__).resolve().parent
DATA=BASE/'ACL_WaterSort_Evaluation'
T=DATA/'tables'; F=DATA/'figures'
@st.cache_data(show_spinner=False)
def read_csv(name): return pd.read_csv(T/(name+'.csv'),low_memory=False)
master=read_csv('model_level_master_table'); puzzles=read_csv('puzzle_level_audit'); tests=read_csv('paired_mcnemar_holm')
# Recover strategy from recorded modes; do not infer missing results.
mode_map={'full_solution':'zero-shot','few_shot_1':'few-shot','step_cot_verified':'step-CoT'}
puzzles['strategy']=puzzles['mode'].map(mode_map).fillna(puzzles['mode'])
puzzles['solved']=puzzles['solved'].astype(str).str.lower().eq('true')
full=master[master.strategy.isin(['zero-shot','few-shot'])].copy()
paired=puzzles[puzzles.strategy.isin(['zero-shot','few-shot'])].copy()

PAGES=[
('01','Matched Success Rates','matched_success_rates','Model performance','Which model and prompting strategy solve the most puzzles?', 'Higher is better, but report absolute percentages and solved/evaluated counts.'),
('02','Model Success Heatmap','success_heatmap_counts','Model performance','How do model-wise success counts compare between prompting strategies?', 'Each heatmap cell is solved/evaluated; colors represent solve rate.'),
('03','Few-Shot Gain or Decline','few_shot_gain','Prompting comparison','Which models improve or decline under few-shot prompting?', 'A percentage-point difference is not the same as a relative percentage change.'),
('04','Success by Difficulty Tier','success_by_tier','Puzzle difficulty','Does success vary with the recorded difficulty tier?', 'These are pooled descriptive rates; models share puzzles, so pooled observations are dependent.'),
('05','Success by Bottle Count','success_by_bottle_count','Puzzle difficulty','How does puzzle size relate to solving success?', 'Look at denominators in each bottle-count group before interpreting trends.'),
('06','Wilson Confidence Intervals','wilson_intervals','Statistical uncertainty','How precise are the observed model-level success rates?', 'Wilson intervals quantify binomial uncertainty within configurations, not generalization across benchmarks.'),
('07','Invalid-Move Taxonomy','invalid_reason_taxonomy','Failure analysis','Which invalid-move reasons are most frequently recorded?', 'Reason counters count events, not necessarily distinct failed puzzles.'),
('08','Success vs Latency','success_vs_latency','Efficiency','How do solve rates relate to recorded inference time?', 'Observational comparison only; hardware and runtime are not controlled.'),
('09','Token Budget','token_budget','Efficiency','How many prompt and generated tokens are used per puzzle?', 'Token totals are normalized by attempted puzzles, not by successful puzzles.'),
('10','Terminal Status Distribution','terminal_status_distribution','Failure analysis','Which terminal statuses occur under each prompting strategy?', 'The chart uses a top-status subset; use the complete table to check all statuses and denominators.'),
('11','Outcome vs Legal Moves','outcome_by_legal_moves','Failure analysis','How does the final outcome vary with executed legal moves?', 'Post-outcome descriptive association only; do not make causal claims.'),
('12','Paired Outcome Transitions','paired_outcome_transition','Prompting comparison','Which matched model-puzzle pairs switch success status?', 'Discordant pairs, not all trials, drive exact McNemar tests.'),
('13','Step-CoT Observed Results','step_cot_observed','Step-CoT','What was observed in the Step-CoT verified runs?', 'Step-CoT samples are unequal and not a matched 700-puzzle comparison.'),
('14','Puzzle-Level Robustness','puzzle_robustness','Robustness','How many models solve each puzzle?', 'A zero means no evaluated model solved that puzzle within that strategy.')]

@st.cache_data(show_spinner=False)
def table_for(n):
    if n in ('01','02','06','08','09','13'):
        t=master.copy()
        if n=='13': t=t[t.strategy.eq('step-CoT')]
        else: t=t[t.strategy.isin(['zero-shot','few-shot'])]
        if n=='06': return t[['model','strategy','n','solved','rate','ci_low','ci_high']].sort_values('rate',ascending=False)
        if n=='08': return t[['model','strategy','n','solved','rate','mean_seconds_per_puzzle','seconds']].sort_values('rate',ascending=False)
        if n=='09':
            t['prompt_tokens_per_puzzle']=t.prompt_tokens/t.n
            t['generated_tokens_per_puzzle']=t.generated_tokens/t.n
            return t[['model','strategy','n','prompt_tokens_per_puzzle','generated_tokens_per_puzzle']]
        return t[['model','strategy','n','solved','rate']].sort_values('rate',ascending=False)
    if n=='03':
        t=full.pivot(index='model',columns='strategy',values='rate').dropna().reset_index()
        t['few_minus_zero_pp']=100*(t['few-shot']-t['zero-shot'])
        return t.sort_values('few_minus_zero_pp',ascending=False)
    if n=='04': return read_csv('success_by_tier')
    if n=='05': return read_csv('success_by_bottle_count')
    if n=='07': return read_csv('invalid_reason_counts')
    if n=='10': return read_csv('status_distribution')
    if n=='11': return read_csv('success_by_executed_moves')
    if n=='12':
        p=paired.pivot(index=['model','puzzle_id'],columns='strategy',values='solved').dropna()
        return pd.crosstab(p['zero-shot'],p['few-shot']).reindex(index=[False,True],columns=[False,True],fill_value=0).rename_axis(index='zero_shot_solved',columns='few_shot_solved').stack().rename('matched_pairs').reset_index()
    if n=='14': return read_csv('puzzle_robustness')
    return pd.DataFrame()

st.markdown('''<style>.block-container{padding-top:1.5rem;max-width:1450px}.small-note{color:#64748b;font-size:.92rem}.section-tag{font-size:.8rem;letter-spacing:.09em;text-transform:uppercase;color:#64748b}</style>''',unsafe_allow_html=True)
st.title('Water Sort | ACL Research Evaluation')
st.caption('Reproducible, source-grounded analysis • 14 figures • one associated measurement table per figure • original records retained')

with st.sidebar:
    st.header('Navigation')
    section=st.radio('View', ['Overview','All 14 figure–table pairs','Single figure explorer','Statistical tests','Full data audit','Downloads & verification'])
    st.divider()
    st.info('No file uploads. All data and figures are bundled with this application.')

c1,c2,c3,c4=st.columns(4)
c1.metric('Puzzle records',f'{len(puzzles):,}')
c2.metric('Models',str(master.model.nunique()))
c3.metric('Configurations',str(len(master)))
c4.metric('Figures',str(len(PAGES)))

@st.cache_data(show_spinner=False)
def png_bytes(path): return path.read_bytes()

def show_pair(info):
    num,title,stem,category,question,note=info
    st.subheader(f'Figure {num} — {title}')
    st.caption(f'{category} | Research question: {question}')
    img=F/f'{num}_{stem}.png'
    pdf=F/f'{num}_{stem}.pdf'
    if img.exists():
        st.image(png_bytes(img),use_container_width=True)
    else: st.error(f'Missing source figure: {img.name}')
    st.markdown(f'**Interpretation:** {note}')
    table=table_for(num)
    st.markdown(f'**Table {num} — Underlying measurements**')
    st.dataframe(table,use_container_width=True,hide_index=True,height=min(420,100+len(table)*35))
