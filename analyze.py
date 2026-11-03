"""Reproduce revised results from supplied name-removed CSVs. Python >=3.9."""
from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import chi2, mannwhitneyu
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
O=P/'results'; O.mkdir(exist_ok=True);(P/'figures').mkdir(exist_ok=True)
f=pd.read_csv(P/'data-release/faculty.csv')
frame=pd.read_csv(P/'data-release/institution-frame.csv')
assert len(frame)==151 and frame.School.is_unique
assert frame.faculty_records.sum()==len(f) and frame.faculty_records.gt(0).sum()==137
audit=pd.read_csv(P/'data-release/gender-audit.csv')
assert audit.person_id.is_unique
assert audit.status.isin(['concordant','discordant','unresolved']).all()
assert audit.loc[audit.status.ne('unresolved'),'audited_gender'].isin(['F','M']).all()
declared=pd.read_csv(P/'data-release/label-corrections.csv')
assert set(declared.person_id)==set(audit.loc[audit.status.eq('discordant'),'person_id'])
assert audit.set_index('person_id').Gender.eq(f.set_index('person_id').Gender.reindex(audit.person_id).to_numpy()).all()
f['Gender_original']=f.Gender
corrections=audit[audit.status.eq('discordant')].set_index('person_id').audited_gender
f['Gender']=f.person_id.map(corrections).fillna(f.Gender)
f['profile_supported']=f.person_id.isin(audit.loc[audit.status.isin(['concordant','discordant']),'person_id'])
f.to_csv(O/'analysis_faculty.csv',index=False)
assert len(f)==553 and f.person_id.is_unique
assert f.Gender.isin(['M','F']).all() and f.tier.between(1,5).all()
f['F']=f.Gender.eq('F').astype(int)
f['R']=f.Rank.map({'Assistant':0,'Associate':1,'Full':2})
f['dec']=(f.PhD_Year-2000)/10
f['cohort']=pd.cut(f.PhD_Year,[0,1999,2009,2019,2100],labels=['1999 or earlier','2000--2009','2010--2019','2020 and later'])
f['cohort_label']=f.cohort.astype('string').fillna('Year missing')
d=f.dropna(subset=['PhD_Year']).copy();assert len(d)==518
rows=[]; models={}
def fit(name,formula,data,cluster=False):
    kw={'cov_type':'cluster','cov_kwds':{'groups':data.institution_id}} if cluster else {}
    m=smf.logit(formula,data).fit(disp=False,**kw)
    assert m.mle_retvals['converged']
    models[name]=m
    ci=m.conf_int()
    for term in m.params.index:
        if term=='Intercept':continue
        rows.append(dict(model=name,term=term,n=int(m.nobs),OR=float(np.exp(m.params[term])),lo=float(np.exp(ci.loc[term,0])),hi=float(np.exp(ci.loc[term,1])),p=float(m.pvalues[term])))
    return m
fit('cohort','F ~ dec',d)
fit('cohort_original_labels','F ~ dec',d.assign(F=d.Gender_original.eq('F').astype(int)))
fit('rank_all','F ~ R',f)
fit('rank_complete','F ~ R',d)
fit('joint','F ~ dec + R',d)
fit('rank_categorical','F ~ dec + C(Rank, Treatment(reference="Assistant"))',d)
fit('cohort_categorical','F ~ C(cohort)',d)
fit('tier','F ~ tier',f)
fit('tier_cohort','F ~ tier + dec',d)
fit('tier_categorical','F ~ C(tier) + dec',d)
fit('training_adjusted','F ~ dec + C(training_group)',d)
fit('UG_training_adjusted','F ~ dec + C(UG_training_group)',d)
fit('exclude_unresolved','F ~ dec',d[~d.person_id.isin(audit.loc[audit.status.eq('unresolved'),'person_id'])])
fit('cohort_clustered','F ~ dec',d,True)
fit('joint_clustered','F ~ dec + R',d,True)
fit('tier_cohort_clustered','F ~ tier + dec',d,True)
fit('cohort_pre2020','F ~ dec',d[d.PhD_Year<2020])
fit('cohort_2000plus','F ~ dec',d[d.PhD_Year>=2000])
fit('rank_overlap','F ~ C(cohort) + C(Rank, Treatment(reference="Assistant"))',d[d.PhD_Year.between(2000,2019)].copy().assign(cohort=lambda x:x.cohort.cat.remove_unused_categories()))
out=pd.DataFrame(rows);out.to_csv(O/'models.csv',index=False)
for key,col in [('cohort','cohort'),('rank','Rank'),('tier','tier'),('missing_year','cohort_label')]:
    tab=f.groupby(col,observed=True).F.agg(['sum','size','mean']);tab.to_csv(O/f'{key}_shares.csv')
pd.crosstab(d.cohort,d.Rank).to_csv(O/'cohort_rank_overlap.csv')
f.groupby(['cohort_label','Gender'],observed=True).first_hire_recorded.agg(['sum','size','mean']).to_csv(O/'first_hire_coverage.csv')
# Recorded first-hire values are not independently validated employment histories.
z=f[f.first_hire_recorded.eq(1)].dropna(subset=['first_hire_tier'])
w=z[z.F.eq(1)].first_hire_tier;m=z[z.F.eq(0)].first_hire_tier
first={'recorded':int(f.first_hire_recorded.sum()),'missing':int(f.first_hire_recorded.eq(0).sum()),'tier_matched':len(z),'women_n':len(w),'men_n':len(m),'women_mean':float(w.mean()),'men_mean':float(m.mean()),'MWU_p':float(mannwhitneyu(w,m,alternative='two-sided').pvalue)}
# Missing-year bounds: assign all 35 to each cohort in turn, treating labels as fixed.
bounds=[]
miss=f[f.PhD_Year.isna()]
for c,g in d.groupby('cohort',observed=True):
    bounds.append({'cohort':str(c),'observed_n':len(g),'observed_F':int(g.F.sum()),'min_share':g.F.sum()/(len(g)+int(miss.F.eq(0).sum())),'max_share':(g.F.sum()+miss.F.sum())/(len(g)+int(miss.F.sum()))})
pd.DataFrame(bounds).to_csv(O/'missing_year_bounds.csv',index=False)
# Directed stress scenarios; never interpreted as measured classification error rates.
# At most ceil(rate * high-confidence stratum size) labels change in each stratum.
# Older M -> F and recent F -> M attenuate the cohort contrast. To reduce the
# continuous-year slope, choose earliest M and latest F within each stratum.
sens=[]
for strat,group in [('cohort','cohort'),('doctoral_training','training_group'),('UG_training','UG_training_group')]:
 for rate in [.05,.10]:
  q=d.copy();changed=0
  for _,g in q[(q.Gender_Method.eq('LLM_firstpass')&~q.profile_supported)].groupby(group,observed=True):
   budget=int(np.ceil(rate*len(g)))
   candidates=g[((g.PhD_Year<2010)&g.F.eq(0))|((g.PhD_Year>=2010)&g.F.eq(1))].copy()
   candidates['distance']=(candidates.PhD_Year-2010).abs()
   idx=candidates.sort_values(['distance','person_id'],ascending=[False,True]).head(budget).index
   q.loc[idx,'F']=1-q.loc[idx,'F'];changed+=len(idx)
  mm=smf.logit('F ~ dec',q).fit(disp=False);ci=np.exp(mm.conf_int().loc['dec'])
  sens.append({'strata':strat,'nominal_rate':rate,'labels_changed':changed,'OR':np.exp(mm.params['dec']),'lo':ci.iloc[0],'hi':ci.iloc[1]})
# Concentrate a globally capped error budget in institutions with the most
# high-confidence records; select adverse candidates within those institutions.
for rate in [.05,.10]:
 q=d.copy();h=q[(q.Gender_Method.eq('LLM_firstpass')&~q.profile_supported)]
 sizes=h.institution_id.value_counts()
 h=h[((h.PhD_Year<2010)&h.F.eq(0))|((h.PhD_Year>=2010)&h.F.eq(1))].copy()
 h['institution_size']=h.institution_id.map(sizes);h['distance']=(h.PhD_Year-2010).abs()
 idx=h.sort_values(['institution_size','distance','person_id'],ascending=[False,False,True]).head(int(np.ceil(rate*len(q[(q.Gender_Method.eq('LLM_firstpass')&~q.profile_supported)])))).index
 q.loc[idx,'F']=1-q.loc[idx,'F']
 mm=smf.logit('F ~ dec',q).fit(disp=False);ci=np.exp(mm.conf_int().loc['dec'])
 sens.append({'strata':'institution_concentrated','nominal_rate':rate,'labels_changed':len(idx),'OR':np.exp(mm.params['dec']),'lo':ci.iloc[0],'hi':ci.iloc[1]})
# All audited high-confidence cases can instead be excluded; avoid using
# outcome agreement to decide which sample members are retained.
sample=pd.read_csv(P/'data-release/gender-audit.csv')
fit_sample=smf.logit('F ~ dec',d[~d.person_id.isin(sample.person_id)]).fit(disp=False)
cs=np.exp(fit_sample.conf_int().loc['dec'])
sens.append({'strata':'exclude_audit_sample','nominal_rate':0,'labels_changed':0,'OR':np.exp(fit_sample.params['dec']),'lo':cs.iloc[0],'hi':cs.iloc[1]})
pd.DataFrame(sens).to_csv(O/'classification_sensitivity.csv',index=False)
# Display cohort observations, with no future years or fitted forecast curve.
g=d.groupby('cohort',observed=True).F.agg(['sum','size','mean']);lo,hi=proportion_confint(g['sum'],g['size'],method='wilson')
fig,ax=plt.subplots(figsize=(7,4.2));x=np.arange(len(g));y=g['mean'].to_numpy()*100
ax.errorbar(x,y,yerr=[y-100*lo,100*hi-y],fmt='o',capsize=5,color='#195a7c',ms=7)
for i,(_,r) in enumerate(g.iterrows()):ax.annotate(f"{r['mean']:.1%}\nn={int(r['size'])}",(i,100*float(hi.iloc[i])),xytext=(0,8),textcoords='offset points',ha='center',fontsize=10)
ax.set_xticks(x);ax.set_xticklabels(g.index);ax.set_ylim(0,60);ax.set_xlim(-.45,3.45)
ax.set_ylabel('Share classified female (%)');ax.set_xlabel('Doctoral completion cohort among faculty on the roster')
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(P/'figures/Figure_1.jpeg',dpi=600);fig.savefig(P/'figures/Figure_1.pdf');plt.close(fig)
checks={}
for name,a,b,df in [('rank_equal_steps','joint','rank_categorical',1),('tier_equal_steps','tier_cohort','tier_categorical',3)]:
 lr=2*(models[b].llf-models[a].llf);checks[name]={'LR':lr,'df':df,'p':float(chi2.sf(lr,df))}
summary={'n':len(f),'F':int(f.F.sum()),'institutions':int(f.institution_id.nunique()),'year_missing':len(miss),'year_missing_F':int(miss.F.sum()),'year_min':float(d.PhD_Year.min()),'year_max':float(d.PhD_Year.max()),'first_hire':first,'categorical_checks':checks,'audit':sample.status.value_counts().to_dict(),'classification_changes':int(len(corrections)),'F_original':int(f.Gender_original.eq('F').sum()),'versions':{k:v for k,v in [('python',platform.python_version()),('numpy',np.__version__),('pandas',pd.__version__),('scipy',scipy.__version__),('statsmodels',statsmodels.__version__),('matplotlib',matplotlib.__version__)]}}
(O/'summary.json').write_text(json.dumps(summary,indent=2))
# Generated LaTeX: single numerical source for manuscript and supplement.
def tex(s):return str(s).replace('_',r'\_')
def table(path,header,body,align):
 (O/path).write_text('\\begin{tabular}{'+align+'}\n\\toprule\n'+header+r' \\'+'\n\\midrule\n'+'\n'.join(body)+'\n\\bottomrule\n\\end{tabular}\n')
def ci_text(r):
 lo=f'{r.lo:.3f}' if abs(r.lo-1)<.005 else f'{r.lo:.2f}'
 hi=f'{r.hi:.3f}' if abs(r.hi-1)<.005 else f'{r.hi:.2f}'
 return f'{r.OR:.2f} ({lo}--{hi})'
def val(name,term):
 r=out[(out.model==name)&(out.term==term)].iloc[0];return ci_text(r)
mainrows=[r'Doctoral year (per decade) & '+val('cohort','dec')+' & --- & --- & '+val('joint','dec')+r' \\',r'Rank (per step) & --- & '+val('rank_all','R')+' & '+val('rank_complete','R')+' & '+val('joint','R')+r' \\',r'$n$ & 518 & 553 & 518 & 518 \\']
table('main_models.tex',r' & Cohort & Rank (all) & Rank (complete) & Joint',mainrows,'lcccc')
short=lambda term: term.replace('C(Rank, Treatment(reference="Assistant"))[T.','Rank: ').replace('C(cohort)[T.','Cohort: ').replace('C(training_group)[T.','Doctoral origin: ').replace('C(UG_training_group)[T.','UG origin: ').replace('C(tier)[T.','Tier: ').replace(']','')
pformat=lambda p: '$<0.0001$' if p<.0001 else f'{p:.4f}'
body=[tex(r.model)+' & '+tex(short(r.term))+' & '+str(r.n)+' & '+ci_text(r)+f' & {pformat(r.p)}'+r' \\' for r in out.itertuples()]
table('all_models.tex','Model & Term & $n$ & OR (95\\% CI) & $p$',body,'lllrr')
body=[tex(r.strata)+f' & {r.nominal_rate:.0%}'.replace('%',r'\%')+f' & {r.labels_changed} & '+ci_text(r)+r' \\' for r in pd.DataFrame(sens).itertuples()]
table('sensitivity.tex',r'Strata & Nominal rate & Changed & OR (95\% CI)',body,'llrr')
coverage=pd.read_csv(O/'first_hire_coverage.csv');body=[tex(r.cohort_label)+' & '+r.Gender+f' & {r.size} & {r.sum} & {r.size-r.sum}'+r' \\' for r in coverage.itertuples()]
table('first_hire_coverage.tex','Cohort & Classification & Total & Recorded & Missing',body,'llrrr')
body=[tex(r.School)+' & '+str(r.US_News_Rank)+r' \\' for r in frame[frame.faculty_records.eq(0)].itertuples()]
table('unrepresented_institutions.tex','Institution & Supplied rank',body,'lr')
print(json.dumps(summary,indent=2));print(out[['model','term','n','OR','lo','hi']].to_string(index=False));print(pd.DataFrame(sens).to_string(index=False))

# Audited sample composition: descriptive checks, not a weighted accuracy estimate.
for field in ['cohort','Gender','doctoral_training_country','undergraduate_training_country']:
    audit.groupby([field,'status'],dropna=False).size().unstack(fill_value=0).to_csv(O/('audit_by_'+field+'.csv'))
# Sensitivity to the single confirmed correction and remaining unresolved cases is
# reported alongside baseline, rather than leaving known bad labels in analysis.
def scalar_or(name,term='dec'):
    r=out[(out.model==name)&out.term.eq(term)].iloc[0]
    return ci_text(r)
macros={
 'FemaleN':str(int(f.F.sum())), 'FemaleShare':f'{100*f.F.mean():.1f}',
 'CohortOR':scalar_or('cohort'),'JointYearOR':scalar_or('joint'),
 'RankAllOR':scalar_or('rank_all','R'),'RankCompleteOR':scalar_or('rank_complete','R'),
 'RankJointOR':scalar_or('joint','R'),'TierOR':scalar_or('tier','tier'),
 'TierAdjustedOR':scalar_or('tier_cohort','tier'),'RecentOR':scalar_or('cohort_2000plus'),
 'PreTwentyOR':scalar_or('cohort_pre2020'),'OriginalCohortOR':scalar_or('cohort_original_labels'),
 'UnresolvedOR':scalar_or('exclude_unresolved'),'UGAdjustedOR':scalar_or('UG_training_adjusted'),
 'FullCategoricalOR':scalar_or('rank_categorical','C(Rank, Treatment(reference="Assistant"))[T.Full]'),
 'OverlapOR':scalar_or('rank_overlap','C(Rank, Treatment(reference="Assistant"))[T.Full]'),
}
for rank in ['Assistant','Associate','Full']:
 rr=f[f.Rank.eq(rank)];macros[rank+'FemaleShare']=f'{100*rr.F.mean():.1f}'
for i,(_,rr) in enumerate(g.iterrows(),1):
 macros['Cohort'+['One','Two','Three','Four'][i-1]+'FemaleShare']=f'{100*rr["mean"]:.1f}';macros['Cohort'+['One','Two','Three','Four'][i-1]+'FemaleN']=str(int(rr['sum']))
for t,rr in f.groupby('tier'):
 macros['Tier'+['One','Two','Three','Four','Five'][int(t)-1]+'Share']=f'{100*rr.F.mean():.1f}'
(O/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')

# Marginal missing-year bounds and ordinal checks for the supplement.
b=pd.DataFrame(bounds)
body=[str(r.cohort)+f' & {r.observed_F}/{r.observed_n} & {100*r.min_share:.1f}--{100*r.max_share:.1f}'+r' \\' for r in b.itertuples()]
table('missing_year_bounds.tex',r'Cohort & Female / known year & Marginal range (\%)',body,'lrr')
# Compact descriptive counts for the article's sampling and tier interpretation.
r=f.groupby('tier').F.agg(['sum','size','mean'])
body=[f'{int(i)} & {int(v["sum"])}/{int(v["size"])} & {100*v["mean"]:.1f}'+r' \\' for i,v in r.iterrows()]
table('tier_shares.tex',r'Tier & Female / total & Female share (\%)',body,'lrr')
