"""Reproduce the two-stage audit selection; source adjudications remain evidence data."""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parent
f=pd.read_csv(P/'data-release/faculty.csv')
f['cohort']=pd.cut(f.PhD_Year,[0,1999,2009,2019,2100],labels=['pre2000','2000s','2010s','2020plus']).astype('string').fillna('missing')
h=f[f.Gender_Method.eq('LLM_firstpass')]
first=pd.concat([g.sample(min(2,len(g)),random_state=20261029) for _,g in h.groupby(['cohort','Gender'])])
h=h[~h.person_id.isin(first.person_id)]
extra=h[h.doctoral_training_country.notna()&h.doctoral_training_country.ne('USA')]
parts=[first,extra]
for country in ['China','India','Iran']:
 g=h[h.undergraduate_training_country.eq(country)&~h.person_id.isin(extra.person_id)]
 parts.append(g.sample(min(2,len(g)),random_state=20261029))
selected=pd.concat(parts)
audit=pd.read_csv(P/'data-release/gender-audit.csv')
assert selected.person_id.is_unique and len(first)==19 and len(extra)==17
assert len(selected)==42 and set(selected.person_id)==set(audit.person_id)
print('Audit selection reproduced: 19 initial +17 non-US doctorate +6 undergraduate-training =42 unique IDs.')
