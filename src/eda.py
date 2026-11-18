"""Generate reproducible exploratory-analysis tables and figures for Home Credit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'input/train.csv'
TEST = ROOT / 'input/test.csv'
OUT = ROOT / 'outputs/eda'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    y = train.TARGET
    num = train.drop(columns=['TARGET', 'SK_ID_CURR']).select_dtypes(include=np.number)
    cat = train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Reproducible tables
    missing = pd.DataFrame({'train_missing': train.isna().sum(), 'train_pct': train.isna().mean()*100,
                            'test_missing': test.isna().sum(), 'test_pct': test.isna().mean()*100})
    missing.sort_values('train_pct', ascending=False).to_csv(OUT/'missingness.csv')
    profile = {'train_shape': list(train.shape), 'test_shape': list(test.shape),
               'target_counts': {str(k): int(v) for k,v in y.value_counts().sort_index().items()},
               'positive_rate': float(y.mean()), 'numeric_predictors': int(len(num.columns)),
               'categorical_predictors': int(len(cat)),
               'categorical_cardinality': {c:int(train[c].nunique(dropna=False)) for c in cat},
               'id_unique_train': bool(train.SK_ID_CURR.is_unique),
               'id_unique_test': bool(test.SK_ID_CURR.is_unique),
               'id_overlap': int(len(set(train.SK_ID_CURR).intersection(test.SK_ID_CURR))),
               'test_ids_match_sample_order': bool(pd.read_csv(ROOT/'input/sample_submission.csv').SK_ID_CURR.equals(test.SK_ID_CURR))}
    (OUT/'eda_summary.json').write_text(json.dumps(profile, indent=2))

    sns.set_theme(style='whitegrid', context='notebook')
    def save(name):
        plt.tight_layout(); plt.savefig(OUT/name, dpi=160, bbox_inches='tight'); plt.close()

    # 1 target distribution
    plt.figure(figsize=(6,4)); ax=sns.countplot(x=y, hue=y, palette='Set2', legend=False)
    ax.set(title='Target class balance', xlabel='TARGET (1 = payment difficulties)', ylabel='Applications')
    for p in ax.patches: ax.annotate(f'{int(p.get_height()):,}', (p.get_x()+p.get_width()/2,p.get_height()), ha='center', va='bottom', fontsize=9)
    save('01_target_balance.png')

    # 2 missingness
    m=missing.loc[missing.train_missing>0].sort_values('train_pct').tail(20)
    plt.figure(figsize=(9,7)); plt.barh(m.index, m.train_pct, label='Train'); plt.barh(m.index, m.test_pct, alpha=.65, label='Test')
    plt.xlabel('Missing values (%)'); plt.title('Features with highest missingness (top 20)'); plt.legend(); save('02_missingness.png')

    # 3 numerical distributions for important numeric fields
    cols=[c for c in ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','OWN_CAR_AGE'] if c in train]
    fig, axes=plt.subplots(5,2,figsize=(12,18));
    for ax,c in zip(axes.flat,cols):
        s=train[c].replace([np.inf,-np.inf],np.nan).dropna()
        if c.startswith('AMT_') and (s>=0).all(): s=np.log1p(s)
        sns.histplot(s, bins=45, ax=ax, color='#4C78A8'); ax.set_title(c + (' (log1p)' if c.startswith('AMT_') else ''))
    for ax in axes.flat[len(cols):]: ax.remove()
    fig.suptitle('Selected numerical feature distributions', y=1.002); fig.tight_layout(); fig.savefig(OUT/'03_numeric_distributions.png',dpi=160,bbox_inches='tight'); plt.close(fig)

    # 4 employment sentinel
    if 'DAYS_EMPLOYED' in train:
        counts=pd.Series({'Negative (observed duration)':int((train.DAYS_EMPLOYED<0).sum()),
                          'Positive sentinel-like':int((train.DAYS_EMPLOYED>0).sum()),
                          'Zero':int((train.DAYS_EMPLOYED==0).sum()),
                          'Missing':int(train.DAYS_EMPLOYED.isna().sum())})
        plt.figure(figsize=(8,4)); sns.barplot(x=counts.index,y=counts.values,hue=counts.index,palette='Set2',legend=False)
        plt.xticks(rotation=15,ha='right'); plt.ylabel('Rows'); plt.title('DAYS_EMPLOYED value-pattern audit'); save('04_employment_sentinel.png')

    # 5 correlation heatmap
    corr_cols=[c for c in ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','DAYS_BIRTH','DAYS_EMPLOYED','CNT_CHILDREN','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_POPULATION_RELATIVE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in train]
    plt.figure(figsize=(11,9)); sns.heatmap(train[corr_cols].corr(), cmap='vlag', center=0, annot=True, fmt='.2f', square=True, cbar_kws={'shrink':.8})
    plt.title('Correlation among selected numeric features'); save('05_numeric_correlations.png')

    # 6 target rate by categories, grouped small classes displayed with sample count
    for i,c in enumerate(['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE']):
        if c not in train: continue
        stats=train.groupby(c, dropna=False).TARGET.agg(['mean','count']).sort_values('mean')
        stats=stats[stats['count']>=30]
        if len(stats)==0: continue
        plt.figure(figsize=(10,max(4,0.32*len(stats))))
        ax=sns.barplot(x=stats['mean']*100,y=stats.index,color='#59A14F')
        ax.set(xlabel='Observed TARGET=1 rate (%)',ylabel='',title=f'Target rate by {c} (categories with n ≥ 30)')
        for j,(_,r) in enumerate(stats.iterrows()): ax.text(r['mean']*100+.08,j,f"n={int(r['count']):,}",va='center',fontsize=8)
        save(f'06_target_rate_{i+1}_{c.lower()}.png')

    # 7 train-test numerical shift standardized mean difference
    rows=[]
    for c in num.columns:
        a=pd.to_numeric(train[c],errors='coerce'); b=pd.to_numeric(test[c],errors='coerce')
        pooled=np.sqrt((a.var()+b.var())/2)
        smd=(b.mean()-a.mean())/pooled if pooled and np.isfinite(pooled) else 0
        rows.append((c,smd,abs(smd)))
    drift=pd.DataFrame(rows,columns=['feature','smd_test_minus_train','abs_smd']).sort_values('abs_smd',ascending=False)
    drift.to_csv(OUT/'numeric_train_test_drift.csv',index=False)
    top=drift.head(15).sort_values('smd_test_minus_train')
    plt.figure(figsize=(9,7)); plt.barh(top.feature,top.smd_test_minus_train); plt.axvline(0,color='black',lw=.8)
    plt.xlabel('Standardized mean difference (test − train)'); plt.title('Largest numerical train/test shifts'); save('07_numeric_train_test_drift.png')

    # 8 categorical distribution shifts: max absolute category proportion gap
    catrows=[]
    for c in cat:
        p=train[c].fillna('__NA__').value_counts(normalize=True); q=test[c].fillna('__NA__').value_counts(normalize=True)
        gap=(p-q).abs().reindex(p.index.union(q.index),fill_value=0).sum()/2
        catrows.append((c,gap))
    cd=pd.DataFrame(catrows,columns=['feature','total_variation_distance']).sort_values('total_variation_distance')
    cd.to_csv(OUT/'categorical_train_test_drift.csv',index=False)
    plt.figure(figsize=(8,5)); plt.barh(cd.feature,cd.total_variation_distance); plt.xlabel('Total variation distance'); plt.title('Categorical train/test distribution differences'); save('08_categorical_train_test_drift.png')

    # 9 missingness comparison
    mm=missing.loc[missing.train_pct>0].copy(); mm['gap_pp']=mm.test_pct-mm.train_pct
    mm=mm.reindex(mm.gap_pp.abs().sort_values(ascending=False).head(15).index).sort_values('gap_pp')
    plt.figure(figsize=(9,6)); plt.barh(mm.index,mm.gap_pp); plt.axvline(0,color='black',lw=.8)
    plt.xlabel('Test missingness − train missingness (percentage points)'); plt.title('Largest train/test missingness gaps'); save('09_missingness_shift.png')

    # 10 EXT_SOURCE by target
    ext=[c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in train]
    fig,axes=plt.subplots(1,len(ext),figsize=(15,4))
    for ax,c in zip(np.atleast_1d(axes),ext):
        sns.kdeplot(data=train,x=c,hue='TARGET',common_norm=False,fill=False,ax=ax); ax.set_title(c)
    fig.suptitle('EXT_SOURCE distributions by target'); fig.tight_layout(); fig.savefig(OUT/'10_ext_source_by_target.png',dpi=160,bbox_inches='tight'); plt.close(fig)

    print(json.dumps(profile,indent=2))
    print(f'Wrote EDA figures and tables to {OUT}')

if __name__=='__main__': main()
