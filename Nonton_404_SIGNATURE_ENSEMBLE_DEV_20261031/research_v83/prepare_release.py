from pathlib import Path
import shutil,json,hashlib,zipfile,time,sys,subprocess
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
D=Path('/mnt/data');SRC=D/'Nonton_V82_EXPERIMENTAL';DEST=D/'Nonton_V83_EXPERIMENTAL';WORK=D/'Nonton_v83_research'
if DEST.exists():shutil.rmtree(DEST)
shutil.copytree(SRC,DEST,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
# Production module, deliberately minimal, no retraining of 42 original models.
program='''"""Nonton v8.3: integer-count correction on v8.2 forecasts.
Preserves v8.1/v8.2 zero mask, 42 trained models untouched.
Usage: python -m app.v83_inference --variant nearest --verify
       python -m app.v83_inference --variant smape --out forecasts/custom.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from .catboost_inference import ROOT
from .v82_inference import predict as v82_predict

VARIANTS={
  'nearest': {'offset':0.50,'maximum':10.0,'filename':'v83_nearest10.csv'},
  'smape': {'offset':0.65,'maximum':10.0,'filename':'v83_smape10.csv'},
}

def snap_quantity(values,offset:float=0.50,maximum:float=10.0):
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Quantity must be nonnegative and finite')
    mask=(values>0)&(values<=maximum)
    result=values.copy()
    result[mask]=np.maximum(1.0,np.floor(values[mask]+offset))
    if not np.array_equal(result>0,values>0):
        raise AssertionError('Zero mask changed')
    return result

def run(variant:str='nearest',out=None,verify:bool=False,recompute_base:bool=False):
    if variant not in VARIANTS:raise ValueError('Unknown variant: '+str(variant))
    v82,baseline,alpha=v82_predict(recompute_base=recompute_base)
    cfg=VARIANTS[variant]
    result=snap_quantity(v82,cfg['offset'],cfg['maximum'])
    keys=pd.read_csv(ROOT/'training/keys.csv',dtype=str)
    if len(keys)!=79_101 or len(result)!=len(keys):raise AssertionError('Wrong target shape')
    if list(keys.columns)!=['week','warehouse_name','article']:raise AssertionError('Wrong key schema')
    if keys.duplicated().any() or len(keys.week.unique())!=9:raise AssertionError('Invalid keys')
    if not (result>=0).all() or not np.isfinite(result).all():raise AssertionError('Invalid quantities')
    if not np.array_equal(result>0,baseline>0):raise AssertionError('Official zero mask must be preserved')
    new=keys.copy();new['quantity']=result
    outfile=Path(out) if out is not None else ROOT/'forecasts'/cfg['filename']
    if verify:
       official=pd.read_csv(ROOT/'forecasts/v81_catboost.csv')
       if np.max(np.abs(official.quantity.to_numpy()-baseline))>1.e-3:raise AssertionError('Original not reproduced')
       if not outfile.exists():raise FileNotFoundError(outfile)
       saved=pd.read_csv(outfile,dtype={'week':str,'warehouse_name':str,'article':str})
       if not saved[['week','warehouse_name','article']].equals(new[['week','warehouse_name','article']]):raise AssertionError('Keys mismatch')
       max_error=float(np.max(np.abs(saved.quantity.to_numpy()-result)))
       if max_error>1.e-3:raise AssertionError(f'v8.3 reproduction mismatch {max_error}')
       print('PASS',variant,'rows',len(result),'total',round(result.sum(),2),'mask_ones',int((result>0).sum()),'max_abs_error',max_error)
       return new
    outfile.parent.mkdir(parents=True,exist_ok=True)
    new.to_csv(outfile,index=False,float_format='%.8g')
    print('SAVED',outfile,'rows',len(new),'total',round(result.sum(),2),'nonzero',int((result>0).sum()))
    return new

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,default='nearest')
 p.add_argument('--out');p.add_argument('--verify',action='store_true');p.add_argument('--recompute-base',action='store_true')
 args=p.parse_args();run(args.variant,args.out,args.verify,args.recompute_base)
'''
(DEST/'app/v83_inference.py').write_text(program,encoding='utf-8')
# Main runner with avoidance of PowerShell variable-colon traps.
runner='''param(
    [ValidateSet('nearest','smape')][string]$Variant = 'nearest',
    [switch]$Verify,
    [switch]$RecomputeBase
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Python = Join-Path $PSScriptRoot '.venv\\Scripts\\python.exe'
if (!(Test-Path $Python)) { throw 'Create .venv first: powershell -ExecutionPolicy Bypass -File .\\install_windows.ps1' }
$cmdargs = @('-m','app.v83_inference','--variant',$Variant)
if ($Verify) { $cmdargs += '--verify' }
if ($RecomputeBase) { $cmdargs += '--recompute-base' }
& $Python @cmdargs
if ($LASTEXITCODE -ne 0) { throw ('v8.3 exited with code '+ $LASTEXITCODE) }
'''
(DEST/'run_v83_windows.ps1').write_text(runner,encoding='utf-8-sig')
# Unit tests include exact zero mask and rounding behavior.
test='''from __future__ import annotations
import unittest
import numpy as np,pandas as pd
from app.v83_inference import snap_quantity,VARIANTS,run
class V83Tests(unittest.TestCase):
 def test_zeros(self):
  x=np.array([0.,1.,1.49,1.51,2.51,9.80,10.2]);v=snap_quantity(x)
  self.assertTrue(np.array_equal(x>0,v>0));self.assertTrue((v>=0).all())
 def test_nearest(self):
  x=np.array([1.49,1.5,1.51,2.45,2.65]);self.assertTrue(np.array_equal(snap_quantity(x),[1,2,2,2,3]))
 def test_upward_asymmetric(self):
  x=np.array([1.30,1.34,1.35,2.35]);self.assertTrue(np.array_equal(snap_quantity(x,.65),[1,1,2,3]))
 def test_cap(self):
  x=np.array([9.85,10.00001,11.65]);v=snap_quantity(x,.65,10.)
  self.assertAlmostEqual(v[0],10.);self.assertAlmostEqual(v[1],x[1]);self.assertAlmostEqual(v[2],x[2])
 def test_invalid(self):
  for arr in [[-1.,0.],[float('nan')],[float('inf')]]:
   with self.assertRaises(ValueError):snap_quantity(arr)
 def test_frozen_forecasts(self):
  for variant in VARIANTS:
   frame=run(variant,verify=True)
   self.assertEqual(len(frame),79101)
   self.assertTrue(frame.quantity.ge(0).all())
   self.assertEqual(frame.week.nunique(),9)
if __name__=='__main__':unittest.main()
'''
(DEST/'tests/test_v83.py').write_text(test,encoding='utf-8')
# Forecasts using saved v8.2 CSV, must verify the recomputed model later.
input_=pd.read_csv(SRC/'forecasts/v82_confidence_amount.csv',dtype={'week':str,'warehouse_name':str,'article':str})
assert len(input_)==79101
x=input_.quantity.to_numpy()
outputs={}
for name,off in [('nearest',.5),('smape',.65)]:
 mask=(x>0)&(x<=10)
 v=x.copy();v[mask]=np.maximum(1,np.floor(x[mask]+off))
 assert np.array_equal(v>0,x>0)
 f=input_.copy();f['quantity']=v
 path=DEST/'forecasts'/f'v83_{"nearest10" if name=="nearest" else "smape10"}.csv'
 f.to_csv(path,index=False,float_format='%.8g')
 outputs[name]=f
 print('FORECAST',name,'sum',round(v.sum(),2),'changed',np.count_nonzero(v!=x),'nonzero',int((v>0).sum()),'path',path)
# Research results and raw executable experiments.
res=DEST/'research_v83';res.mkdir(exist_ok=True)
for pattern in ['*.csv','*.py']:
 for path in WORK.glob(pattern):
  shutil.copyfile(path,res/path.name)
# weekly comparison chart, no manual styling
baseline=pd.read_csv(SRC/'forecasts/v81_catboost.csv')
plotdf=pd.DataFrame({'week':pd.to_datetime(baseline['week'].unique())})
for name,frame in [('v8.1 официальный',baseline),('v8.2',input_),('v8.3 nearest',outputs['nearest']),('v8.3 SMAPE',outputs['smape'])]:
 series=frame.groupby('week',sort=True).quantity.sum()
 plotdf[name]=plotdf.week.map(lambda v:series.loc[v.strftime('%Y-%m-%d')])
for col in plotdf.columns[1:]:plt.plot(plotdf.week,plotdf[col],marker='o',label=col)
plt.ylabel('Штук / неделя');plt.title('Нонтон: 9 недель прогноза, v8.1 — v8.3');plt.legend();plt.grid(alpha=.25);plt.xticks(plotdf.week,plotdf.week.dt.strftime('%d.%m'));plt.tight_layout()
plt.savefig(res/'weekly_forecasts.png',dpi=175);plt.close()
plotdf.assign(week=plotdf.week.dt.strftime('%Y-%m-%d')).to_csv(res/'weekly_forecasts.csv',index=False)
# Percent per fold summary
T=pd.read_csv(WORK/'threshold_snap_folds.csv');M=['SMAPE','WAPE','RMSE','Bias','RMSLE'];
chosen=['v81_t0.53_off0.0_cap10','v82_t0.53_off0.0_cap10','v82_t0.53_off0.5_cap10','v82_t0.53_off0.65_cap10']
pr=T[T.method.isin(chosen)].groupby('method')[M].mean().reindex(chosen)
mean_table=pr.to_markdown(floatfmt='.5f')
roll=pd.read_csv(WORK/'threshold_snap_rolling.csv');ri=roll[M].mean();rb=roll[['reference_'+k for k in M]].mean();w=[.5,.2,.15,.1,.05]
rel=float(sum(w[j]*ri[k]/rb['reference_'+k] for j,k in enumerate(M)))
# Markdown report: precise caveat forward feature existence
report=f'''# Nonton v8.3: реальный эксперимент коррекции целочисленного спроса

**Дата:** 31 октября 2026. **База:** загруженная сигнатурная модель v8.1 (39 LightGBM + 2 CatBoost), дополненная экспертной моделью количества v8.2 (ещё 1 CatBoost). 42 веса НЕ менялись.

## Оригинальная проблема
Все 426 112 положительных значений в исторической недельной матрице `quantity` являются целыми числами. Между тем v8.1 и v8.2 предсказывают вещественные количества. SMAPE даёт высокую цену ошибкам на низком количестве; при этом ложная положительная продажа стоит 200% независимо от величины. Проверена дискретная постобработка только ненулевых прогнозов.

## Результат (5 последовательных исторических OOF, 2025-06-16..2026-03-23)

{mean_table}

Для локальной взвешенной относительной оценки с весами 50/20/15/10/5 значения:
- v8.1 = 1.000000;
- v8.2 = 0.99880;
- **v8.3 ближайшее целое** = 0.9973566;
- **v8.3 вверх-смещённое округление** = 0.9972743.

Чем ниже, тем лучше. **Не официальный Score**. Параметр 0.65 выбирался по этим же окнам: переобучение выбора на них создаёт оптимистическое смещение.

**Более строгая последовательная проверка выбора политики:** на четырёх следующих окнах вариант выбирался только по более ранним OOF. Средняя относительная оценка = {rel:.7f} против 1.0000 у v8.1 (см. `threshold_snap_rolling.csv`). При этом исходные метамодели также разрабатывались на этих же исторических окнах — это не полностью независимый holdout.

## Формулы

Вариант `nearest` для 0 < q <= 10: `q_new = max(1, floor(q + 0.50))`, иначе q не меняется.

Вариант `smape` для 0 < q <= 10: `q_new = max(1, floor(q + 0.65))`, иначе q не меняется.

Оба сохраняют нулевую маску в точности и не допускают отрицательных прогнозов. **Не округлять значения > 10**: это сохраняет непрерывный прогноз больших объёмов и защищает RMSE.

## Испытания: что не подтверждено

- Отдельная LightGBM для предсказания выгоды выбора нуля / продажи в формате rolling-window на 4 окнах не победила v8.1 по совокупной метрике (см. `comparison_from_fold3.csv`).
- Дополнительная CatBoost количества с v8.2 сама по себе даёт небольшое улучшение, но ещё один LightGBM-эксперт количества на 4 следующих окнах не дал надёжного роста proxy.
- Пересобранный v7.4 META на тех же исходных OOF не улучшает CatBoost существенно при обычном смешивании (см. `blend_summary_from_fold2.csv`): корреляция высокая, эффект комплементарности слабый.
- Изменение порога вероятности продажи 0.53 на фиксированные 0.51 / 0.55 / 0.57 / 0.59 / 0.61 при сочетании с округлением не улучшило лучший пятиоконный proxy.
- Отдельный прогностик на уровне склад × категория был проверен в v8.2; простое линейное масштабирование часто ухудшало WAPE и Bias. Не включён.

## Что честно делать дальше

1. **Для конкурса:** сохранить официальный v8.1 как подтверждённую базу; запрашивать официальную оценку v8.3 только как отдельного кандидата, а не называть гарантированной победой.
2. **Для возможного прорыва:** независимая модель *изменения* абсолютного спроса по департамент × склад × категория. Обучить не на текущих итоговых 9-недельных целевых суммах, а на полном ряде закрытых прошлых 9-недельных горизонтов, используя все департаменты (не `OTHER`).
3. **После независимого эксперта:** обучить allocation/распределитель объёма внутри группы, не просто умножать все SKU на общую поправку.
4. **Для проверки:** вложенный walk-forward, последние окна без подбора гиперпараметров, отчёт по пяти метрикам + точность направления и объёма по складам/категориям; все as-of снимки строго раньше начала прогнозного периода.

### Важные ограничения

- Скрытые конкурсные факты неизвестны, **официальные метрики v8.3 не проверены**.
- SMAPE выигрыш достигнут на уже многократно анализировавшихся исторических окнах; финальный скрытый тест может повести себя иначе.
- Внешние продажи после 2026-07-26 НЕ использованы для создания конкурсных CSV.
- В этом эксперименте **не запускалось полное переобучение 39 LightGBM**; использованы проверенные сохранённые OOF и веса.
- Frozen API не поддерживает прогноз из произвольных новых sales + stock.
'''
(DEST/'RESEARCH_V83_RU.md').write_text(report,encoding='utf-8')
readme='''# Nonton v8.3 — две корректировки количества (эксперимент)

Базовая модель: 39 LightGBM + 2 CatBoost (v8.1) и ещё одна CatBoost условного количества (v8.2).
Дополнительного ML-обучения v8.3 **не требует**: изменяется только постобработка положительных прогнозов.
Официальный прогноз №5 сохранён и не перезаписывается.

## Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\\install_windows.ps1
powershell -ExecutionPolicy Bypass -File .\\run_v83_windows.ps1 -Variant nearest -Verify
powershell -ExecutionPolicy Bypass -File .\\run_v83_windows.ps1 -Variant smape -Verify
.\\.venv\\Scripts\\python.exe -m unittest tests.test_v83 -v
```

Проверка всех 39 исходных LightGBM, а не кэша:
```powershell
powershell -ExecutionPolicy Bypass -File .\\run_v83_windows.ps1 -Variant nearest -RecomputeBase -Verify
```

## Готовые CSV

- `forecasts/v83_nearest10.csv` — основная консервативная кандидатура (симметричное округление до 10).
- `forecasts/v83_smape10.csv` — альтернативная кандидатура с более сильным фокусом на SMAPE.
- `forecasts/v81_catboost.csv`, `forecasts/v82_confidence_amount.csv` — исходные контрольные точки.

## Подробности

- `RESEARCH_V83_RU.md` — числа и ограничения.
- `research_v83/` — исходники воспроизводимых тестов, пооконные CSV и график.
- **Проверка не гарантирует улучшение официального Score**; метрики локальных исторических окон и скрытого набора разные.
'''
(DEST/'README_V83_RU.md').write_text(readme,encoding='utf-8')
# Files provenance manifest
manifest={'base_release':'v8.2 full 2026-10-31','source':'user supplied v8.1, previously constructed v8.2',
 'variants':{},'frozen_weights':'39 LightGBM + 2 CatBoost + 1 positive qty CatBoost','target_cutoff':'2026-07-26'}
for name,f in outputs.items():
 path=DEST/'forecasts'/f'v83_{"nearest10" if name=="nearest" else "smape10"}.csv'
 manifest['variants'][name]={'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'rows':len(f),'total':float(f.quantity.sum()),'nonzero':int((f.quantity>0).sum())}
(DEST/'MANIFEST_V83.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# archive patch and full
patch_files=['app/v83_inference.py','tests/test_v83.py','run_v83_windows.ps1','README_V83_RU.md','RESEARCH_V83_RU.md','MANIFEST_V83.json']
patch_files += [str(p.relative_to(DEST)) for p in (DEST/'forecasts').glob('v83_*.csv')]
patch_files += [str(p.relative_to(DEST)) for p in (DEST/'research_v83').iterdir() if p.is_file()]
full_name=D/'Nonton_V83_EXPERIMENTAL_FULL_20261031.zip';patch_name=D/'Nonton_V83_EXPERIMENTAL_PATCH_20261031.zip'
for out,paths in [(patch_name,patch_files),(full_name,[str(p.relative_to(DEST)) for p in DEST.rglob('*') if p.is_file()])]:
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for file in paths:z.write(DEST/file,arcname='Nonton_V83_EXPERIMENTAL/'+file)
 print('ARCHIVE',out,'bytes',out.stat().st_size,'files',len(paths))
print('DONE',DEST)
