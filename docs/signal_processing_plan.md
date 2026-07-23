# План исследования и развития PSD_ML

Обновлено: 2026-07-21. Исходный план был создан в задаче Codex
[«Спланировать обработку сигналов»](thread://019f6bc4-aca8-7b21-9d28-a9119618ea4d).
Новая программа учитывает обсуждение «VAE для PSD анализа» и текущий аудит проекта.

Теоретический ликбез и архитектурные варианты собраны в
[`Идеи развития PSD_ML.md`](../Идеи%20развития%20PSD_ML.md).

## Цели

1. Получить принципиально новую и максимально эффективную метрику разделения нейтронных
   и гамма-сигналов по форме импульса. VAE является одним из кандидатов, но выбор метода
   определяется общим benchmark.
2. Интерпретировать найденные скрытые параметры через изменения формы импульса.
3. Интерпретировать скрытые параметры через измеряемые физические и технические
   величины.
4. Найти минимальное скрытое пространство, в котором информация о частице, заряде,
   форме хвоста, шуме и условиях регистрации максимально разделена.
5. Выделить признак `z_particle`, информативный для \(n/\gamma\) и по возможности
   инвариантный к запуску, детектору и условиям.
6. После набора факторного набора данных реализовать condition-aware PSD через явное
   conditioning по напряжению, температуре, полю, детектору и другим известным условиям.

## Критерии успеха

Новая метрика считается улучшением только если на независимых полных запусках она лучше
классического PSD по заранее выбранной совокупности показателей:

- gamma leakage при фиксированной neutron efficiency;
- ложные нейтроноподобные события в час на gamma/background;
- качество отдельно по калиброванной энергии или, до калибровки, по `Qlong`;
- стабильность по запускам, каналам, напряжению, температуре и полю;
- калибровка, неопределённость, OOD и доля `unknown`;
- вычислительная стоимость на целевой аппаратуре.

Пока нет event-level neutron truth, результаты называются shape score или
нейтроноподобными кандидатами. Метки Co/Cf не подменяют истинный тип частицы.

## Этап 0. Устранить ограничения экспериментального дизайна

Перед финальным supervised и condition-aware исследованием необходимо:

1. выяснить семантику `RAW/FILTERED/UNFILTERED` и acquisition-time PSD cuts;
2. получить background и независимые повторные Co/Cf runs;
3. получить независимый neutron-якорь: tagged/TOF/совпадения либо обоснованную разметку;
4. физически откалибровать `Qlong` или явно оставить его зарядовым proxy;
5. провести рандомизированные повторные серии по безопасной сетке \(U,T,B\), сохраняя
   setpoint/readback, время, rate и конфигурацию;
6. по возможности сформировать перекрёстную матрицу ФЭУ × сцинтиллятор.

До появления этих данных разрешены разведочный поиск формы и mixture classification,
но не заявления о neutron efficiency или универсальном переносе.

## Этап 1. Зафиксировать воспроизводимый вход

Использовать существующий pipeline: provenance CSV↔ROOT, 12-sample median baseline,
инверсию полярности, CFD-50 alignment к sample 20, QC и раздельные amplitude-retaining
и peak-normalized ветви.

Для каждого события сохранять:

- raw и обработанную форму;
- `Qlong`, `Qshort`, amplitude, classical PSD;
- baseline, slope, RMS, CFD shift и все QC-флаги;
- run, source mixture, channel и полное описание детектора;
- \(U,T,B\), rate и другие условия, когда они появятся;
- происхождение event-level label или статус `unknown`.

Разбиения создаются только по полным runs/days. Все нормировки, окна, feature selection
и модели обучаются без доступа к test runs.

## Этап 2. Единый benchmark кандидатов `new_PSD`

На одинаковых splits и стратах сравнить:

1. CoMPASS classical PSD;
2. текущий фиксированный `shape_score` как демонстрационный baseline;
3. оптимизированные multi-window/time-domain признаки;
4. logistic regression и gradient boosting на физических признаках;
5. LDA/PLS, PCA и linear AE;
6. shapelets, wavelets и MiniROCKET;
7. supervised contrastive/metric learning;
8. компактную 1D CNN/TCN;
9. AE, VAE, \(\beta\)-VAE и CVAE;
10. factorized/adversarial модели после появления достаточных labels и domains.

Гиперпараметры подбираются nested validation только по train/validation runs. Итоговый
test выполняется один раз на отложенных runs. Простые методы не являются формальностью:
новизна должна быть доказана сравнением с сильным оптимизированным PSD baseline.

## Этап 3. Построить и проверить новую метрику

Для каждой модели определить скалярный score \(s(X,c)\): отдельный latent, линейную или
нелинейную проекцию малого латента либо классификационный logit.

Проверить:

- ROC/PR и рабочие точки по каждому `Qlong`/energy-интервалу;
- Co и background false-positive rate;
- стабильность score между runs, channels, seed и preprocessing variants;
- отсутствие разделения по предимпульсному участку и техническим признакам;
- вклад каждого временного интервала через masking/occlusion;
- сравнение с существующей двухветвевой Cf-структурой без назначения ей particle labels.

Победитель определяется по независимой валидации, а не по визуально красивому latent
plot или reconstruction loss.

## Этап 4. Интерпретация скрытых параметров через форму

Для каждого латента или learned direction:

1. выполнять traversal в диапазоне реального posterior;
2. показывать исходную и реконструированную форму для конкретных событий;
3. вычислять \(\partial\hat X(t)/\partial z_j\) или конечные разности;
4. сравнивать медианы и 10–90% bands при фиксированном `Qlong`;
5. маскировать front/peak/tail и повторно оценивать score;
6. сопоставлять с tail fraction, rise/fall times, decay fits, wavelets и shapelets;
7. проверять, сохраняется ли смысл между seed и held-out runs.

Интерпретация относится к направлению изменения формы, а не к номеру координаты: оси
латентного пространства могут переставляться, менять знак и вращаться.

## Этап 5. Интерпретация через физические величины

Сформировать event-level таблицу \((\mu,\sigma)\) или deterministic latent вместе с
зарядом, классическим PSD, каналом, запуском, \(U,T,B\), baseline, шумом и timing.

Для каждого латента оценить:

- Pearson/Spearman и mutual information;
- предсказуемость каждого известного фактора простым probe-классификатором;
- зависимости внутри узких charge/energy strata;
- partial dependence/conditional analyses;
- стабильность на held-out runs, conditions и detectors;
- неопределённость \(\sigma_j(X)\), особенно при low SNR.

Высокая корреляция — диагностический факт, не доказательство причинности.

## Этап 6. Найти минимальное факторизованное пространство

Сравнить последовательность размерностей, например
\(d\in\{1,2,3,4,6,8,12,16\}\), и несколько семейств моделей. Для каждой построить
Pareto-таблицу:

- discrimination;
- reconstruction важных частей формы;
- particle/condition/domain leakage;
- disentanglement/factor predictability;
- seed/run stability;
- uncertainty и latency.

Выбрать наименьшее \(d\), после которого увеличение пространства не даёт практически
значимого улучшения и ухудшает интерпретируемость. Не считать низкий reconstruction loss
достаточным критерием.

## Этап 7. Выделить `z_particle`

После появления event-level anchors разделить представление:

\[
z=(z_{particle},z_{shape},z_{nuisance}).
\]

Обучать `z_particle` с supervised или weakly supervised particle loss. Одновременно
ограничивать информацию о run/channel/conditions с помощью domain-adversarial loss,
MMD/CORAL, contrastive constraints или class-conditional alignment.

Критерии:

- particle probe по `z_particle` силён на held-out runs;
- run/channel/condition probes близки к случайному уровню;
- particle-информация не переместилась целиком в `z_nuisance`;
- adversarial alignment не стёр редкий Cf-компонент;
- результат устойчив к charge/energy и low-SNR strata.

Если истинных labels недостаточно, сохранять `unknown` и не принуждать всю смесь к
псевдометкам.

## Этап 8. Condition-aware PSD

После набора данных на сетке условий обучить

\[
s=f(X,Qlong,detector,U,T,B,rate,\ldots)
\]

и/или CVAE

\[
q(z\mid X,c),\qquad p(X\mid z,c).
\]

Сравнить по leave-one-condition-out и полным held-out runs:

1. неизменную модель;
2. физическую gain/time correction;
3. явные condition inputs;
4. detector/condition adapters;
5. CORAL/MMD/domain-adversarial и class-conditional adaptation.

Явное conditioning является первым вариантом. Безусловное выравнивание Co и Cf
запрещено, поскольку может удалить физически полезный mixture shift. Test-time adaptation
разрешается только как поздний guarded эксперимент с frozen reference, bounded updates,
OOD/unknown, журналом версий, rollback и shadow mode.

## Этап 9. Перенос на новые детекторы

Исследовать shared backbone + detector embedding/adapter. Проверять:

- leave-one-detector-out zero-shot;
- calibration-only перенос без particle labels;
- few-shot настройку только малого adapter;
- зависимость требуемого объёма калибровки от новизны сборки;
- OOD/unknown для сборок вне обучающего диапазона.

Целевой результат — одна базовая модель и короткая контролируемая калибровка новой
сборки, а не обещание универсального PSD без настройки.

## Контрольные точки

1. **Data gate:** есть независимые runs, anchors и метаданные условий?
2. **Benchmark gate:** новая метрика превосходит оптимизированный classical PSD?
3. **Interpretability gate:** форма и известные факторы объясняют latent на held-out data?
4. **Minimality gate:** выбрана минимальная стабильная размерность по Pareto-критериям?
5. **Particle gate:** `z_particle` переносит particle-информацию без domain leakage?
6. **Condition gate:** явная адаптация интерполирует и экстраполирует на held-out
   conditions безопаснее неизменной модели?
7. **Detector gate:** малая калибровка даёт приемлемый перенос на невиденную сборку?

## Ближайшие практические шаги

1. Реализовать сильный classical PSD и multi-window benchmark на текущем pipeline.
2. Получить код/веса/labels/preprocessing VAE коллеги и воспроизвести его отдельно по
   каналам.
3. Добавить PCA, linear AE, compact AE/VAE и общий latent audit внутри `Qlong`-страт.
4. Исследовать физическую природу higher-tail Cf-кандидатов и acquisition selection.
5. Спроектировать и провести повторные background/Co/Cf и condition-sweep измерения.
6. После появления независимых runs и anchors перейти к supervised `z_particle`, CVAE и
   domain adaptation.
