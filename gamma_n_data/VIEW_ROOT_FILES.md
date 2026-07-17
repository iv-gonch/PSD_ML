# Просмотр файлов ROOT

В системе уже установлен CERN ROOT с Python-модулем `ROOT`, поэтому скрипты не
требуют установки дополнительных пакетов. Команды ниже запускаются из корня
проекта.

## Интерактивное окно

```bash
python3 gamma_n_data/root_browser.py gamma_n_data/call_all_252Cf/UNFILTERED/Hcompass_call_all_252Cf_20250917_122156.root
```

В левой части окна раскройте папку и дважды щёлкните по гистограмме. Для
`Data_*.root` раскройте дерево `Data`, чтобы увидеть его ветви.

## Структура и первые события в терминале

```bash
python3 gamma_n_data/root_inspect.py FILE.root
python3 gamma_n_data/root_inspect.py FILE.root --entries 5
python3 gamma_n_data/root_inspect.py FILE.root --entries 5 --branches Channel,Timestamp,Energy,EnergyShort
```

Последняя форма удобнее для больших `Data_*.root`: ветвь `Samples` содержит
осциллограмму и без необходимости её лучше не читать.

## Сохранение графиков

Готовая одномерная или двумерная гистограмма из `Hcompass*.root`:

```bash
python3 gamma_n_data/root_plot.py FILE.root \
  --object 'Energy/EnergyCH0@DT5730SB_27616' -o energy.png

python3 gamma_n_data/root_plot.py FILE.root \
  --object 'PSD_E/PSDvsECH0@DT5730SB_27616' -o psd_vs_energy.png
```

Гистограмма ветви из дерева `Data` и PSD-график, рассчитанный из ветвей:

```bash
python3 gamma_n_data/root_plot.py DATA.root \
  --expr Energy --bins 400,0,32000 -o energy.png

python3 gamma_n_data/root_plot.py DATA.root \
  --expr '(Energy-EnergyShort)/Energy:Energy' \
  --cut 'Energy>0' --bins 300,0,30000,200,0,1 -o psd.png
```

Осциллограмма конкретного события (нумерация с нуля):

```bash
python3 gamma_n_data/root_plot.py DATA.root --waveform 0 -o waveform_0.png
```

Путь объекта для `--object` можно скопировать из вывода `root_inspect.py`.
