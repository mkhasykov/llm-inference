# WORKLOG — состояние и следующие шаги

Живой лог для продолжения работы (читать после сжатия контекста). План экспериментов — в `EXPERIMENT_PLAN.md`.

## Где мы сейчас (одним абзацем)
Экспериментальная часть диплома по оптимизации инференса **полностью готова и закоммичена** на ветке `methods-study` (база — `rollback-pr2`, не `main`: от направления `main` пользователь сознательно отказался, брали оттуда только проверенные механики). Перемер 4-бит с ядрами **Marlin** **выполнен и закоммичен** (см. вывод ниже — он переворачивает прежнее «квант=только память»). Главный незакрытый кусок проекта — **текст диплома** (ещё не начат).

## Окружение (важно для запуска)
- Машина: Windows + WSL2 Ubuntu, RTX 3090 24GB. venv: `.venv` (uv). torch 2.11.0+cu128, transformers 5.8.1.
- **CUDA toolkit установлен**: `/usr/local/cuda-12.8` (nvcc 12.8) — поставили ради Marlin. В PATH глобально НЕ добавлен. Для Marlin-прогонов выставлять: `export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH`.
- Доп. зависимости в venv: `bitsandbytes 0.49`, `gptqmodel 7.0`, `optimum 2.1`, `lm-eval 0.4.12`, `matplotlib`, `pandas` (в pyproject). `hf_transfer`/`autoawq` НЕ нужны (autoawq удалён; transformers 5.8 грузит AWQ через gptqmodel).
- Модели в HF-кэше: Qwen2.5-0.5B/1.5B/3B/7B-Instruct + AWQ/GPTQ-Int4/Int8 чекпойнты 1.5B и 7B.
- **Загрузки HF капризят** на больших файлах: `HF_XET_HIGH_PERFORMANCE=1` стопорит → качать дефолтным `snapshot_download` (без env).

## Карта кода (scripts/)
- Хелперы: `data.py`, `modeling.py`, `timing.py` (CUDA-events, finish_measure / finish_measure_blocks / finish_measure_batch), `decode.py`, `quality.py` (perplexity), `summary.py` (повторы→mean±std), `runner.py`, `cli.py`, `env.py` (gpu temp/clock), `plotstyle.py`.
- Методы: `benchmark_baseline.py` (`--no-cache`), `manual_kv_loop.py`, `benchmark_quant.py` (`--quant int8/nf4/fp4/awq/gptq-int4/gptq-int8`, `--marlin`), `benchmark_spec.py` (draft + UAD при разном vocab + `--prompt-lookup-tokens`), `benchmark_batch.py` (`--batch-size`, left-pad).
- Оркестрация/анализ: `run_matrix.py` (драйвер, `--only/--skip-cells/--cell-timeout`), `aggregate.py` (таблица, джойн качества по (модель,формат)), `plots.py` (methods/quant/batch/pareto), `plots_methods.py` (vs_baseline/spec_variants/length), `check_fidelity.py`.
- Лаунчеры: `run_all.sh`, `resume_7b.sh`, `run_night.sh`, `run_batch_dense.sh`, `run_marlin.sh`.
- Общие флаги: `--limit 40 --repeats 3 --max-new-tokens 256 --fixed-length`. Результаты: `results/qwen1.5b/`, `results/qwen7b/`, `results/qwen3b/`, `results/length_sweep/{qwen1.5b,qwen7b}/`. Per-prompt `.jsonl` в gitignore; summary `.json` + `plots/*.png` коммитятся.

## Что сделано (методы × 1.5B и 7B)
Baseline(cache/nocache), manual_kv, квант (bnb int8/nf4/fp4 + AWQ/GPTQ-int4 на Triton), спек (draft 0.5B; на 7B через UAD из-за разного vocab 152064 vs 151936), батчинг (sweep {1,2,4,8,16,32}), комбо (nf4/awq + spec), качество (MMLU/GSM8K/perplexity, только 7B), length-sweep (128/256/512/1024), 3B vanilla-спек.

### Ключевые выводы (чтобы не переоткрывать)
- **KV-cache:** всегда выигрывает, эффект растёт с длиной — 7B: ×1.5(128т)→×5.0(1024т); ×2.3 на 256т.
- **Квантизация — ЗАВИСИТ ОТ ЯДРА (ключевой нюанс диплома):**
  - На **Triton/bnb** 4-бит — выигрыш только по ПАМЯТИ, не по скорости: 7B nf4 39 ≈ bf16 37 tok/s при 5.7 vs 15.3GB; на 1.5B 4-бит даже *медленнее* bf16; int8 всегда самый медленный (~10–20 tok/s, LLM.int8).
  - На **оптимизированных ядрах Marlin** 4-бит даёт И память, И скорость. **7B: AWQ/GPTQ-Marlin 60–61 tok/s против bf16 35 → ×1.7 быстрее** при 5.6 vs 15.3GB (×2.7 меньше). 1.5B: Marlin 57–58 почти догоняет bf16 64 (×1.4 к Triton), но не обгоняет — маленькая модель меньше упирается в bandwidth.
  - Вывод: «квант = только память» был **артефактом Triton-ядра**. Правильный вывод: на memory-bound декоде 7B+ 4-бит+Marlin — лучший single-stream выбор (и память, и скорость).
  - Качество (лоссы 4-бит): int8 ≈ без потерь; 4-бит −0.3 ppl / −1–2% MMLU (gptq-int4 заметно хуже по GSM8K: 0.675 vs 0.76). Marlin не меняет качество (то же AWQ/GPTQ, иное ядро).
- **Батчинг:** главный рычаг throughput, почти линейно до b32 (7B bf16 37→688, awq→650, nf4→442); 4-бит влезает в ~6 vs ~16GB.
- **Спек-декодинг:** НЕ окупается на 3090+HF для ≤7B ни при каком сетапе (1.5B/3B vanilla, 7B UAD, +квант) — overhead draft+фреймворка > экономии. prompt-lookup ≈ baseline (нет draft-оверхеда). Lossless подтверждён в fp32. Выигрыш требует EAGLE/Medusa-голов, либо target 70B+, либо vLLM.
- **Комбинации не аддитивны:** quant+spec медленнее каждого по отдельности.
- **Marlin (полный перемер, закоммичен):** см. блок «Квантизация» выше. 7B AWQ/GPTQ-Marlin 60–61 tok/s обгоняют bf16 35 (×1.7); 1.5B 57–58 почти догоняют bf16 64.

## Marlin: ВЫПОЛНЕНО
- Код: `benchmark_quant.py --marlin` (AWQ→backend `marlin`, GPTQ→`marlin`; kind получает суффикс `_marlin`, Triton-числа НЕ трогаются). `plots.py` `_quant_rows` пропускает `*_marlin` (чтобы quant.png не слипался); `plotstyle.py` подписи различают Triton/Marlin. Лаунчер `run_marlin.sh` (внутри выставлен CUDA_HOME).
- Результаты записаны: `results/{qwen1.5b,qwen7b}/quant_{awq,gptq-int4}_marlin_*.json`, таблицы+графики пере-построены. Всё закоммичено.
- Повторить при нужде: `bash scripts/run_marlin.sh` (~35 мин, ядро закэшировано → загрузка быстрая).

## Следующие шаги (по порядку)
1. ~~Завершить Marlin~~ — **сделано и закоммичено.**
2. ~~Почистить `cuda-keyring_*.deb`~~ — **удалён; `*.deb` и `results/*.log` в .gitignore.**
3. **(опц.) EAGLE для спека** — гейт: есть ли готовая EAGLE-2/3 голова под Qwen2.5-7B (искать на HF). Если есть → запускать через vLLM (не HF-харнесс; отдельная точка «SOTA-спек»). Если нет (обучать) — вне объёма.
4. **ТЕКСТ ДИПЛОМА — главный остаток.** Тема: «Методы оптимизации производительности инференса глубоких нейросетей», магистерский, новизна не нужна, упор на выводы/рекомендации. Дедлайн был 10 июня — пользователь сказал «забудь про дедлайн». Структура: введение / обзор методов (+roofline-постановка) / методика (харнесс, метрики, железо) / результаты (наши таблицы+графики) / выводы + матрица рекомендаций. Писать «под руководством пользователя».

## Открытые вопросы к пользователю
- **Формат текста: Word (.docx) или LaTeX?** и **требования вуза** (ГОСТ/шаблон/объём) — НЕ выяснены, нужны до старта текста.
- PR-workflow: ветка `methods-study` закоммичена локально, **не запушена и не влита в main**. Пушить/PR — по слову пользователя.

## Git
- Ветка `methods-study`, впереди `main`, не запушена.
- Marlin-правки закоммичены (`benchmark_quant.py`, `plots.py`, `plotstyle.py`, `run_marlin.sh`, `WORKLOG.md`, результаты+графики). Дефолт без `--marlin` не изменён (Triton остаётся).
