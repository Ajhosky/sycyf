# Define-stage reference model

Model referencyjny dla etapu Define projektu **„Ostatni Strażnik Pustki”**.

Architektura:

```text
ramka 8x8
-> ekstrakcja cech geometrycznych
-> lekki klasyfikator MLP
-> bufor / majority voting z 6 decyzji
-> komenda ruchu robota
```

## Uruchomienie

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Po uruchomieniu w katalogu `outputs/` zostaną zapisane:

- `quantized_reference_model.npz` — parametry modelu float i fixed-point,
- `mlp_params.vh` — pomocniczy plik z wagami MLP w formie stałych Verilog.

## Jak testować program

Wszystkie testy są wykonywane przez `main.py` i drukowane w konsoli.

1. **Test bazowy** (domyślne parametry):

   ```bash
   python main.py
   ```

   Sprawdzasz sekcje:
   - `T1` i `T2` (czyste/shiftowane wzorce),
   - `T3-T5` (szum lekki -> bardzo mocny),
   - `T6-T8` (sekwencje + majority voting),
   - `Confusion matrix for T4`.

2. **Szybszy test lokalny** (mniejszy zbiór i mniej epok), np.:

   ```bash
   TRAIN_SAMPLES_PER_CLASS=500 TRAIN_EPOCHS=250 EVAL_SAMPLES_PER_CLASS=100 EVAL_SEQUENCE_COUNT=100 python main.py
   ```

3. **Test powtarzalności**: uruchamiasz 2-3 razy z tym samym `EXPERIMENT_SEED` i porównujesz metryki.

## Konfiguracja przez `.env`

Program automatycznie czyta plik `.env` (jeśli istnieje), więc nie trzeba wpisywać parametrów ręcznie w `main.py`.

Najważniejsze pola:

- `TRAIN_SAMPLES_PER_CLASS`, `TRAIN_EPOCHS`, `MLP_HIDDEN_DIM`, `MLP_LEARNING_RATE`,
- `TRAIN_FLIP_PROBABILITY`, `TRAIN_MAX_SHIFT`,
- `EVAL_SAMPLES_PER_CLASS`, `EVAL_SEQUENCE_COUNT`,
- `FRACTIONAL_BITS`.

Przykład:

```env
TRAIN_SAMPLES_PER_CLASS=3000
TRAIN_EPOCHS=1500
MLP_LEARNING_RATE=0.06
```

## Jak „bardziej przetrenować” model

Dodałem etap **hardening/fine-tuning**, który trenuje model dodatkowymi rundami na trudniejszych danych (większy szum), bez zmiany kodu.

Ustaw w `.env`:

```env
EXTRA_TRAINING_ROUNDS=3
EXTRA_SAMPLES_PER_CLASS=1500
EXTRA_FLIP_PROBABILITY=0.12
EXTRA_EPOCHS_PER_ROUND=300
```

To zwykle poprawia odporność na zakłócenia (T4/T5/T7/T8), kosztem dłuższego treningu.

## Struktura plików

- `config.py` — stałe projektu i etykiety klas,
- `patterns.py` — bazowe wzorce znaków 8x8,
- `frame_ops.py` — przesunięcia i szum impulsowy,
- `feature_extractor.py` — momenty geometryczne,
- `dataset_generator.py` — generacja zbiorów treningowych/testowych,
- `mlp_classifier.py` — ręcznie zaimplementowany MLP,
- `decision_buffer.py` — majority voting i bufor decyzji,
- `evaluator.py` — scenariusze testowe i metryki,
- `fixed_point.py` — eksport do reprezentacji fixed-point,
- `experiment_config.py` — konfiguracja uruchomienia z ENV / `.env`,
- `main.py` — główny eksperyment.
