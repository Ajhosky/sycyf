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

### Opis zmiennych `.env`

| Zmienna | Domyślnie | Co robi | Kiedy zwiększyć / zmniejszyć |
|---|---:|---|---|
| `EXPERIMENT_SEED` | `17` | Kontroluje losowość (inicjalizacja wag i generacja danych). | Ustaw stałą wartość do porównań; zmień seed, aby sprawdzić stabilność. |
| `TRAIN_SAMPLES_PER_CLASS` | `2000` | Liczba próbek treningowych na klasę. | Zwiększ, gdy T4/T5 są słabe; zmniejsz do szybkich testów. |
| `TRAIN_FLIP_PROBABILITY` | `0.05` | Szum bitowy w danych treningowych. | Zwiększ, aby uczyć odporności; za duża wartość może pogorszyć T1/T2. |
| `TRAIN_MAX_SHIFT` | `1` | Maksymalne przesunięcie wzorca w treningu. | Zwiększ, jeśli realne dane mają większe przesunięcia. |
| `TRAIN_EPOCHS` | `1000` | Liczba epok głównego treningu. | Zwiększ, jeśli model niedouczony; zmniejsz przy szybkim debugu. |
| `MLP_HIDDEN_DIM` | `12` | Rozmiar warstwy ukrytej MLP. | Lekko zwiększ (np. 16), gdy model za słaby; zbyt duże zwiększa ryzyko przeuczenia. |
| `MLP_LEARNING_RATE` | `0.08` | Krok uczenia. | Zmniejsz, gdy uczenie niestabilne; zwiększ ostrożnie, gdy uczenie jest zbyt wolne. |
| `VERBOSE_TRAINING` | `true` | Logowanie postępu epok. | Ustaw `false`, aby skrócić/uprościć logi. |
| `EXTRA_TRAINING_ROUNDS` | `0` | Ile dodatkowych rund hardeningu wykonać. | Zwiększ, gdy chcesz poprawić T4/T5/T7/T8. |
| `EXTRA_SAMPLES_PER_CLASS` | `1000` | Próbki/klasę w każdej rundzie hardeningu. | Zwiększ przy słabej odporności na szum. |
| `EXTRA_FLIP_PROBABILITY` | `0.12` | Szum danych w hardeningu. | Zwiększ, jeśli testy z dużym szumem są słabe. |
| `EXTRA_MAX_SHIFT` | `1` | Przesunięcie wzorców w hardeningu. | Zwiększ, gdy spodziewasz się większych przesunięć w praktyce. |
| `EXTRA_EPOCHS_PER_ROUND` | `250` | Epoki na każdą rundę hardeningu. | Zwiększ, gdy każda runda daje za mały efekt. |
| `EVAL_SAMPLES_PER_CLASS` | `250` | Rozmiar testów ramkowych T3-T5. | Zwiększ dla bardziej wiarygodnego pomiaru. |
| `EVAL_SEQUENCE_COUNT` | `500` | Liczba sekwencji w T6-T8. | Zwiększ dla stabilniejszego `voting accuracy`. |
| `FRACTIONAL_BITS` | `8` | Precyzja fixed-point przy eksporcie modelu. | Dostosuj do wymagań implementacji HDL. |

Przykład:

```env
TRAIN_SAMPLES_PER_CLASS=3000
TRAIN_EPOCHS=1500
MLP_LEARNING_RATE=0.06
EXTRA_TRAINING_ROUNDS=2
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


## Jak czytać wyniki z konsoli (interpretacja)

Po treningu `main.py` wypisuje metryki dla kilku scenariuszy:

- `T1: clean base signs` — dokładność na idealnych wzorcach 8x8 (bez szumu).
- `T2: shifted clean signs` — dokładność po przesunięciach o `TRAIN_MAX_SHIFT`.
- `T3-T5` — dokładność ramek przy coraz większym szumie (`flip_probability = 0.05, 0.10, 0.15`).
- `T6-T8` — jak wyżej, ale dla sekwencji i końcowej decyzji po majority voting.
- `Confusion matrix for T4` — szczegółowo pokazuje, które klasy model myli najczęściej.

### Co znaczą liczby

- `frame accuracy` = procent pojedynczych ramek sklasyfikowanych poprawnie.
- `voting accuracy` = procent poprawnych decyzji końcowych dla całych sekwencji (`SEQUENCE_LENGTH=6`).
- W macierzy pomyłek: **wiersz** to klasa prawdziwa, **kolumna** to klasa przewidziana.
  - Im większe wartości na przekątnej, tym lepiej.
  - Duże wartości poza przekątną oznaczają konkretne pomyłki (np. LEFT mylony z RIGHT).

### Od czego te wyniki zależą

Najmocniej od:

- jakości i wielkości treningu: `TRAIN_SAMPLES_PER_CLASS`, `TRAIN_EPOCHS`, `MLP_HIDDEN_DIM`, `MLP_LEARNING_RATE`,
- trudności danych: `TRAIN_FLIP_PROBABILITY`, `TRAIN_MAX_SHIFT`,
- dodatkowego dogrywania na trudnych danych: `EXTRA_*`,
- losowości: `EXPERIMENT_SEED` (zmienia inicjalizację i wygenerowane próbki),
- poziomu testu: T3/T6 są łatwiejsze niż T5/T8.

### Kiedy jest "dobrze", a kiedy "źle" (praktyczne progi)

To są progi orientacyjne (dla tego projektu i syntetycznych danych):

- **Bardzo dobrze**:
  - T1/T2 blisko `100%`,
  - T4 `>= 85%`,
  - T5 `>= 75%`,
  - T7 voting `>= 95%`, T8 voting `>= 90%`.
- **Akceptowalnie**:
  - T4 `75-85%`,
  - T5 `65-75%`,
  - T8 voting `80-90%`.
- **Słabo / do poprawy**:
  - T2 < `95%` (za słaba odporność na przesunięcia),
  - T4 < `75%` lub T5 < `65%`,
  - T8 voting < `80%`,
  - silne skupienie pomyłek jednej klasy w confusion matrix.

Jeśli wyniki są słabe, zwiększaj stopniowo:

1. `TRAIN_SAMPLES_PER_CLASS` (np. 2000 -> 4000),
2. `TRAIN_EPOCHS` (np. 1000 -> 1500),
3. `EXTRA_TRAINING_ROUNDS` i `EXTRA_SAMPLES_PER_CLASS`,
4. ewentualnie lekko `MLP_HIDDEN_DIM` (np. 12 -> 16).

Po każdej zmianie uruchom ponownie test i porównaj T4/T5/T7/T8 przy tym samym `EXPERIMENT_SEED`.

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
