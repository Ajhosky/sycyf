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
python main.py
```

Po uruchomieniu w katalogu `outputs/` zostaną zapisane:

- `quantized_reference_model.npz` — parametry modelu float i fixed-point,
- `mlp_params.vh` — pomocniczy plik z wagami MLP w formie stałych Verilog.

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
- `main.py` — główny eksperyment.
